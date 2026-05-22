"""
Automated Data Acquisition Module.

Interfaces with the Copernicus Climate Data Store (CDS) API to calculate
required geographic bounding boxes and download ERA5 reanalysis data for
model initialization and lateral boundary forcing.

Pressure-level coverage is preset-driven. See PRESSURE_LEVEL_PRESETS for the
options and the rationale behind each.
"""

import os
import math
import cdsapi


# ---------------------------------------------------------------------------
# Pressure-level presets.
#
# ERA5 provides 37 native pressure levels (hPa):
#   1, 2, 3, 5, 7, 10, 20, 30, 50, 70,
#   100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600,
#   650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000.
#
# Selection guide for a NAM-22 (0.22 deg, ~22 km) regional setup:
#
#   * "minimal" (14 levels, 100-1000 hPa): only safe if the model lid is
#     well below 100 hPa (say, below 150 hPa). Otherwise the top driving
#     level coincides with the lid and the LBC is extrapolated.
#
#   * "buffered" (17 levels, 30-1000 hPa): appropriate for the current
#     run_simulation_12km.py configuration (nz=32, dz=500 m, lid ~16 km
#     ~ 100 hPa). Provides three driving levels (70, 50, 30 hPa) above
#     the lid for clean interpolation.
#
#   * "full" (37 levels, 1-1000 hPa): required if the model lid is pushed
#     up to ~10 hPa to match the CRCM5 NAM-22 reference (56 hybrid levels,
#     top at 10 hPa) or the CRCM6/GEM5.0 NAM-11 reference (71 hybrid
#     levels, top at 10 hPa; Roberge et al. 2024).
# ---------------------------------------------------------------------------
PRESSURE_LEVEL_PRESETS = {
    "minimal": [
        "100", "200", "300", "400", "500", "600",
        "700", "800", "850", "900", "925", "950", "975", "1000",
    ],
    "buffered": [
        "30", "50", "70",
        "100", "150", "200", "250", "300", "400", "500", "600",
        "700", "800", "850", "900", "950", "1000",
    ],
    "full": [
        "1", "2", "3", "5", "7", "10", "20", "30", "50", "70",
        "100", "125", "150", "175", "200", "225", "250", "300", "350", "400",
        "450", "500", "550", "600", "650", "700", "750", "775", "800", "825",
        "850", "875", "900", "925", "950", "975", "1000",
    ],
}


class ERA5Manager:
    """
    Manages the downloading and local caching of ERA5 NetCDF files.

    Parameters
    ----------
    data_dir : str
        Directory for cached NetCDF files.
    pressure_levels : str or list of str
        Either a preset key in PRESSURE_LEVEL_PRESETS ("minimal", "buffered",
        "full") or an explicit list of hPa strings. Default "buffered" is
        appropriate for a NAM-22 run with a ~100 hPa model lid.
    """

    def __init__(self, data_dir="/Data/whittaker/suetes/data",
                 pressure_levels="buffered"):
        self.data_dir = data_dir
        self.client = cdsapi.Client()

        if isinstance(pressure_levels, str):
            if pressure_levels not in PRESSURE_LEVEL_PRESETS:
                raise ValueError(
                    f"Unknown pressure-level preset '{pressure_levels}'. "
                    f"Choose one of {list(PRESSURE_LEVEL_PRESETS)} "
                    "or pass an explicit list of hPa strings."
                )
            self.pressure_levels = PRESSURE_LEVEL_PRESETS[pressure_levels]
            self._preset_name = pressure_levels
        else:
            self.pressure_levels = [str(p) for p in pressure_levels]
            self._preset_name = "custom"

        os.makedirs(self.data_dir, exist_ok=True)

    @staticmethod
    def calculate_required_bbox(lat_c, lon_c, nx, ny, dx, dy, buffer_deg=2.0):
        r"""
        Compute the geographic bounding box enclosing the physical grid.

        Because meridians converge at the poles, the longitudinal extent is
        evaluated at the highest absolute latitude reached by the domain:

        $$ \Delta \lambda = \frac{L_x / 2}{R \cos(|\phi_{max}|)}
                            \times \frac{180}{\pi}. $$

        Returns
        -------
        list
            [North, West, South, East] in degrees, formatted for the CDS API.
        """
        R = 6371000.0
        half_Lx = (nx * dx) / 2.0
        half_Ly = (ny * dy) / 2.0

        delta_lat = math.degrees(half_Ly / R)
        max_abs_lat = min(89.0, abs(lat_c) + delta_lat)
        delta_lon = math.degrees(half_Lx / (R * math.cos(math.radians(max_abs_lat))))

        return [
            round(lat_c + delta_lat + buffer_deg, 2),
            round(lon_c - delta_lon - buffer_deg, 2),
            round(lat_c - delta_lat - buffer_deg, 2),
            round(lon_c + delta_lon + buffer_deg, 2),
        ]

    def download_regional_subset(self, year, month, days, area,
                                 prefix="test_case"):
        """
        Download ERA5 single and pressure level data if not already cached.

        Returns
        -------
        (single_level_filepath, pressure_level_filepath) : (str, str)
        """
        sl_filepath = os.path.join(self.data_dir, f"{prefix}_single_levels.nc")
        pl_filepath = os.path.join(self.data_dir, f"{prefix}_pressure_levels.nc")
        times = [f"{str(i).zfill(2)}:00" for i in range(24)]

        if os.path.exists(sl_filepath):
            print(f"[DATA] Single levels already exist at {sl_filepath}. "
                  "Skipping download.")
        else:
            print(f"[DATA] Fetching Single Levels for {year}-{month}...")
            self.client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": ["reanalysis"],
                    "variable": [
                        "10m_u_component_of_wind", "10m_v_component_of_wind",
                        "2m_dewpoint_temperature", "2m_temperature",
                        "surface_pressure", "geopotential",
                        "land_sea_mask", "skin_temperature",
                    ],
                    "year": [year],
                    "month": [month],
                    "day": days,
                    "time": times,
                    "area": area,
                    "data_format": "netcdf",
                    "download_format": "unarchived",
                },
                sl_filepath,
            )

        if os.path.exists(pl_filepath):
            print(f"[DATA] Pressure levels already exist at {pl_filepath}. "
                  "Skipping download.")
        else:
            print(f"[DATA] Fetching Pressure Levels for {year}-{month} "
                  f"(preset='{self._preset_name}', "
                  f"{len(self.pressure_levels)} levels: "
                  f"{self.pressure_levels[0]} to "
                  f"{self.pressure_levels[-1]} hPa)...")
            self.client.retrieve(
                "reanalysis-era5-pressure-levels",
                {
                    "product_type": ["reanalysis"],
                    "variable": [
                        "geopotential", "specific_humidity", "temperature",
                        "u_component_of_wind", "v_component_of_wind",
                        "vertical_velocity",
                    ],
                    "pressure_level": self.pressure_levels,
                    "year": [year],
                    "month": [month],
                    "day": days,
                    "time": times,
                    "area": area,
                    "data_format": "netcdf",
                    "download_format": "unarchived",
                },
                pl_filepath,
            )

        print("[DATA] Data is ready for the model!")
        return sl_filepath, pl_filepath
