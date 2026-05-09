import os
import math
import cdsapi

class ERA5Manager:
    def __init__(self, data_dir="suetes/data"):
        self.data_dir = data_dir
        self.client = cdsapi.Client()
        
        # Ensure the data directory exists
        os.makedirs(self.data_dir, exist_ok=True)

    @staticmethod
    def calculate_required_bbox(lat_c, lon_c, nx, ny, dx, dy, buffer_deg=2.0):
        """
        Calculates the minimal ERA5 bounding box required to cover the physical grid.
        Returns: [North, West, South, East] in degrees.
        """
        R = 6371000.0  # Earth radius in meters
        
        # Physical distance from the center to the edge
        half_Lx = (nx * dx) / 2.0
        half_Ly = (ny * dy) / 2.0
        
        # Convert meters to degrees latitude (constant scaling)
        delta_lat = math.degrees(half_Ly / R)
        
        # Convert meters to degrees longitude
        # Evaluate at the highest absolute latitude in the domain to ensure 
        # the box is wide enough where the meridians converge the most.
        max_abs_lat = min(89.0, abs(lat_c) + delta_lat)
        delta_lon = math.degrees(half_Lx / (R * math.cos(math.radians(max_abs_lat))))
        
        # Construct [North, West, South, East] with the safety buffer
        return [
            round(lat_c + delta_lat + buffer_deg, 2),  # North
            round(lon_c - delta_lon - buffer_deg, 2),  # West
            round(lat_c - delta_lat - buffer_deg, 2),  # South
            round(lon_c + delta_lon + buffer_deg, 2)   # East
        ]

    def download_regional_subset(self, year, month, days, area, prefix="test_case"):
        """
        Downloads ERA5 single and pressure level data if it doesn't already exist.
        
        :param year: str, e.g., "2026"
        :param month: str, e.g., "03"
        :param days: list of str, e.g., ["01", "02", "03", ...]
        :param area: list of floats [North, West, South, East] 
        :param prefix: str, name of the file prefix for caching
        """
        
        # Define the file paths for the cache
        sl_filepath = os.path.join(self.data_dir, f"{prefix}_single_levels.nc")
        pl_filepath = os.path.join(self.data_dir, f"{prefix}_pressure_levels.nc")

        # Standard time block (every hour)
        times = [f"{str(i).zfill(2)}:00" for i in range(24)]

        # SINGLE LEVELS
        if os.path.exists(sl_filepath):
            print(f"[CACHE] Single levels already exist at {sl_filepath}. Skipping download.")
        else:
            print(f"[DOWNLOAD] Fetching Single Levels for {year}-{month}...")
            self.client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": ["reanalysis"],
                    "variable": [
                        "10m_u_component_of_wind", "10m_v_component_of_wind",
                        "2m_dewpoint_temperature", "2m_temperature",
                        "surface_pressure", "geopotential"
                    ],
                    "year": [year],
                    "month": [month],
                    "day": days,
                    "time": times,
                    "area": area, 
                    "data_format": "netcdf",
                    "download_format": "unarchived"
                },
                sl_filepath
            )

        # PRESSURE LEVELS
        if os.path.exists(pl_filepath):
            print(f"[CACHE] Pressure levels already exist at {pl_filepath}. Skipping download.")
        else:
            print(f"[DOWNLOAD] Fetching Pressure Levels for {year}-{month}...")
            self.client.retrieve(
                "reanalysis-era5-pressure-levels",
                {
                    "product_type": ["reanalysis"],
                    "variable": [
                        "geopotential", "specific_humidity", "temperature",
                        "u_component_of_wind", "v_component_of_wind", "vertical_velocity"
                    ],
                    "pressure_level": [
                        "100", "200", "300", "400", "500", "600",
                        "700", "800", "850", "900", "925", "950",
                        "975", "1000"
                    ],
                    "year": [year],
                    "month": [month],
                    "day": days,
                    "time": times,
                    "area": area, 
                    "data_format": "netcdf",
                    "download_format": "unarchived"
                },
                pl_filepath
            )
            
        print("Data is ready for the model!")
        return sl_filepath, pl_filepath