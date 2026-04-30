import os
import cdsapi

class ERA5Manager:
    def __init__(self, data_dir="suetes/data"):
        self.data_dir = data_dir
        self.client = cdsapi.Client()
        
        # Ensure the data directory exists
        os.makedirs(self.data_dir, exist_ok=True)

    def download_regional_subset(self, year, month, days, area, prefix="test_case"):
        """
        Downloads ERA5 single and pressure level data if it doesn't already exist.
        
        :param year: str, e.g., "2026"
        :param month: str, e.g., "03"
        :param days: list of str, e.g., ["01", "02", "03", ...]
        :param area: list of floats [North, West, South, East] 
                     e.g., [50, -10, 40, 10] for parts of Western Europe
        :param prefix: str, name of the file prefix for caching
        """
        
        # Define the file paths for the cache
        sl_filepath = os.path.join(self.data_dir, f"{prefix}_single_levels.nc")
        pl_filepath = os.path.join(self.data_dir, f"{prefix}_pressure_levels.nc")

        # Standard time block (every hour)
        times = [f"{str(i).zfill(2)}:00" for i in range(24)]

        # --- 1. SINGLE LEVELS ---
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
                    "area": area, # Spatially subsets the data on the server!
                    "data_format": "netcdf",
                    "download_format": "unarchived"
                },
                sl_filepath
            )

        # --- 2. PRESSURE LEVELS ---
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
                    "area": area, # Spatially subsets the data on the server!
                    "data_format": "netcdf",
                    "download_format": "unarchived"
                },
                pl_filepath
            )
            
        print("Data is ready for the model!")
        return sl_filepath, pl_filepath

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    manager = ERA5Manager(data_dir="suetes/data")
    
    # Bounding box: [North, West, South, East]
    # Make sure this box is slightly larger than your RegionalGrid3D domain
    # so the DaviesSponge has valid boundary data to relax against.
    domain_bbox = [55.0, -15.0, 35.0, 15.0] 
    
    days_to_run = [str(i).zfill(2) for i in range(1, 5)] # ["01", "02", ..., "10"]

    sl_file, pl_file = manager.download_regional_subset(
        year="2026",
        month="03",
        days=days_to_run,
        area=domain_bbox,
        prefix="suetes_test_run"
    )