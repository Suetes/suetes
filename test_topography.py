from suetes.preprocessing.topography import TopographyProcessor
from suetes.regional3d.geometry import RegionalGrid3D

def main():
    # 1. Base parameters
    nx, ny, nz = 100, 100, 40
    dx, dy, dz = 2000.0, 2000.0, 500.0  # e.g., 2km resolution
    lat_c, lon_c = 45.0, 5.0 # Center of your Oblique Stereographic map
    sponge_depth = 10

    # 2. Build a "Dummy" Grid just to establish Cartesian Coordinates
    print("Initializing base geometry...")
    base_grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_c, lon_c)

    # 3. Process Topography
    print("Processing Topography...")
    topo_proc = TopographyProcessor(
        era5_sl_path="suetes/data/suetes_test_run_single_levels.nc",
        gebco_path="suetes/data/gebco_data.nc"
    )
    # Pass the base_grid so the processor knows the target (x, y) coordinates
    h_func_jax = topo_proc.process_and_blend(base_grid, sponge_depth=sponge_depth)

    # 4. Rebuild the TRUE Grid with the 3D terrain function applied!
    print("Building final 3D terrain-following grid...")
    grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_c, lon_c, h_func=h_func_jax)

    # 5. Now you are safe to run era2suetes.py!
    # processor = ERA5Processor(...)
    # suetes_state = processor.process_era5_slice(regridded_state, grid)

if __name__ == "__main__":
    main()
