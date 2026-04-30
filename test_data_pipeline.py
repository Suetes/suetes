import xarray as xr
from suetes.preprocessing.processor import ERA5Processor
from suetes.vis.visualizer import ERA5Visualizer

def main():
    print("1. Initializing Processor...")
    processor = ERA5Processor(
        pl_path="suetes/data/suetes_test_run_pressure_levels.nc",
        sl_path="suetes/data/suetes_test_run_single_levels.nc"
    )

    print("2. Stitching State...")
    state = processor.get_stitched_state(time_idx=0)

    print("3. Initializing Visualizer...")
    viz = ERA5Visualizer()

    print("4. Generating Plots...")
    # Cross Section
    mid_lat_idx = len(state['latitude']) // 2
    viz.plot_cross_section(state, lat_idx=mid_lat_idx, save_path="out_cross_section.png")

    # Surface Map
    viz.plot_map(processor.ds_pl, processor.ds_sl, variable='t2m', level='surface', 
                 save_path="out_map_surface.png")

    # Topography Comparison
    ds_gebco = xr.open_dataset("suetes/data/gebco_data.nc")
    viz.plot_topography_comparison(processor.ds_sl, ds_gebco, time_idx=0, 
                                   save_path="out_topo_compare.png")

    print("Done! Check your root directory for the new plots.")

if __name__ == "__main__":
    main()