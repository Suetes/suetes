import numpy as np
import matplotlib.pyplot as plt
from suetes.regional3d.geometry import RegionalGrid3D
from suetes.preprocessing.processor import ERA5Processor
from suetes.preprocessing.topography import TopographyProcessor
from suetes.preprocessing.era2suetes import BoundaryProcessor

def main():
    # 1. Base Setup (600x600 km domain)
    nx, ny, nz = 300, 300, 40
    dx, dy, dz = 2000.0, 2000.0, 500.0
    lat_c, lon_c = 45.0, 5.0
    sponge_depth = 15

    print("1. Initializing Topography and Grid...")
    base_grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_c, lon_c)
    
    topo_proc = TopographyProcessor(
        era5_sl_path="suetes/data/suetes_test_run_single_levels.nc",
        gebco_path="suetes/data/gebco_data.nc"
    )
    h_func = topo_proc.process_and_blend(base_grid, sponge_depth=sponge_depth)
    
    # Rebuild the final grid with high-res 3D terrain
    grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_c, lon_c, h_func=h_func)

    # 2. Process ERA5 Raw Data
    print("2. Stitching ERA5 Data...")
    era5_proc = ERA5Processor(
        pl_path="suetes/data/suetes_test_run_pressure_levels.nc",
        sl_path="suetes/data/suetes_test_run_single_levels.nc"
    )
    stitched_state = era5_proc.get_stitched_state(time_idx=0)

    # 3. The Bridge (ERA5 -> Suetes)
    print("3. Running Boundary Processor (Horizontal & Vertical Interpolation)...")
    constants = {
        'g': 9.81, 'Rd': 287.0, 'cp': 1004.0, 'p0': 100000.0, 'epsilon': 0.622
    }
    bridge = BoundaryProcessor(grid, stitched_state['latitude'], stitched_state['longitude'], constants)
    
    # BOOM. This is your model-ready state!
    suetes_state = bridge.process(stitched_state)

    # 4. Verification Plot
    print("4. Generating Verification Plot...")
    
    mid_y_idx = ny // 2
    x_coords = grid.x_m / 1000.0  
    
    # Get the geometric height (Z_m) and Virtual Potential Temp (th_v) slices
    Z_slice = grid.Z_m[:, mid_y_idx, :]
    th_v_slice = suetes_state['th_v'][:, mid_y_idx, :]
    
    # Get true surface topography from the W-grid (bottom interface)
    terrain_z = grid.Z_w[:, mid_y_idx, 0]

    # --- THE VISUAL FIX ---
    # Pad the data down to the exact terrain surface so there is no visual gap
    Z_slice_padded = np.concatenate([terrain_z[:, None], Z_slice], axis=1)
    th_v_slice_padded = np.concatenate([th_v_slice[:, 0:1], th_v_slice], axis=1)
    X_2d_padded = np.broadcast_to(x_coords[:, None], Z_slice_padded.shape)

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot using the padded arrays!
    contour = ax.contourf(X_2d_padded, Z_slice_padded, th_v_slice_padded, levels=30, cmap='inferno')
    plt.colorbar(contour, label='Virtual Potential Temp (th_v) [K]')
    
    ax.fill_between(x_coords, 0, terrain_z, color='dimgray', label='Suetes Topography (GEBCO+ERA5)')

    ax.set_ylim(0, 15000) 
    ax.set_title(f"Suetes Model State (Y-Index: {mid_y_idx}) - Interpolated from ERA5")
    ax.set_xlabel("X Distance from Domain Center [km]")
    ax.set_ylabel("Geometric Height [m]")
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig("out_suetes_interpolated_state.png", dpi=200)
    print("Success! Check out_suetes_interpolated_state.png")

if __name__ == "__main__":
    main()