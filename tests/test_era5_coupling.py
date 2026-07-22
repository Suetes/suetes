import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.preprocessing.processor import ERA5Processor
from suetes.preprocessing.topography import TopographyProcessor
from suetes.preprocessing.era2suetes import BoundaryProcessor
from suetes.vis.visualizer import Visualizer

import pytest

required_files = [
    "inputs/suetes_test_run_pressure_levels.nc",
    "inputs/suetes_test_run_single_levels.nc",
    "inputs/gebco_data.nc"
]
missing_files = [f for f in required_files if not os.path.exists(f)]
pytestmark = pytest.mark.skipif(
    bool(missing_files),
    reason=f"Missing required test data files: {missing_files}"
)

def get_test_env():
    # Downscaled slightly from production (150x150) for faster CI testing
    nx, ny, nz = 150, 150, 40 
    dx, dy, dz = 6000.0, 6000.0, 500.0
    lat_c, lon_c = 45.0, 5.0
    constants = {'g': 9.81, 'Rd': 287.0, 'cp': 1004.0, 'cvd': 717.0, 'p0': 100000.0, 'epsilon': 0.622}
    
    base_grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_c, lon_c)
    return constants, base_grid

def test_1_era5_stitching():
    print("\n--- 1. ERA5 RAW DATA STITCHING ---")
    processor = ERA5Processor(
        pl_path="inputs/suetes_test_run_pressure_levels.nc",
        sl_path="inputs/suetes_test_run_single_levels.nc"
    )
    state = processor.get_stitched_state(time_idx=0)
    
    # Assert critical fields were successfully merged
    required_keys = ['u', 'v', 'T', 'p', 'q', 'geopotential', 'omega']
    for k in required_keys:
        assert k in state, f"Missing variable '{k}' in ERA5 stitched state!"
        assert not np.isnan(state[k]).any(), f"NaNs detected in raw ERA5 field '{k}'!"
        
    print("STATUS: SUCCESS (ERA5 variables loaded and stitched cleanly)")
    return processor, state

def test_2_topography_blending(base_grid):
    print("\n--- 2. GEBCO + ERA5 TOPOGRAPHY BLENDING ---")
    topo_proc = TopographyProcessor(
        era5_sl_path="inputs/suetes_test_run_single_levels.nc",
        gebco_path="inputs/gebco_data.nc"
    )
    # Blend topography with sponge smoothing
    h_func = topo_proc.process_and_blend(base_grid, sponge_depth=20, smooth_sigma=2.0)
    
    # Rebuild the 3D grid with the actual blended topography
    grid_3d = RegionalGrid3D(base_grid.nx, base_grid.ny, base_grid.nz, 
                             base_grid.dx, base_grid.dy, base_grid.dz, 
                             45.0, 5.0, h_func=h_func)
                             
    # Test if the blended h_func causes severe grid tangling
    dz_min = float(jnp.min(grid_3d.dz_m_full))
    print(f"Minimum physical layer thickness (dz): {dz_min:.2f} m")
    
    assert dz_min > 5.0, "Topography blending created gradients that are too steep, causing grid tangling!"
    print("STATUS: SUCCESS (Topography blended and 3D grid metrics are safe)")
    return grid_3d

def test_3_boundary_processor(grid, raw_state, constants):
    print("\n--- 3. BOUNDARY PROCESSOR (THERMODYNAMICS & KINEMATICS) ---")
    bridge = BoundaryProcessor(grid, raw_state['latitude'], raw_state['longitude'], constants)
    suetes_state = bridge.process(raw_state)
    
    # 1. Check Top Rigid Lid Enforcement
    max_w_top = float(jnp.max(jnp.abs(suetes_state['w'][:, :, -1])))
    assert max_w_top < 1e-10, "Rigid lid boundary condition (w=0 at TOA) failed during processing!"
    
    # 2. Check Thermodynamic Reconstruction Validity (Top-Down integration)
    assert jnp.all(jnp.isfinite(suetes_state['rho'])), "NaNs detected in density reconstruction!"
    assert float(jnp.min(suetes_state['rho'])) > 0.05, "Unphysical negative or near-zero density reconstructed!"
    
    print("STATUS: SUCCESS (State interpolated, top-down hydrostatics sound, kinematic boundaries enforced)")
    return suetes_state

def test_4_cold_start_divergence(grid, suetes_state):
    print("\n--- 4. INITIAL DIVERGENCE (COLD START) ---")
    op = CGridOperator3D(grid)
    
    # Calculate horizontal divergence on the C-Grid directly
    u, v = suetes_state['u'], suetes_state['v']
    div_x = op.diff(u, axis=0, from_loc='u', to_loc='m')
    div_y = op.diff(v, axis=1, from_loc='v', to_loc='m')
    div_h = div_x + div_y
    
    max_div = float(jnp.max(jnp.abs(div_h)))
    mean_div = float(jnp.mean(jnp.abs(div_h)))
    
    print(f"Max Horizontal Divergence:  {max_div:.4e} s^-1")
    print(f"Mean Horizontal Divergence: {mean_div:.4e} s^-1")
    
    assert max_div < 1e-3, "The regridded wind field is violently divergent! Check rotation matrices."
    print("STATUS: SUCCESS (Wind field rotation and interpolation preserves continuity)")

def test_5_visualizations(processor, raw_state, grid):
    print("\n--- 5. GENERATING VISUALIZATIONS ---")
    viz = Visualizer()

    # 1. Stitched ERA5 Data (Surface Map of t2m)
    print("Plotting ERA5 Surface Map (t2m)...")
    viz.plot_map(processor.ds_pl, processor.ds_sl, variable='t2m', level='surface', 
                 save_path="out_map_surface.png")

    # 2. Topography Comparison (ERA5 vs GEBCO)
    print("Plotting Topography Comparison...")
    ds_gebco = xr.open_dataset("inputs/gebco_data.nc")
    viz.plot_topography_comparison(processor.ds_sl, ds_gebco, time_idx=0, 
                                   save_path="out_topo_compare.png")

    # 3. ERA5 Vertical Cross-Section (Stitched Data)
    # Visualizes the raw thermal structure before any regridding occurs.
    print("Plotting Stitched ERA5 Vertical Cross-Section...")
    mid_lat_idx = len(raw_state['latitude']) // 2
    viz.plot_cross_section(raw_state, lat_idx=mid_lat_idx, save_path="out_era5_cross_section.png")

    # 4. Grid Comparison (Fixed Aspect Ratio & Lon Wrap)
    print("Plotting Grid Resolution Comparison...")
    
    Xi_m, Yi_m = np.meshgrid(np.array(grid.x_m), np.array(grid.y_m), indexing='ij')
    suetes_h = grid.Z_w[:, :, 0] # Discrete bottom boundary
    suetes_lat, suetes_lon = grid.proj.get_lat_lon(Xi_m, Yi_m)
    
    # FIX: Normalize all longitudes to [-180, 180] to prevent the "around-the-world" extent bug
    suetes_lon = (suetes_lon + 180) % 360 - 180

    time_dim = 'valid_time' if 'valid_time' in processor.ds_sl.dims else 'time'
    era5_h = processor.ds_sl['z'].isel({time_dim: 0}).values / 9.81
    era5_lat = processor.ds_sl.latitude.values
    era5_lon = (processor.ds_sl.longitude.values + 180) % 360 - 180

    # Use a vertical stack to give the regional maps more vertical breathing room
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), 
                                   subplot_kw={'projection': ccrs.PlateCarree()})

    vmin, vmax = 0, 3000
    extent = [float(suetes_lon.min()), float(suetes_lon.max()), 
              float(suetes_lat.min()), float(suetes_lat.max())]

    for ax in [ax1, ax2]:
        ax.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)
        # Apply the normalized extent
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    # ERA5 plot
    im1 = ax1.pcolormesh(era5_lon, era5_lat, era5_h, transform=ccrs.PlateCarree(),
                         cmap='terrain', vmin=vmin, vmax=vmax, edgecolors='black', linewidth=0.1)
    ax1.set_title("ERA5 Native Grid (~31km resolution)", fontsize=13, pad=10)

    # Suetes plot
    im2 = ax2.pcolormesh(suetes_lon, suetes_lat, suetes_h, transform=ccrs.PlateCarree(),
                         cmap='terrain', vmin=vmin, vmax=vmax, shading='auto') 

    # Overlay grid lines (every 10th cell)
    ld = 10 
    ax2.pcolormesh(suetes_lon[::ld, ::ld], suetes_lat[::ld, ::ld], suetes_h[::ld, ::ld],
                   transform=ccrs.PlateCarree(), facecolor='none', 
                   edgecolors='black', linewidth=0.3, alpha=0.3)

    ax2.set_title(f"Suetes Model Grid ({int(grid.dx/1000)}km resolution)", fontsize=13, pad=10)

    # Place a horizontal colorbar nicely at the bottom
    plt.subplots_adjust(hspace=0.3, bottom=0.15)
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.02])
    cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Elevation [m]', fontsize=11)

    plt.savefig("out_grid_comparison.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("STATUS: SUCCESS")

if __name__ == "__main__":
    print("==================================================")
    print("       ERA5 DATA COUPLING TEST SUITE              ")
    print("==================================================")
    
    # Ensure data files exist before running
    required_files = [
        "inputs/suetes_test_run_pressure_levels.nc",
        "inputs/suetes_test_run_single_levels.nc",
        "inputs/gebco_data.nc"
    ]
    for f in required_files:
        assert os.path.exists(f), f"Required test data missing: {f}"

    constants, base_grid = get_test_env()
    
    processor, raw_state = test_1_era5_stitching()
    grid = test_2_topography_blending(base_grid)
    suetes_state = test_3_boundary_processor(grid, raw_state, constants)
    test_4_cold_start_divergence(grid, suetes_state)
    test_5_visualizations(processor, raw_state, grid)
    
    print("\n==================================================")
    print("All ERA5 data pipeline and boundary coupling tests completed successfully!")
    print("==================================================")