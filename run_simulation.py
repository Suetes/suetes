import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress all but FATAL CUDA/XLA warnings

import jax
import jax.numpy as jnp
import time
import numpy as np

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.steppers import SISLStepper3D
from suetes.regional3d.boundaries import DaviesSponge
from suetes.shared.transforms import SleveSimple

from suetes.preprocessing.processor import ERA5Processor
from suetes.preprocessing.topography import TopographyProcessor
from suetes.preprocessing.era2suetes import BoundaryProcessor, TimeManager
from suetes.preprocessing.era5downloader import ERA5Manager # Make sure this import path matches your setup

from suetes.vis.visualizer import Visualizer

# ==========================================
# 0. DOMAIN PRESETS
# ==========================================
DOMAINS = {
    "labrador_sea": {
        "lat_c": 48.0, "lon_c": -60.0,
        "bbox": [65.0, -80.0, 30.0, -40.0]  # N, W, S, E
    },
    "bonavista_peninsula": {
        "lat_c": 48.5, "lon_c": -53.5,
        "bbox": [60.0, -65.0, 35.0, -40.0]
    },
    "alps": {
        "lat_c": 45.0, "lon_c": 5.0,
        "bbox": [55.0, -15.0, 35.0, 15.0]
    }
}

def main():
    # ==========================================
    # 1. PIPELINE ORCHESTRATION & SETUP
    # ==========================================
    
    # --- Select your region here ---
    ACTIVE_DOMAIN = "alps" 
    cfg = DOMAINS[ACTIVE_DOMAIN]

    output_dir = "suetes/plots"
    os.makedirs(output_dir, exist_ok=True)

    nx, ny, nz = 300, 300, 40
    dx, dy, dz = 6000.0, 6000.0, 500.0
    sponge_depth = 30
    
    dt = 30.0 # timestep (in seconds)
    sim_hours = 3
    sim_time_seconds = sim_hours * 3600.0
    num_steps = int(sim_time_seconds / dt)
    num_era5_states = int(sim_hours) + 1 

    constants = {'g': 9.81, 'Rd': 287.0, 'cp': 1004.0, 'cvd': 717.0, 'p0': 100000.0, 'epsilon': 0.622}
    USE_MOISTURE = True 

    # ==========================================
    # 2. DATA ACQUISITION
    # ==========================================
    print(f"Checking ERA5 Data for region: {ACTIVE_DOMAIN.upper()}...")
    manager = ERA5Manager(data_dir="suetes/data")
    
    # Adjust dates as needed for your specific test case
    days_to_run = [str(i).zfill(2) for i in range(1, 5)] 
    
    sl_file, pl_file = manager.download_regional_subset(
        year="2026",
        month="05",
        days=days_to_run,
        area=cfg["bbox"],
        prefix=ACTIVE_DOMAIN
    )

    # ==========================================
    # 3. GEOMETRY & TOPOGRAPHY
    # ==========================================
    print(f"Initializing Suetes Domain ({nx}x{ny}x{nz}) at {dx}m resolution...")
    base_grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, cfg["lat_c"], cfg["lon_c"])
    
    topo_proc = TopographyProcessor(
        era5_sl_path=sl_file,
        gebco_path="suetes/data/gebco_data.nc"
    )
    h_func = topo_proc.process_and_blend(base_grid, sponge_depth=sponge_depth, smooth_sigma=2.0)  
    sleve_transform = SleveSimple(scale_s=10000.0, n=1.0)
    grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, cfg["lat_c"], cfg["lon_c"], h_func=h_func, transform=sleve_transform)

    # ==========================================
    # 4. ERA5 BOUNDARY PROCESSING
    # ==========================================
    print(f"Processing ERA5 Boundaries for a {sim_hours}-hour simulation...")
    era5_proc = ERA5Processor(pl_path=pl_file, sl_path=sl_file)
    
    suetes_bc_states = []
    times_sec = []
    
    raw_t0 = era5_proc.get_stitched_state(time_idx=0)
    bridge = BoundaryProcessor(grid, raw_t0['latitude'], raw_t0['longitude'], constants)
    
    for i in range(num_era5_states):
        print(f"  -> Regridding ERA5 state for T={i}h")
        raw_state = era5_proc.get_stitched_state(time_idx=i)
        bc_state = bridge.process(raw_state)
        
        suetes_bc_states.append(bc_state)
        times_sec.append(float(i * 3600.0))
        
    time_manager = TimeManager(suetes_bc_states, times_sec, grid)
    initial_state = suetes_bc_states[0]
    initial_state['q_c'] = jnp.zeros_like(initial_state['q'])

    # ==========================================
    # 5. PHYSICS & STEPPER INITIALIZATION
    # ==========================================
    print("Initializing dynamical core...")
    operators = CGridOperator3D(grid)
    physics = Euler3D(grid, operators, constants, dt=dt, 
                       initial_era5_state=initial_state, damp_height=9000.0, max_damp=3.0, 
                       nu_div_factor=0.8, nu_h_factor=0.1)
    
    stepper = SISLStepper3D(physics, dt, tracer_keys=['q', 'q_c'], use_moisture=USE_MOISTURE)
    sponge = DaviesSponge(grid, sponge_depth=sponge_depth, dt=dt, tau_bndy_factor=10.0)

    # ==========================================
    # 6. THE INTEGRATION LOOP
    # ==========================================
    print(f"Starting integration: {num_steps} steps (dt={dt}s).")
    
    def scan_fn(curr_state, step_idx):
        t_curr = step_idx * dt
        bc_state_t = time_manager.get_forcing(t_curr)
        
        def bc_fn(state_next, _):
            return sponge.blend(state_next, bc_state_t)
            
        next_state = stepper.step(curr_state, t_curr, forcing=None, bc_fn=bc_fn)
        max_w = jnp.max(jnp.abs(next_state['w']))
        
        return next_state, max_w 

    start_time = time.time()
    final_state, final_max_w_array = jax.lax.scan(scan_fn, initial_state, jnp.arange(num_steps))
    jax.block_until_ready(final_state['u']) 
    
    print(f"Integration complete! Wall time: {time.time() - start_time:.2f} seconds.")

    # ==========================================
    # 7. VISUALIZE THE RESULTS
    # ==========================================
    print("Generating comparison plots...")
    visualizer = Visualizer()
    final_era5_state = suetes_bc_states[-1]
    
    # We pass ACTIVE_DOMAIN directly into the filename strings
    visualizer.plot_comparison(grid, final_state, final_era5_state, 'th_v', z_idx=5, sponge_depth=sponge_depth, 
                               save_path=os.path.join(output_dir, f"{ACTIVE_DOMAIN}_compare_th_v_{sim_hours}h.png"))

    visualizer.plot_comparison(grid, final_state, final_era5_state, 'u', z_idx=5, sponge_depth=sponge_depth, 
                               save_path=os.path.join(output_dir, f"{ACTIVE_DOMAIN}_compare_u_{sim_hours}h.png"))

    visualizer.plot_comparison(grid, final_state, final_era5_state, 'v', z_idx=5, sponge_depth=sponge_depth, 
                               save_path=os.path.join(output_dir, f"{ACTIVE_DOMAIN}_compare_v_{sim_hours}h.png"))

    visualizer.plot_energy_spectrum(grid, final_state, 'w', z_idx=5, sponge_depth=sponge_depth, 
                                    save_path=os.path.join(output_dir, f"{ACTIVE_DOMAIN}_energy_{sim_hours}h.png"))

    fields_to_plot = [
        {'var': 'w', 'cmap': 'seismic', 'title': 'Vertical Velocity [m/s]', 'scale': 'sym'},
        {'var': 'div', 'cmap': 'seismic', 'title': 'Horizontal Divergence [s⁻¹]', 'scale': 'sym'},
        {'var': 'u', 'cmap': 'seismic', 'title': 'Zonal Wind [m/s]', 'scale': 'sym'},
        {'var': 'v', 'cmap': 'seismic', 'title': 'Meridional Wind [m/s]', 'scale': 'sym'}
    ]
    visualizer.plot_dashboard(grid, final_state, z_idx=5, sponge_depth=sponge_depth, fields=fields_to_plot, 
                              time_hours=sim_hours, save_path=os.path.join(output_dir, f"{ACTIVE_DOMAIN}_dash_{sim_hours}h.png"))

    # Slices
    mid_y = grid.ny // 2 
    visualizer.plot_cross_section(grid, final_state, 'w', y_idx=mid_y, sponge_depth=sponge_depth, 
                                  save_path=os.path.join(output_dir, f"{ACTIVE_DOMAIN}_slice_w_{sim_hours}h.png"))
                                  
    visualizer.plot_cross_section(grid, final_state, 'q_c', y_idx=mid_y, sponge_depth=sponge_depth, 
                                  save_path=os.path.join(output_dir, f"{ACTIVE_DOMAIN}_slice_qc_{sim_hours}h.png"))

if __name__ == "__main__":
    main()