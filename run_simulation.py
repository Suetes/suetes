import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress all but FATAL CUDA/XLA warnings

import jax
import jax.numpy as jnp
import time
import numpy as np

from suetes.preprocessing.era5downloader import ERA5Manager
from suetes.preprocessing.processor import ERA5Processor
from suetes.preprocessing.topography import TopographyProcessor
from suetes.preprocessing.era2suetes import BoundaryProcessor, TimeManager

from suetes.shared.transforms import SleveSimple
from suetes.shared.driver import Simulation

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.steppers import SISLStepper3D
from suetes.regional3d.boundaries import DaviesSponge
from suetes.regional3d.physics import PhysicsSuite, SmagorinskyLillySGS, BulkAerodynamicPBL, SimpleMicrophysics

from suetes.vis.visualizer import Visualizer

# ==========================================
# 0. DOMAIN PRESETS
# ==========================================
DOMAINS = {
    "labrador_sea": {
        "lat_c": 48.0, "lon_c": -60.0
    },
    "alps": {
        "lat_c": 45.0, "lon_c": 5.0
    },
    "nz_south_island": {
        "lat_c": -43.5, "lon_c": 170.5
    },
    "western_canada": {
        "lat_c": 50.0, "lon_c": -120.0
    }
}

def main():
    # ==========================================
    # 1. PIPELINE ORCHESTRATION & SETUP
    # ==========================================
    
    # --- Select your region here ---
    ACTIVE_DOMAIN = "labrador_sea" 
    cfg = DOMAINS[ACTIVE_DOMAIN]

    output_dir = "suetes/plots"
    os.makedirs(output_dir, exist_ok=True)

    nx, ny, nz = 300, 300, 40
    dx, dy, dz = 6000.0, 6000.0, 500.0
    sponge_depth = 30
    
    dt = 30.0 # timestep (in seconds)
    sim_hours = 6
    
    sim_time_seconds = sim_hours * 3600.0
    num_steps = int(sim_time_seconds / dt)
    num_era5_states = int(sim_hours) + 1 

    constants = {'g': 9.81, 'Rd': 287.0, 'cp': 1004.0, 'cvd': 717.0, 'p0': 100000.0, 'epsilon': 0.622}
    USE_MOISTURE = True 

    # Dynamically calculate the bounding box
    dynamic_bbox = ERA5Manager.calculate_required_bbox(
        cfg["lat_c"], cfg["lon_c"], 
        nx, ny, dx, dy, 
        buffer_deg=2.0
    )

    print(f"[CONFIG] Domain: {ACTIVE_DOMAIN.upper()}")
    print(f"[CONFIG] ERA5 Bounding Box [N, W, S, E]: {[round(x, 2) for x in dynamic_bbox]}")

    # ==========================================
    # 2. DATA ACQUISITION
    # ==========================================
    print(f"[DATA] Validating ERA5 forcing files...")
    manager = ERA5Manager(data_dir="suetes/data")
    
    # Adjust dates as needed for your specific test case
    days_to_run = [str(i).zfill(2) for i in range(1, 5)] 
    
    cache_prefix = f"{ACTIVE_DOMAIN}_Nx{nx}_Ny{ny}_dx{int(dx)}"

    sl_file, pl_file = manager.download_regional_subset(
        year="2026",
        month="05",
        days=days_to_run,
        area=dynamic_bbox,
        prefix=cache_prefix
    )

    # ==========================================
    # 3. GEOMETRY & TOPOGRAPHY
    # ==========================================
    print(f"[GEOMETRY] Building {nx}x{ny}x{nz} terrain-following mesh (dx={dx/1000}km)...")
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
    print(f"[BOUNDARY] Processing lateral conditions for {sim_hours}h simulation...")
    era5_proc = ERA5Processor(pl_path=pl_file, sl_path=sl_file)
    
    suetes_bc_states = []
    times_sec = []
    
    raw_t0 = era5_proc.get_stitched_state(time_idx=0)
    bridge = BoundaryProcessor(grid, raw_t0['latitude'], raw_t0['longitude'], constants)
    
    for i in range(num_era5_states):
        print(f"[BOUNDARY] -> Regridding ERA5 state for T={i}h")
        raw_state = era5_proc.get_stitched_state(time_idx=i)
        bc_state = bridge.process(raw_state)
        
        suetes_bc_states.append(bc_state)
        times_sec.append(float(i * 3600.0))
        
    time_manager = TimeManager(suetes_bc_states, times_sec, grid)
    initial_state = suetes_bc_states[0]
    initial_state['q_c'] = jnp.zeros_like(initial_state['q'])

    # ==========================================
    # 5. PHYSICS, STEPPER & SPONGE INITIALIZATION
    # ==========================================
    print(f"[DYNAMICS] Initializing dynamical core...")
    operators = CGridOperator3D(grid)
    
    # Build the Suite
    physics_suite = PhysicsSuite()
    
    # Register continuous tendencies
    # physics_suite.add_tendency_scheme(SmagorinskyLillySGS(grid, operators, constants, Cs=0.15))
    physics_suite.add_tendency_scheme(BulkAerodynamicPBL(grid, operators, Cd_ocean=0.001))
    
    # Register state updates
    if USE_MOISTURE:
        physics_suite.add_update_scheme(SimpleMicrophysics(constants))
        # Tell the dynamical core which variables need conservative advection
        physics_suite.register_tracer('q')
        physics_suite.register_tracer('q_c')

    # Inject into the Solver
    physics = Euler3D(grid, operators, constants, dt=dt, 
                       initial_era5_state=initial_state, damp_height=9000.0, max_damp=3.0, 
                       nu_div_factor=0.1, nu_h_factor=0.1, physics_suite=physics_suite)
    
    stepper = SISLStepper3D(physics, dt)
    # Sponge layer
    sponge = DaviesSponge(grid, operators, sponge_depth=sponge_depth, dt=dt, tau_bndy_factor=10.0)

    # ==========================================
    # 6. INTEGRATION LOOP 
    # ==========================================
    print("-" * 60)
    
    chunk_steps = 120  # Execute 1 hour of simulation per chunk
    
    def step_fn(curr_state, step_idx):
        t_curr = step_idx * dt
        
        # Interpolate boundaries at exactly t_curr
        bc_state_t = time_manager.get_forcing(t_curr)
        
        def bc_fn(state_next, _):
            return sponge.blend(state_next, bc_state_t)
            
        next_state = stepper.step(curr_state, t_curr, forcing=None, bc_fn=bc_fn)
        
        # Track max W for console diagnostics
        max_w = jnp.max(jnp.abs(next_state['w']))
        return next_state, max_w 

    # Initialize the centralized driver
    sim = Simulation(step_fn=step_fn, dt=dt)

    start_time = time.time()
    
    # Run the simulation
    final_state = sim.run(
        initial_state, 
        t_start=0.0, 
        t_end=sim_time_seconds, 
        chunk_steps=chunk_steps
    )

    # ==========================================
    # 7. VISUALIZE RESULTS
    # ==========================================
    print("-" * 60)
    print("[PLOT] Generating standard diagnostic plots...")
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

    visualizer.plot_dashboard(grid, final_state, z_idx=5, sponge_depth=sponge_depth, 
                              time_hours=sim_hours, save_path=os.path.join(output_dir, f"{ACTIVE_DOMAIN}_dash_{sim_hours}h.png"))

    # Slices
    mid_y = grid.ny // 2 
    visualizer.plot_cross_section(grid, final_state, 'w', y_idx=mid_y, sponge_depth=sponge_depth, 
                                  save_path=os.path.join(output_dir, f"{ACTIVE_DOMAIN}_slice_w_{sim_hours}h.png"))
                                  
    visualizer.plot_cross_section(grid, final_state, 'q_c', y_idx=mid_y, sponge_depth=sponge_depth, 
                                  save_path=os.path.join(output_dir, f"{ACTIVE_DOMAIN}_slice_qc_{sim_hours}h.png"))

if __name__ == "__main__":
    main()