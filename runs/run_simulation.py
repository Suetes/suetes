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
from suetes.regional3d.physics import PhysicsSuite, BulkAerodynamicPBL, SimpleMicrophysics, FastVerticalDiffusion, NewtonianRelaxation

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
    ACTIVE_DOMAIN = "alps" 
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
    USE_MOISTURE = False 

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
    
    initial_state = suetes_bc_states[0].copy() 
    
    if USE_MOISTURE:
        initial_state['q_c'] = jnp.zeros_like(initial_state['q'])
    else:
        initial_state.pop('q', None)
        initial_state.pop('q_c', None)

    # Initialize the forcing variables so the PyTree structure matches
    initial_state['theta_surf'] = initial_state['th_v'][:, :, 0]
    initial_state['target_th_v'] = initial_state['th_v']

    # ==========================================
    # 5. PHYSICS, STEPPER & SPONGE INITIALIZATION
    # ==========================================
    print(f"[DYNAMICS] Initializing dynamical core...")
    operators = CGridOperator3D(grid)
    
    # Build the Suite
    physics_suite = PhysicsSuite()
    
    # Extract the initial ERA5 surface temperature to use as our static boundary condition
    theta_surf = initial_state['th_v'][:, :, 0]
    
    # Instantiate the schemes but keep references to them
    pbl_scheme = BulkAerodynamicPBL(grid, operators, theta_surf=initial_state['th_v'][:, :, 0], Cd_ocean=0.001, Ch_ocean=0.0)
    vert_diff_scheme = FastVerticalDiffusion(grid, operators) # Add the new scheme
    nudging_scheme = NewtonianRelaxation(tau_relax_hours=6.0)
    
    physics_suite.add_tendency_scheme(pbl_scheme)
    physics_suite.add_tendency_scheme(vert_diff_scheme)
    physics_suite.add_tendency_scheme(nudging_scheme)
    
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
    
    # Setup Hovmöller Data Trackers
    hov_times = [0.0]
    # Calculate initial anomaly at the surface (level 0)
    initial_anom = initial_state['th_v'][:, :, 0] - initial_state['th_v'][:, :, 0] # 0 at T=0
    hov_data = [np.array(jnp.mean(initial_anom, axis=1))] 

    # Define a pure Python callback function to handle the appending
    def save_hovmoller_data(hour, anom_array):
        hov_times.append(float(hour))
        hov_data.append(np.array(anom_array))

    def step_fn(curr_state, step_idx):
        t_curr = step_idx * dt
        
        # Interpolate boundaries at exactly t_curr
        bc_state_t = time_manager.get_forcing(t_curr)

        # Update the PBL scheme with the current ERA5 surface temperature
        curr_state['theta_surf'] = bc_state_t['th_v'][:, :, 0]
        curr_state['target_th_v'] = bc_state_t['th_v']
        
        def bc_fn(state_next, _):
            return sponge.blend(state_next, bc_state_t)
            
        next_state = stepper.step(curr_state, t_curr, forcing=None, bc_fn=bc_fn)
        next_state['theta_surf'] = curr_state['theta_surf']
        next_state['target_th_v'] = curr_state['target_th_v']
        
        max_w = jnp.max(jnp.abs(next_state['w']))
        
        # Compute the anomaly every step
        anom = next_state['th_v'][:, :, 0] - bc_state_t['th_v'][:, :, 0]
        y_avg_anom = jnp.mean(anom, axis=1)
        current_hour = (step_idx + 1) * dt / 3600.0
        
        # Define the condition as a JAX array
        is_hourly = ((step_idx + 1) % int(3600.0 / dt)) == 0
        
        # Use jax.lax.cond to conditionally trigger the python callback
        jax.lax.cond(
            is_hourly,
            lambda: jax.debug.callback(save_hovmoller_data, current_hour, y_avg_anom),
            lambda: None
        )

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
    print("[PLOT] Generating diagnostic plots...")
    visualizer = Visualizer()
    final_era5_state = suetes_bc_states[-1]
    mid_x, mid_y = grid.nx // 2, grid.ny // 2
    
    # General overview and stability check
    for z in [0, 5, 15, 30]: # Logical model levels
        visualizer.plot_dashboard(grid, final_state, z_idx=z, sponge_depth=sponge_depth, 
                                  time_hours=sim_hours, save_path=os.path.join(output_dir, f"{ACTIVE_DOMAIN}_dash_z{z}_{sim_hours}h.png"))

    for z in [500.0, 3000.0, 5000.0, 10000.0]: # Geometric heights (m)
        visualizer.plot_dashboard(grid, final_state, z_idx=z, sponge_depth=sponge_depth, 
                                  time_hours=sim_hours, save_path=os.path.join(output_dir, f"{ACTIVE_DOMAIN}_dash_z{int(z)}m_{sim_hours}h.png"))

    visualizer.plot_energy_spectrum(grid, final_state, 'w', z_idx=5, sponge_depth=sponge_depth, 
                                    save_path=os.path.join(output_dir, f"{ACTIVE_DOMAIN}_energy_{sim_hours}h.png"))

    # Thermodynamic drift diagnostics
    # Comparison of the surface layer to see the spatial footprint of the bias
    visualizer.plot_comparison(grid, final_state, final_era5_state, 'th_v', z_idx=0, sponge_depth=sponge_depth, 
                               save_path=os.path.join(output_dir, f"{ACTIVE_DOMAIN}_compare_th_v_surf_{sim_hours}h.png"))

    # Level strip to see how deep the drift penetrates vertically
    visualizer.plot_level_strip(grid, final_state, 'th_v', z_indices=[0, 5, 15, 30], sponge_depth=sponge_depth,
                                save_path=os.path.join(output_dir, f"{ACTIVE_DOMAIN}_levels_th_v_{sim_hours}h.png"))

    # Hovmöller diagram to watch the drift evolve over time and space
    visualizer.plot_hovmoller(grid, hov_times, np.array(hov_data), variable='Surface th_v Anomaly [K]',
                              save_path=os.path.join(output_dir, f"{ACTIVE_DOMAIN}_hovmoller_th_v_{sim_hours}h.png"))

    # Dynamics & Mountain Waves
    visualizer.plot_slice_locator_dashboard(grid, final_state, map_var='th_v', slice_var='w', map_z=5, 
                                            sponge_depth=sponge_depth,
                                            save_path=os.path.join(output_dir, f"{ACTIVE_DOMAIN}_slices_w_{sim_hours}h.png"))
                                            
    if USE_MOISTURE:
        visualizer.plot_slice_locator_dashboard(grid, final_state, map_var='q_c', slice_var='q_c', map_z=5, 
                                                sponge_depth=sponge_depth,
                                                save_path=os.path.join(output_dir, f"{ACTIVE_DOMAIN}_slices_qc_{sim_hours}h.png"))

if __name__ == "__main__":
    main()