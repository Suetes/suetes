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

from suetes.vis.visualizer import Visualizer


def main():
    # ==========================================
    # 1. DOMAIN & TIME SETUP
    # ==========================================
    nx, ny, nz = 300, 300, 40
    dx, dy, dz = 6000.0, 6000.0, 500.0
    lat_c, lon_c = 45.0, 5.0
    sponge_depth = 30
    
    dt = 30.0 # timestep (in seconds)
    sim_hours = 3 
    
    sim_time_seconds = sim_hours * 3600.0
    num_steps = int(sim_time_seconds / dt)
    num_era5_states = int(sim_hours) + 1 # Need +1 to cap the interpolation interval

    constants = {'g': 9.81, 'Rd': 287.0, 'cp': 1004.0, 'cvd': 717.0, 'p0': 100000.0, 'epsilon': 0.622}

    print(f"Initializing Suetes Domain ({nx}x{ny}x{nz}) at {dx}m resolution...")

    # ==========================================
    # 2. GEOMETRY & TOPOGRAPHY
    # ==========================================
    base_grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_c, lon_c)
    
    topo_proc = TopographyProcessor(
        era5_sl_path="suetes/data/suetes_test_run_single_levels.nc",
        gebco_path="suetes/data/gebco_data.nc"
    )
    h_func = topo_proc.process_and_blend(base_grid, sponge_depth=sponge_depth, smooth_sigma=2.0)  
    sleve_transform = SleveSimple(scale_s=10000.0, n=1.0)
    grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_c, lon_c, h_func=h_func, transform=sleve_transform)

    # ==========================================
    # 3. ERA5 BOUNDARY PROCESSING
    # ==========================================
    print(f"Processing ERA5 Boundaries for a {sim_hours}-hour simulation ({num_era5_states} states)...")
    era5_proc = ERA5Processor(
        pl_path="suetes/data/suetes_test_run_pressure_levels.nc",
        sl_path="suetes/data/suetes_test_run_single_levels.nc"
    )
    
    suetes_bc_states = []
    times_sec = []
    
    # We just create a dummy bridge initialization to get access to the process() function
    raw_t0 = era5_proc.get_stitched_state(time_idx=0)
    bridge = BoundaryProcessor(grid, raw_t0['latitude'], raw_t0['longitude'], constants)
    
    for i in range(num_era5_states):
        print(f"  -> Regridding ERA5 state for T={i}h")
        raw_state = era5_proc.get_stitched_state(time_idx=i)
        bc_state = bridge.process(raw_state)
        
        suetes_bc_states.append(bc_state)
        times_sec.append(float(i * 3600.0))
        
    time_manager = TimeManager(suetes_bc_states, times_sec, grid)
    
    # The starting state for the model is always the first index
    initial_state = suetes_bc_states[0]
    
    # ==========================================
    # 4. PHYSICS & STEPPER INITIALIZATION
    # ==========================================
    print("Initializing dynamical core...")
    operators = CGridOperator3D(grid)
    physics = Euler3D(grid, operators, constants, dt=dt, 
                       initial_era5_state=initial_state, damp_height=9000.0, max_damp=3.0, 
                       nu_div_factor=0.8, nu_h_factor=0.1)
    
    stepper = SISLStepper3D(physics, dt, tracer_keys=['q'])
    sponge = DaviesSponge(grid, sponge_depth=sponge_depth, dt=dt, tau_bndy_factor=10.0)

    # ==========================================
    # 5. THE INTEGRATION LOOP
    # ==========================================
    print(f"Starting integration: {num_steps} steps (dt={dt}s).")
    print("Compiling JAX graph...")
    
    # We write a custom scan_fn here so we can update the time dynamically
    def scan_fn(curr_state, step_idx):
        t_curr = step_idx * dt
        bc_state_t = time_manager.get_forcing(t_curr)
        
        def bc_fn(state_next, _):
            return sponge.blend(state_next, bc_state_t)
            
        next_state = stepper.step(curr_state, t_curr, forcing=None, bc_fn=bc_fn)
        
        # Track the maximum vertical velocity in the domain
        max_w = jnp.max(jnp.abs(next_state['w']))
        
        return next_state, max_w # Return max_w as the accumulated output

    start_time = time.time()

    # final_max_w_array will contain the max w for every single timestep
    final_state, final_max_w_array = jax.lax.scan(scan_fn, initial_state, jnp.arange(num_steps))
    jax.block_until_ready(final_state['u']) 
    
    print(f"Integration complete! Wall time: {time.time() - start_time:.2f} seconds.")
    
    # Print the profile of the first 10 steps, and the last 10 steps (to check stability)
    w_array = np.array(final_max_w_array)
    print(f"Max W (Steps 1-10):  {w_array[:10]}")
    print(f"Max W (Last 10):     {w_array[-10:]}")

    # ==========================================
    # 6. VISUALIZE THE RESULTS
    # ==========================================
    print("Generating comparison plots...")
    visualizer = Visualizer()
    
    # Grab the final ERA5 state to compare against the final model state
    final_era5_state = suetes_bc_states[-1]
    
    # Plot Virtual Potential Temperature in the lower troposphere (e.g., Level 5)
    visualizer.plot_model_vs_era5_map(
        grid=grid, 
        state_model=final_state, 
        state_era5=final_era5_state, 
        variable='th_v', 
        z_idx=5, 
        sponge_depth=sponge_depth,
        save_path=f"compare_th_v_{sim_hours}h.png"
    )

    # Plot U-Wind higher up (e.g., Level 15) to see the synoptic flow
    visualizer.plot_model_vs_era5_map(
        grid=grid, 
        state_model=final_state, 
        state_era5=final_era5_state, 
        variable='u', 
        z_idx=15, 
        sponge_depth=sponge_depth,
        save_path=f"compare_u_wind_{sim_hours}h.png"
    )

    # Plot the Anomaly to verify the imprint is gone
    visualizer.plot_anomaly(
        grid=grid, 
        state_model=final_state, 
        state_era5=final_era5_state, 
        variable='u', 
        z_idx=15, 
        sponge_depth=sponge_depth,
        save_path=f"anomaly_u_{sim_hours}h.png"
    )
    
    # y_idx = ny // 2 slices right through the center of the domain (over the Alps)
    mid_y = grid.ny // 2
    visualizer.plot_suetes_w_cross_section(
        grid=grid, 
        state_model=final_state, 
        y_idx=mid_y, 
        sponge_depth=sponge_depth,
        save_path=f"cross_section_w_{sim_hours}h.png")

    visualizer.plot_divergence(
        grid=grid, 
        state_model=final_state, 
        z_idx=5, 
        sponge_depth=sponge_depth,
        save_path=f"divergence_{sim_hours}h.png")

    visualizer.plot_w_and_isentropes(
        grid=grid, 
        state_model=final_state, 
        y_idx=mid_y, 
        sponge_depth=sponge_depth,
        save_path=f"isentropes_{sim_hours}h.png")

    visualizer.plot_energy_spectrum(
        grid=grid, 
        state_model=final_state, 
        variable='w', 
        z_idx=5, 
        sponge_depth=sponge_depth,
        save_path=f"energy_spectrum_{sim_hours}h.png")

if __name__ == "__main__":
    main()