import jax
import jax.numpy as jnp
import time

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.steppers import SISLStepper3D
from suetes.regional3d.boundaries import DaviesSponge
from suetes.shared.transforms import SleveSimple

from suetes.preprocessing.processor import ERA5Processor
from suetes.preprocessing.topography import TopographyProcessor
from suetes.preprocessing.era2suetes import BoundaryProcessor

from suetes.vis.visualizer import Visualizer

class TimeManager:
    def __init__(self, state_t0, state_t1, t0_sec=0.0, t1_sec=3600.0):
        """Holds two boundary states and linearly interpolates between them."""
        self.state_t0 = state_t0
        self.state_t1 = state_t1
        self.t0 = t0_sec
        self.t1 = t1_sec
        
    def get_forcing(self, t):
        """Returns the linearly interpolated 3D boundary state for time t."""
        alpha = (t - self.t0) / (self.t1 - self.t0)
        alpha = jnp.clip(alpha, 0.0, 1.0) # Ensure we don't extrapolate
        
        interp_state = {}
        for k in self.state_t0.keys():
            interp_state[k] = (1.0 - alpha) * self.state_t0[k] + alpha * self.state_t1[k]
            
        return interp_state


def main():
    # ==========================================
    # 1. DOMAIN & TIME SETUP
    # ==========================================
    nx, ny, nz = 300, 300, 40
    dx, dy, dz = 6000.0, 6000.0, 500.0
    lat_c, lon_c = 45.0, 5.0
    sponge_depth = 30
    
    dt = 20.0 # 20 second timestep
    sim_time_seconds = 3600.0 # Run for 1 hour
    num_steps = int(sim_time_seconds / dt)

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
    print("Processing ERA5 Boundaries for T=0h and T=1h...")
    era5_proc = ERA5Processor(
        pl_path="suetes/data/suetes_test_run_pressure_levels.nc",
        sl_path="suetes/data/suetes_test_run_single_levels.nc"
    )
    
    # Process T=0 (Start) and T=1 (Target 1 hour later)
    raw_t0 = era5_proc.get_stitched_state(time_idx=0)
    raw_t1 = era5_proc.get_stitched_state(time_idx=1)
    
    bridge = BoundaryProcessor(grid, raw_t0['latitude'], raw_t0['longitude'], constants)
    
    suetes_bc_t0 = bridge.process(raw_t0)
    suetes_bc_t1 = bridge.process(raw_t1)
    
    time_manager = TimeManager(suetes_bc_t0, suetes_bc_t1, t0_sec=0.0, t1_sec=3600.0)

    # ==========================================
    # 4. PHYSICS & STEPPER INITIALIZATION
    # ==========================================
    print("Initializing Dynamical Core...")
    operators = CGridOperator3D(grid)
    physics = Euler3D(grid, operators, constants, damp_height=11500.0, nu_h=5e5)
    
    stepper = SISLStepper3D(physics, dt, tracer_keys=['q'])
    sponge = DaviesSponge(grid, sponge_depth=sponge_depth)

    # We start the model identically to the T=0 boundary condition
    initial_state = suetes_bc_t0

    # ==========================================
    # 5. THE INTEGRATION LOOP
    # ==========================================
    print(f"Starting integration: {num_steps} steps (dt={dt}s).")
    print("Compiling JAX graph... (This will take a few minutes on the first run!)")
    
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
    num_steps = 10
    final_state, final_max_w_array = jax.lax.scan(scan_fn, initial_state, jnp.arange(num_steps))
    jax.block_until_ready(final_state['u']) 
    
    print(f"Integration Complete! Wall time: {time.time() - start_time:.2f} seconds.")
    
    # Print the profile of the first 10 steps, and the last 10 steps
    import numpy as np
    w_array = np.array(final_max_w_array)
    print(f"Max W (Steps 1-10):  {w_array[:10]}")
    print(f"Max W (Last 10):     {w_array[-10:]}")

    # ==========================================
    # 6. VISUALIZE THE RESULTS
    # ==========================================
    print("Generating comparison plots...")
    visualizer = Visualizer()
    
    # Plot Virtual Potential Temperature in the lower troposphere (e.g., Level 5)
    visualizer.plot_model_vs_era5_map(
        grid=grid, 
        state_model=final_state, 
        state_era5=suetes_bc_t1, # This is the exact ERA5 state interpolated to the Suetes Grid!
        variable='th_v', 
        z_idx=5, 
        save_path="compare_th_v_1h.png"
    )

    # Plot U-Wind higher up (e.g., Level 15) to see the synoptic flow
    visualizer.plot_model_vs_era5_map(
        grid=grid, 
        state_model=final_state, 
        state_era5=suetes_bc_t1, 
        variable='u', 
        z_idx=15, 
        save_path="compare_u_wind_1h.png"
    )

    # Plot the Anomaly to verify the imprint is gone
    visualizer.plot_anomaly(grid, final_state, suetes_bc_t1, variable='u', z_idx=15, save_path="anomaly_u_1h.png")
    
    # Plot the Lie Detector test! 
    # y_idx = ny // 2 slices right through the center of the domain (over the Alps)
    mid_y = grid.ny // 2
    visualizer.plot_suetes_w_cross_section(grid, final_state, y_idx=mid_y, save_path="cross_section_w_1h.png")

if __name__ == "__main__":
    main()