import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import jax
import jax.numpy as jnp
import time
import numpy as np

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.steppers import SISLStepper3D
from suetes.regional3d.boundaries import DaviesSponge
from suetes.vis.visualizer import Visualizer

def main():
    # ==========================================
    # 1. DOMAIN SETUP (FLAT EARTH)
    # ==========================================
    output_dir = "suetes/plots"
    os.makedirs(output_dir, exist_ok=True)

    nx, ny, nz = 150, 150, 40 # Smaller grid for faster debugging
    dx, dy, dz = 6000.0, 6000.0, 500.0
    lat_c, lon_c = 48.0, -60.0 
    sponge_depth = 15 # Scaled down for the smaller grid
    
    dt = 30.0 
    sim_hours = 1.0 # 1 hour is plenty to see if corners blow up
    num_steps = int((sim_hours * 3600.0) / dt)

    constants = {'g': 9.81, 'Rd': 287.0, 'cp': 1004.0, 'cvd': 717.0, 'p0': 100000.0, 'epsilon': 0.622}

    print("Initializing Flat Grid...")
    # Passing no h_func defaults it to lambda x, y: 0.0
    grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_c, lon_c)
    operators = CGridOperator3D(grid)

    # ==========================================
    # 2. IDEALIZED PHYSICS & STATE SETUP
    # ==========================================
    print("Initializing Idealized Physics...")
    # Pass None to initial_era5_state to trigger the analytical N_bv background
    physics = Euler3D(grid, operators, constants, dt=dt, 
                      initial_era5_state=None, damp_height=15000.0, max_damp=3.0, 
                      use_pbl=False) # Turn off PBL to remove surface friction noise

    print("Building Uniform Flow State (U=10, V=10)...")
    # Base analytical density from the equation of state
    rho_bg = constants['p0'] / (constants['Rd'] * physics.theta_bg) * \
             (physics.pi_bg ** (constants['cvd'] / constants['Rd']))

    uniform_state = {
        'u': jnp.ones((nx + 1, ny, nz)) * 10.0,
        'v': jnp.ones((nx, ny + 1, nz)) * 10.0,
        'w': jnp.zeros((nx, ny, nz + 1)),
        'eta_dot': jnp.zeros((nx, ny, nz + 1)),
        'pi': physics.pi_bg,
        'th_v': physics.theta_bg,
        'rho': rho_bg,
        'q': jnp.zeros((nx, ny, nz)),
        'q_c': jnp.zeros((nx, ny, nz))
    }

    stepper = SISLStepper3D(physics, dt, tracer_keys=['q', 'q_c'], use_moisture=False)
    sponge = DaviesSponge(grid, sponge_depth=sponge_depth, dt=dt, tau_bndy_factor=10.0)

    # ==========================================
    # 3. INTEGRATION LOOP
    # ==========================================
    print(f"Starting idealized integration: {num_steps} steps.")
    
    def scan_fn(curr_state, step_idx):
        t_curr = step_idx * dt
        
        # The boundary forcing is perfectly constant
        def bc_fn(state_next, _):
            return sponge.blend(state_next, uniform_state)
            
        next_state = stepper.step(curr_state, t_curr, forcing=None, bc_fn=bc_fn)
        
        # Track Max/Min U-Wind to catch instabilities immediately
        max_u = jnp.max(next_state['u'])
        min_u = jnp.min(next_state['u'])
        return next_state, jnp.array([max_u, min_u])

    start_time = time.time()
    final_state, u_extrema = jax.lax.scan(scan_fn, uniform_state, jnp.arange(num_steps))
    jax.block_until_ready(final_state['u']) 
    
    print(f"Integration complete in {time.time() - start_time:.2f} seconds.")
    
    u_extrema = np.array(u_extrema)
    print("\n--- Stability Check ---")
    print(f"Starting U-Wind: 10.00 m/s")
    print(f"Final Max U:     {u_extrema[-1, 0]:.5f} m/s")
    print(f"Final Min U:     {u_extrema[-1, 1]:.5f} m/s")
    
    if np.max(u_extrema[:, 0]) > 15.0 or np.min(u_extrema[:, 1]) < 5.0:
        print("⚠️ INSTABILITY DETECTED: Winds deviated significantly from 10 m/s.")
    else:
        print("✅ STABLE: Winds remained close to 10 m/s.")

    # ==========================================
    # 4. VISUALIZATION
    # ==========================================
    visualizer = Visualizer()
    print("Generating U-Wind diagnostic plot...")
    
    visualizer.plot_2d_field(grid, final_state, 'u', z_idx=5, sponge_depth=sponge_depth, 
                             cmap='seismic', scale='sym', 
                             save_path=os.path.join(output_dir, f"idealized_u_wind_{sim_hours}h.png"))

if __name__ == "__main__":
    main()