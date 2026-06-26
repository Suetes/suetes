import os
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.steppers import build_dynamical_core
from suetes.shared.driver import Simulation

# =====================================================================
# CONFIGURATION SWITCHES
# =====================================================================
CORE_TYPE = "sisl"  # Toggle to "sisl" or "split-explicit"

def run_bubble_at_resolution(dx):
    nx = int(10000 / dx)
    nz = int(10000 / dx)
    ny = 3
    
    # Establish default scales based on core layout choice
    if CORE_TYPE.lower() == "sisl":
        dt = (dx / 125.0) * 5.0  # SISL timesteps map linearly to grid sizes
        core_kwargs = {"dt": dt, "nu_div_factor": 0.0, "nu_h_factor": 0.0, "damp_height": 7500.0, "max_damp": 0.05, "alpha": 0.5}
    elif CORE_TYPE.lower() == "split-explicit":
        dt = (dx / 125.0) * 2.5  # Split explicit large step obeys advection limits
        core_kwargs = {"dt": dt, "ns": 24, "nu_div_factor": 0.0, "nu_h_factor": 0.0, "damp_height": 7500.0, "max_damp": 0.05}

    grid = RegionalGrid3D(nx, ny, nz, dx, dx, dx, lat_center=0.0, lon_center=0.0)
    op = CGridOperator3D(grid)
    constants = {'g': 9.81, 'cp': 1004.0, 'Rd': 287.0, 'cvd': 717.0, 'p0': 100000.0}

    from suetes.regional3d.euler import Euler3D
    tmp_phys = Euler3D(grid, op, constants, dt=dt)
    bg_ref = {
        'rho': tmp_phys.c['p0'] / (tmp_phys.c['Rd'] * tmp_phys.theta_bg) * \
               (tmp_phys.pi_bg ** (tmp_phys.c['cvd'] / tmp_phys.c['Rd'])),
        'pi': tmp_phys.pi_bg, 'th_v': tmp_phys.theta_bg
    }

    state = {
        'u': jnp.zeros((nx+1, ny, nz)), 'v': jnp.zeros((nx, ny+1, nz)), 'w': jnp.zeros((nx, ny, nz+1)),
        'pi': bg_ref['pi'], 'eta_dot': jnp.zeros((nx, ny, nz+1)), 'rho': bg_ref['rho'],
    }

    X, Y, Z = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_m, indexing='ij')
    bubble = jnp.where((X**2 + (Z - 2000.0)**2) <= 1500.0**2, 
                       2.0 * jnp.cos(0.5 * jnp.pi * jnp.sqrt(X**2 + (Z - 2000.0)**2) / 1500.0)**2, 0.0)
    
    state['th_v'] = bg_ref['th_v'] + bubble
    state['rho'] = constants['p0'] / (constants['Rd'] * state['th_v']) * \
                   (bg_ref['pi'] ** (constants['cvd'] / constants['Rd']))

    stepper, dt = build_dynamical_core(
        core_type=CORE_TYPE, grid=grid, operators=op, constants=constants,
        initial_state=state, **core_kwargs
    )

    sim = Simulation(step_fn=lambda s, i: (stepper.step(s, i*dt, None, lambda x, f: x), jnp.max(jnp.abs(s['w']))), dt=dt)
    return sim.run(state, t_start=0.0, t_end=200.0, chunk_steps=int(200.0/dt))


if __name__ == "__main__":
    print(f"Running Spatial Convergence Study using ({CORE_TYPE.upper()}) Engine...")
    res_250 = run_bubble_at_resolution(dx=250.0) 
    res_125 = run_bubble_at_resolution(dx=125.0) 
    res_062 = run_bubble_at_resolution(dx=62.5) 

    def block_average_2d(field_2d, factor):
        nx, nz = field_2d.shape
        return jnp.mean(field_2d.reshape(nx // factor, factor, nz // factor, factor), axis=(1, 3))

    err_coarse = jnp.sqrt(jnp.mean((block_average_2d(res_125['th_v'][:, 1, :], 2) - res_250['th_v'][:, 1, :])**2))
    err_fine = jnp.sqrt(jnp.mean((block_average_2d(res_062['th_v'][:, 1, :], 4) - block_average_2d(res_125['th_v'][:, 1, :], 2))**2))

    print(f"\n--- Convergence Results ({CORE_TYPE.capitalize()}) ---")
    print(f"L2 Error (250m vs 125m): {err_coarse:.5f}")
    print(f"L2 Error (125m vs  62m): {err_fine:.5f}")
    print(f"Empirical Order of Convergence: {np.log2(err_coarse / err_fine):.2f}")