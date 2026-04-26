"""
run_regional.py

The main driver script for the 3D Regional Non-Hydrostatic Model.
"""

import jax
# jax.config.update("jax_enable_x64", True) # Enable if hydrostatic precision is needed
import jax.numpy as jnp

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.slice2d.operators import CGridOperator3D
from suetes.regional3d.euler import Euler3D
from suetes.dynamics.steppers import SISLStepper3D
from suetes.regional3d.boundaries import DaviesSponge

# ====================================================================
# 1. INITIALIZATION
# ====================================================================
nx, ny, nz = 100, 100, 40
dx, dy, dz = 2000.0, 2000.0, 500.0  # 2km horizontal resolution

# Setup regional projection centered over St. John's, NL
lat_center, lon_center = 47.56, -52.71 

print(f"Initializing RegionalGrid3D over {lat_center}N, {lon_center}E...")
grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_center, lon_center)
operators = CGridOperator3D(grid)

constants = {'g': 9.81, 'cp': 1004.0, 'cvd': 717.0, 'Rd': 287.0, 'p0': 100000.0}
physics = Euler3D(grid, operators, constants, damp_height=15000.0, N_bv=0.01)

dt = 20.0
stepper = SISLStepper3D(physics, dt)
sponge = DaviesSponge(grid, sponge_depth=8)

# ====================================================================
# 2. STATE ALLOCATION
# ====================================================================
def create_initial_state():
    """Generates a base hydrostatically balanced state dictionary."""
    return {
        'u': 10.0 * jnp.ones((nx + 1, ny, nz)),
        'v': 0.0 * jnp.ones((nx, ny + 1, nz)),
        'w': jnp.zeros((nx, ny, nz + 1)),
        'rho': physics.c['p0'] / (physics.c['Rd'] * physics.theta_bg) * \
               (physics.pi_bg ** (physics.c['cvd'] / physics.c['Rd'])),
        'pi': physics.pi_bg,
        'th_v': physics.theta_bg,
        'eta_dot': jnp.zeros((nx, ny, nz + 1))
    }

state = create_initial_state()

# In a real run, this would be loaded from GRIB/NetCDF and interpolated.
# For now, we assume the driving boundaries match our initial state.
era5_target_state = create_initial_state() 

# ====================================================================
# 3. JIT-COMPILED STEP FUNCTION
# ====================================================================
@jax.jit
def integration_step(curr_state, boundary_state):
    """
    One complete pass of the dynamical core. 
    End-to-end differentiable and fully fused by XLA.
    """
    # 1. Physics & Advection (Semi-Implicit Semi-Lagrangian)
    next_state = stepper.step(curr_state)
    
    # 2. Kinematic Bottom Boundary (Topography)
    # (Assuming flat terrain for this initial template)
    next_state['w'] = next_state['w'].at[:, :, 0].set(0.0)
    next_state['w'] = next_state['w'].at[:, :, -1].set(0.0)
    next_state['eta_dot'] = next_state['eta_dot'].at[:, :, 0].set(0.0)
    next_state['eta_dot'] = next_state['eta_dot'].at[:, :, -1].set(0.0)
    
    # 3. Lateral Boundaries (Davies Relaxation)
    final_state = sponge.blend(next_state, boundary_state)
    
    return final_state

# ====================================================================
# 4. MAIN LOOP
# ====================================================================
num_steps = 100
print(f"Compiling and running {num_steps} steps...")

# Using jax.lax.scan is highly recommended here to prevent Python from unrolling 
# the loop and drastically reducing overhead.
def scan_fn(carry_state, step_idx):
    # In a real simulation, era5_target_state would update over time
    new_state = integration_step(carry_state, era5_target_state)
    return new_state, None

final_state, _ = jax.lax.scan(scan_fn, state, jnp.arange(num_steps))

max_w = float(jnp.max(jnp.abs(final_state['w'])))
print(f"Simulation Complete. Max vertical velocity: {max_w:.4e} m/s")