import jax
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.steppers import SISLStepper3D
from suetes.regional3d.boundaries import DaviesSponge

nx, ny, nz = 50, 50, 30
dx, dy, dz = 2000.0, 2000.0, 500.0
lat_center, lon_center = 47.56, -52.71 

grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_center, lon_center)
operators = CGridOperator3D(grid)
constants = {'g': 9.81, 'cp': 1004.0, 'cvd': 717.0, 'Rd': 287.0, 'p0': 100000.0}
physics = Euler3D(grid, operators, constants, damp_height=10000.0, N_bv=0.01)

dt = 10.0
stepper = SISLStepper3D(physics, dt)
sponge = DaviesSponge(grid, sponge_depth=5)

def create_initial_state():
    return {
        'u': 10.0 * jnp.ones((nx + 1, ny, nz)),
        'v': 0.0 * jnp.ones((nx, ny + 1, nz)),
        'w': jnp.zeros((nx, ny, nz + 1)),
        'rho': physics.c['p0'] / (physics.c['Rd'] * physics.theta_bg) * (physics.pi_bg ** (physics.c['cvd'] / physics.c['Rd'])),
        'pi': physics.pi_bg,
        'th_v': physics.theta_bg,
        'eta_dot': jnp.zeros((nx, ny, nz + 1)) # Strict Pytree structure
    }

state = create_initial_state()
era5_target_state = create_initial_state() 

@jax.jit
def integration_step(curr_state, boundary_state):
    next_state = stepper.step(curr_state)
    
    # Kinematic bottom boundary and strictly constrained eta_dot
    next_state['w'] = next_state['w'].at[:, :, 0].set(0.0)
    next_state['w'] = next_state['w'].at[:, :, -1].set(0.0)
    next_state['eta_dot'] = next_state['eta_dot'].at[:, :, 0].set(0.0)
    next_state['eta_dot'] = next_state['eta_dot'].at[:, :, -1].set(0.0)
    
    return sponge.blend(next_state, boundary_state)

def scan_fn(carry_state, step_idx):
    return integration_step(carry_state, era5_target_state), None

print("Compiling and Running 3D Regional Simulation...")
final_state, _ = jax.lax.scan(scan_fn, state, jnp.arange(10))
print(f"Success! Max vertical velocity: {float(jnp.max(jnp.abs(final_state['w']))):.4e} m/s")