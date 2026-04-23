import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from suetes.core.grids import StaggeredGrid
from suetes.dynamics.euler import ICON2DSlice
from suetes.core.steppers import HEVIStepper

def init_acoustic_pulse(grid, constants):
    X_m, Z_m = grid.X_m, grid.Z_m
    
    theta_bg = 300.0
    pi_bg = 1.0  # Exact because g=0
    rho_bg = (constants['p0'] / (constants['Rd'] * theta_bg))
    
    # Gaussian Pressure Spike in center
    xc, zc = (grid.nx * grid.dx) / 2, (grid.nz * grid.dz) / 2
    r = jnp.sqrt((X_m - xc)**2 + (Z_m - zc)**2)
    pi_pert = jnp.where(r < 1000.0, 0.001 * jnp.exp(-(r/500.0)**2), 0.0)
    
    state = {
        'u': jnp.zeros_like(grid.X_u),
        'w': jnp.zeros_like(grid.X_w),
        'rho': rho_bg * jnp.ones_like(X_m),
        'pi': pi_bg + pi_pert,
        'th_v': theta_bg * jnp.ones_like(X_m)
    }
    return state

# ====================================================================
# SETUP & EXECUTION
# ====================================================================
# CRITICAL: Setting g=0 isolates the acoustic solver
constants = {'g': 0.0, 'cp': 1004.0, 'cvd': 717.0, 'Rd': 287.0, 'p0': 100000.0}

nx, nz = 100, 100
dx, dz = 100.0, 100.0
grid = StaggeredGrid(nx, nz, nx*dx, nz*dz, h_func=lambda x: 0.0)
grid.periodic_x = True

# Initialize native physics and stepper
physics = ICON2DSlice(grid, constants, damp_height=20000.0)
dt = 0.1 # Must be < dx / c_s for explicit horizontal acoustics
stepper = HEVIStepper(physics, dt)
stepper.beta1 = 0.5 # ICON recommendation for pure acoustic tests

def boundary_conditions(state, forcing):
    state['w'] = state['w'].at[:, 0].set(0.0)
    state['w'] = state['w'].at[:, -1].set(0.0)
    return state

@jax.jit
def run_loop(init_state):
    def step_fn(s, step_idx):
        return stepper.step(s, 0.0, forcing=None, bc_fn=boundary_conditions), None
    final_state, _ = jax.lax.scan(step_fn, init_state, jnp.arange(100))
    return final_state

print("Running Native Acoustic Pulse for 10 seconds (100 steps)...")
state = init_acoustic_pulse(grid, constants)
final_state = run_loop(state)

# Plotting Wind Divergence
div_u = (final_state['u'][1:, :] - final_state['u'][:-1, :]) / dx
div_w = (final_state['w'][:, 1:] - final_state['w'][:, :-1]) / dz
divergence = div_u + div_w

plt.figure(figsize=(8, 6))
plt.imshow(divergence.T, origin='lower', extent=[0, nx*dx/1000, 0, nz*dz/1000], cmap='seismic')
plt.title("Acoustic Wave Front (Native Solver) at t=10s")
plt.xlabel("x (km)")
plt.ylabel("z (km)")
plt.colorbar(label="Divergence (s^-1)")
plt.savefig('acoustic_pulse.png', dpi=150)
print("Saved acoustic_pulse.png")