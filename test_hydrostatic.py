import jax
import jax.numpy as jnp

from suetes.core.grids import StaggeredGrid
from suetes.dynamics.euler import ICON2DSlice
from suetes.core.steppers import HEVIStepper

def init_rest_state(grid, constants):
    X_m, Z_m = grid.X_m, grid.Z_m
    
    # Background state
    theta_bg = 300.0  # Isentropic background
    
    # Analytic Hydrostatic Balance
    pi_bg = 1.0 - (constants['g'] * Z_m) / (constants['cp'] * theta_bg)
    
    # Ideal gas law
    rho_bg = (constants['p0'] / (constants['Rd'] * theta_bg)) * \
             (pi_bg ** (constants['cvd'] / constants['Rd']))
    
    # NO WARM BUBBLE - Atmosphere perfectly at rest
    th_v = theta_bg * jnp.ones_like(X_m)
    
    state = {
        'u': jnp.zeros_like(grid.X_u),
        'w': jnp.zeros_like(grid.X_w),
        'rho': rho_bg,
        'pi': pi_bg, 
        'th_v': th_v
    }
    return state

def create_simulation(physics, stepper, dt, num_steps):
    def boundary_conditions(state, forcing):
        # Apply strict rigid wall boundaries for the vertical velocity
        state['w'] = state['w'].at[:, 0].set(0.0)
        state['w'] = state['w'].at[:, -1].set(0.0)
        
        if not physics.grid.periodic_x:
            state['u'] = state['u'].at[0, :].set(0.0)
            state['u'] = state['u'].at[-1, :].set(0.0)
            
        return state

    @jax.jit
    def run_loop(init_state):
        def step_fn(state, step_idx):
            t = step_idx * dt 
            new_state = stepper.step(state, t, forcing=None, bc_fn=boundary_conditions)
            return new_state, None
        
        final_state, _ = jax.lax.scan(step_fn, init_state, jnp.arange(num_steps))
        return final_state
        
    return run_loop

# ====================================================================
# 1. SETUP & EXECUTION
# ====================================================================
constants = {
    'g': 9.81, 'cp': 1004.0, 'cvd': 717.0, 'Rd': 287.0, 'p0': 100000.0
}

nx, nz = 200, 100 
Lx, Lz = 20000.0, 10000.0
grid = StaggeredGrid(nx, nz, Lx, Lz, h_func=lambda x: 0.0)
grid.periodic_x = True

physics = ICON2DSlice(grid, constants, damp_height=8000.0)
dt = 1.0 
stepper = HEVIStepper(physics, dt)

print("Initializing Atmosphere at Rest...")
init_state = init_rest_state(grid, constants)

num_steps = 100
run_simulation = create_simulation(physics, stepper, dt, num_steps)

print(f"Running {num_steps} steps to test hydrostatic balance...")
final_state = run_simulation(init_state)

# ====================================================================
# 2. EVALUATE SPURIOUS WINDS
# ====================================================================
max_w = jnp.max(jnp.abs(final_state['w']))
max_u = jnp.max(jnp.abs(final_state['u']))

print("\n--- RESULTS ---")
print(f"Max absolute vertical velocity (w): {max_w:.8e} m/s")
print(f"Max absolute horizontal velocity (u): {max_u:.8e} m/s")

# A perfectly balanced model should return values around 1.0e-15
if max_w > 1e-10:
    print("\nWARNING: Severe hydrostatic imbalance detected!")