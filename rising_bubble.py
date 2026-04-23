import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from suetes.core.grids import StaggeredGrid
from suetes.dynamics.euler import ICON2DSlice
from suetes.core.steppers import HEVIStepper

def init_warm_bubble(grid, constants):
    X_m, Z_m = grid.X_m, grid.Z_m
    
    # Background state
    theta_bg = 300.0  # Isentropic background
    
    # Analytic Hydrostatic Balance
    # pi_bg(z) = 1 - (g * z) / (cp * theta_bg)
    pi_bg = 1.0 - (constants['g'] * Z_m) / (constants['cp'] * theta_bg)
    
    # Ideal gas law in terms of pi and theta to get density
    # rho = (p0 / (Rd * theta)) * pi^(cv/Rd)
    rho_bg = (constants['p0'] / (constants['Rd'] * theta_bg)) * \
             (pi_bg ** (constants['cvd'] / constants['Rd']))
    
    # The Warm Bubble Anomaly
    # Centered at x=10km, z=2km, with a 2km radius and 2K max amplitude
    xc, zc = 10000.0, 2000.0
    r_bubble = 2000.0
    r = jnp.sqrt((X_m - xc)**2 + (Z_m - zc)**2)
    
    # Cosine squared bubble perturbation
    theta_pert = jnp.where(
        r < r_bubble, 
        2.0 * jnp.cos(0.5 * jnp.pi * r / r_bubble)**2, 
        0.0
    )
    th_v = theta_bg + theta_pert
    
    # Initialize state dictionary
    # Note: X_u and X_w would be pulled from your StaggeredGrid object
    state = {
        'u': jnp.zeros_like(grid.X_u),
        'w': jnp.zeros_like(grid.X_w),
        'rho': rho_bg,
        'pi': pi_bg,       # We initialize pi/rho with the background. The solver handles the acoustic adjustment.
        'th_v': th_v
    }
    return state

def create_simulation(physics, stepper, dt, num_steps):
    
    def boundary_conditions(state, forcing):
        # Apply strict rigid wall boundaries for the vertical velocity
        state['w'] = state['w'].at[:, 0].set(0.0)
        state['w'] = state['w'].at[:, -1].set(0.0)
        
        # Periodic boundaries in X (Assuming your operators handle the padding)
        # For simplicity in this slice model, if periodic_x=False, we assume 
        # rigid walls at x=0 and x=Lx for u.
        if not physics.grid.periodic_x:
            state['u'] = state['u'].at[0, :].set(0.0)
            state['u'] = state['u'].at[-1, :].set(0.0)
            
        return state

    @jax.jit
    def run_loop(init_state):
        def step_fn(state, step_idx):
            # Time isn't explicitly needed for this unforced benchmark, 
            # but standard signature includes it.
            t = step_idx * dt 
            
            # Execute one HE-VI Predictor step
            new_state = stepper.step(state, t, forcing=None, bc_fn=boundary_conditions)
            
            return new_state, None # Return None for the history output to save memory
        
        # JAX scan executes the loop natively on device
        final_state, _ = jax.lax.scan(step_fn, init_state, jnp.arange(num_steps))
        return final_state
        
    return run_loop

# --- 1. Constants & Grid ---
constants = {
    'g': 9.81, 'cp': 1004.0, 'cvd': 717.0, 'Rd': 287.0, 
    'p0': 100000.0, 'R': 287.0 # Legacy key mapping if needed
}

# Domain: 20km wide x 10km high. Resolution: 100m.
nx, nz = 200, 100 
Lx, Lz = 20000.0, 10000.0
# (Assuming your StaggeredGrid init takes these params)
grid = StaggeredGrid(nx, nz, Lx, Lz, h_func=lambda x: 0.0)
grid.periodic_x = True

# --- 2. Initialize Physics & Stepper ---
# Damp out acoustic/gravity waves above 8km to prevent roof reflections
physics = ICON2DSlice(grid, constants, damp_height=8000.0)

# The time step. Sound speed is ~340m/s. dx=100m.
# An explicit solver would crash at dt > 0.15s.
# With HE-VI, we can safely use dt = 1.0s or more depending on max updraft.
dt = 0.15 
stepper = HEVIStepper(physics, dt)

# --- 3. Run Benchmark ---
print("Initializing Warm Bubble...")
init_state = init_warm_bubble(grid, constants)

# Run for 1000 seconds (approx 16 minutes of simulated time)
num_steps = 10000
run_simulation = create_simulation(physics, stepper, dt, num_steps)

print(f"Compiling and running {num_steps} steps...")
# JAX will JIT compile on the first call
final_state = run_simulation(init_state)
print("Simulation complete. Bubble has ascended.")

print("Generating plots...")
th_pert = final_state['th_v'] - 300.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# th_v is shape (nx, nz). We transpose it for imshow so z is on the vertical axis.
c1 = ax1.imshow(th_pert.T, origin='lower', extent=[0, Lx/1000, 0, Lz/1000], cmap='RdBu_r')
ax1.set_title("Potential Temperature Perturbation (K)")
ax1.set_xlabel("x (km)")
ax1.set_ylabel("z (km)")
fig.colorbar(c1, ax=ax1)

# w is shape (nx, nz+1). We can plot the first nz levels for simplicity, or just plot all of it.
c2 = ax2.imshow(final_state['w'][:, :-1].T, origin='lower', extent=[0, Lx/1000, 0, Lz/1000], cmap='RdBu_r')
ax2.set_title("Vertical Velocity (m/s)")
ax2.set_xlabel("x (km)")
fig.colorbar(c2, ax=ax2)

plt.tight_layout()
plt.savefig('bubble.png', dpi=150)
print("Plot successfully saved to bubble.png")