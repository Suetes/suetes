import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.steppers import SISLStepper3D
from suetes.shared.driver import Simulation

from suetes.physics.base import PhysicsSuite
from suetes.physics.gravity_waves import UpperRayleighDamping, McFarlaneGWD

output_dir = "suetes/plots/physics"
os.makedirs(output_dir, exist_ok=True)

# --- 1. SCHÄR MOUNTAIN PROFILE ---
def schaer_mountain(x, y):
    h0 = 250.0      # Maximum height of 250m
    a = 5000.0      # Envelope half-width of 5km
    lam = 4000.0    # Wavelength of 4km
    return h0 * jnp.exp(-(x / a)**2) * jnp.cos(jnp.pi * x / lam)**2

# --- 2. SETUP GRID & PHYSICS ---
nx, ny, nz = 200, 3, 50
dx, dy, dz = 500.0, 500.0, 400.0  

# Simulation end time and time step
t_end = 7200.0 
dt = 4.0

grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_center=45.0, lon_center=0.0, h_func=schaer_mountain)
op = CGridOperator3D(grid)

constants = {'g': 9.81, 'cp': 1004.0, 'Rd': 287.0, 'cvd': 717.0, 'p0': 100000.0}

# --- 2b. SETUP PHYSICS SUITE ---
suite = PhysicsSuite()

# 2. Add Orographic Gravity Wave Drag
# Create a smooth, differentiable Gaussian subgrid variance
a = 5000.0  # Match the Schär mountain half-width
var_1d = 100.0 * jnp.exp(-(grid.x_m / a)**2)
synthetic_h_var = jnp.broadcast_to(var_1d[:, None], (nx, ny))

suite.add_tendency_scheme(
    McFarlaneGWD(grid, op, constants, h_variance=synthetic_h_var)
)

# Pass the suite into Euler3D
physics = Euler3D(grid, op, constants, dt=dt, 
N_bv=0.01, damp_height=12000.0, max_damp=0.5,
                  nu_div_factor=0.0, nu_h_factor=0.0, physics_suite=suite)

# --- 3. INITIALIZE STATE ---
u_bg = 10.0

bg_ref = {
    'rho': physics.c['p0'] / (physics.c['Rd'] * physics.theta_bg) * \
           (physics.pi_bg ** (physics.c['cvd'] / physics.c['Rd'])),
    'pi': physics.pi_bg,
    'th_v': physics.theta_bg
}

state = {
    'u': jnp.ones((nx+1, ny, nz)) * u_bg,
    'v': jnp.zeros((nx, ny+1, nz)),
    'w': jnp.zeros((nx, ny, nz+1)),
    'pi': bg_ref['pi'],
    'eta_dot': jnp.zeros((nx, ny, nz+1)),
    'rho': bg_ref['rho'],
    'th_v': bg_ref['th_v'],

    # REQUIRED BY UPPER RAYLEIGH DAMPING:
    'target_u': jnp.ones((nx+1, ny, nz)) * u_bg,
    'target_v': jnp.zeros((nx, ny+1, nz)),
}

# --- 4. BOUNDARIES & STEPPER ---
# Custom 1D X-only sponge for pseudo-2D domains
sponge_depth = 8
x_idx = jnp.arange(nx, dtype=jnp.float32)
dist_x = jnp.minimum(x_idx, nx - x_idx)
weight_x = jnp.where(dist_x < sponge_depth, jnp.cos(0.5 * jnp.pi * dist_x / sponge_depth)**2, 0.0)
mask_x = weight_x[:, None, None] # Broadcast to 3D

def bc_fn(state_in, forcing):
    ext_state = {
        'u': jnp.ones_like(state_in['u']) * u_bg,
        'v': jnp.zeros_like(state_in['v']),
        'th_v': bg_ref['th_v'],
        'rho': bg_ref['rho'],
        'pi': bg_ref['pi']
    }
    
    blended = {}
    blend_vars = ['u', 'v', 'th_v', 'pi', 'rho'] 
    
    for k in state_in.keys():
        if k in blend_vars and k in ext_state:
            m = jnp.pad(mask_x, ((0, 1), (0, 0), (0, 0)), mode='edge') if k == 'u' else mask_x
            blended[k] = (1.0 - m) * state_in[k] + m * ext_state[k]
        else:
            # Passes w, eta_dot, target_u, and target_v through untouched
            blended[k] = state_in[k] 
    return blended


stepper = SISLStepper3D(physics, dt)

def step_fn(curr_state, step_idx):
    t_curr = step_idx * dt
    # Advance the state
    next_state = stepper.step(curr_state, t_curr, forcing=None, bc_fn=bc_fn)
    
    # Re-attach the static fields that the stepper drops so JAX PyTree structure matches
    next_state['target_u'] = curr_state['target_u']
    next_state['target_v'] = curr_state['target_v']
    
    # Track the maximum vertical velocity for diagnostics
    max_w = jnp.max(jnp.abs(next_state['w']))
    return next_state, max_w

sim = Simulation(step_fn=step_fn, dt=dt)

# --- 5. RUN SIMULATION ---
print("\nLaunching Schär Mountain Benchmark...")
final_state = sim.run(state, t_start=0.0, t_end=t_end, chunk_steps=50)

# --- 6. VISUALIZE RESULTS ---
print("\nPlotting final state...")
# Vertical velocity at the mid-level in the vertical (index 1)
w_slice = final_state['w'][:, 1, :] 

x_start_idx = (nx // 4)
x_end_idx = 3 * (nx // 4)
x_plot_1d = grid.x_m[x_start_idx:x_end_idx] / 1000.0 

# Use a dummy array of size nz+1 to get the right shape for X_plot
X_plot, _ = jnp.meshgrid(x_plot_1d, jnp.arange(nz + 1), indexing='ij')

# Use the terrain-following W-grid heights!
Z_plot = grid.Z_w[x_start_idx:x_end_idx, 1, :] / 1000.0 

plt.figure(figsize=(12, 6))
contour = plt.contourf(X_plot, Z_plot, w_slice[x_start_idx:x_end_idx, :], 
                       levels=jnp.linspace(-2.0, 2.0, 41), cmap='RdBu_r', extend='both')
plt.colorbar(contour, label='Vertical Velocity W (m/s)')
plt.title(f'Schär Mountain Waves at T = {t_end}s')
plt.xlabel('Distance (km)')
plt.ylabel('Altitude (km)')

mountain_terrain = schaer_mountain(grid.x_m[x_start_idx:x_end_idx], 0.0) / 1000.0
plt.fill_between(x_plot_1d, 0, mountain_terrain, color='black')

plt.savefig(f'{output_dir}/schaer_mountain_3d_{t_end}.png', dpi=150, bbox_inches='tight')
print(f"Saved plot to '{output_dir}/schaer_mountain_3d_{t_end}.png'")