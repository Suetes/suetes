import os

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.steppers import SISLStepper3D
from suetes.shared.driver import Simulation

output_dir = "suetes/plots/benchmarks"
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

#Time step of simulation
dt = 4.0

grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_center=45.0, lon_center=0.0, h_func=schaer_mountain)
op = CGridOperator3D(grid)

constants = {'g': 9.81, 'cp': 1004.0, 'Rd': 287.0, 'cvd': 717.0, 'p0': 100000.0}

# This one requires no diffusion and no extra physics
physics = Euler3D(grid, op, constants, dt=dt, N_bv=0.01, damp_height=12000.0, max_damp=0.5,
                  nu_div_factor=0.0, nu_h_factor=0.0, physics_suite=None)

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
    'th_v': bg_ref['th_v']
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
    # Strictly exclude w and eta_dot to protect the kinematic boundary constraint!
    blend_vars = ['u', 'v', 'th_v', 'pi', 'rho'] 
    
    for k in state_in.keys():
        if k in blend_vars and k in ext_state:
            # Pad the mask appropriately for staggered U-winds
            m = jnp.pad(mask_x, ((0, 1), (0, 0), (0, 0)), mode='edge') if k == 'u' else mask_x
            blended[k] = (1.0 - m) * state_in[k] + m * ext_state[k]
        else:
            blended[k] = state_in[k]
    return blended

def forcing_fn(state, t):
    return state

# Use the SISLStepper3D
stepper = SISLStepper3D(physics, dt)
sim = Simulation(stepper, forcing_fn, bc_fn)

# --- 5. RUN SIMULATION ---
t_end = 7200.0 
print("\nLaunching Schär Mountain Benchmark...")
final_state = sim.run(state, t_start=0.0, t_end=t_end, dt=dt, chunk_steps=50)

# --- 6. VISUALIZE RESULTS ---
print("\nPlotting final state...")
w_slice = final_state['w'][:, 1, :-1] 

x_start_idx = (nx // 4)
x_end_idx = 3 * (nx // 4)
x_plot_1d = grid.x_m[x_start_idx:x_end_idx] / 1000.0 

X_plot, _ = jnp.meshgrid(x_plot_1d, grid.z_m, indexing='ij')
Z_plot = grid.Z_m[x_start_idx:x_end_idx, 1, :] / 1000.0 

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