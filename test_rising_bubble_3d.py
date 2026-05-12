import os

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.steppers import SISLStepper3D
from suetes.regional3d.boundaries import DaviesSponge
from suetes.shared.driver import Simulation

output_dir = "suetes/plots/benchmarks"
os.makedirs(output_dir, exist_ok=True)

# --- 1. SETUP GRID & PHYSICS ---
nx, ny, nz = 80, 3, 80
dx, dy, dz = 125.0, 125.0, 125.0  # 125m resolution

# Simulation end time and time step
t_end = 1000.0 
dt = 2.5  

grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_center=0.0, lon_center=0.0)
op = CGridOperator3D(grid)

constants = {'g': 9.81, 'cp': 1004.0, 'Rd': 287.0, 'cvd': 717.0, 'p0': 100000.0}

# Euler solver parameters
physics = Euler3D(grid, op, constants, dt=dt, N_bv=0.00, damp_height=7500.0, max_damp=0.5,
                  nu_div_factor=0.0, nu_h_factor=0.0, physics_suite=None)

# --- 2. INITIALIZE STATE ---
bg_ref = {
    'rho': physics.c['p0'] / (physics.c['Rd'] * physics.theta_bg) * \
           (physics.pi_bg ** (physics.c['cvd'] / physics.c['Rd'])),
    'pi': physics.pi_bg,
    'th_v': physics.theta_bg
}

state = {
    'u': jnp.zeros((nx+1, ny, nz)),
    'v': jnp.zeros((nx, ny+1, nz)),
    'w': jnp.zeros((nx, ny, nz+1)),
    'pi': bg_ref['pi'], 
    'eta_dot': jnp.zeros((nx, ny, nz+1)),
    'rho': bg_ref['rho'],
}

# Inject the +2K Warm Bubble
X, Y, Z = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_m, indexing='ij')
x_c, z_c = 0.0, 2000.0  # Center bubble
r = jnp.sqrt((X - x_c)**2 + (Z - z_c)**2)

# Cosine-squared bubble profile (radius 1500m)
bubble = jnp.where(r <= 1500.0, 2.0 * jnp.cos(0.5 * jnp.pi * r / 1500.0)**2, 0.0)

# Update the potential temperature
state['th_v'] = bg_ref['th_v'] + bubble

# UPDATE THE DENSITY to maintain pressure equilibrium!
state['rho'] = physics.c['p0'] / (physics.c['Rd'] * state['th_v']) * \
               (bg_ref['pi'] ** (physics.c['cvd'] / physics.c['Rd']))

# --- 3. BOUNDARIES & STEPPER ---
sponge = DaviesSponge(grid, op, sponge_depth=5, dt=dt)

def bc_fn(state_in, forcing):
    # For a pseudo-2D rising bubble, rigid lateral walls are perfectly fine.
    # The explicit top Rayleigh damping will absorb the vertical acoustic/gravity waves.
    # Do not apply the sponge, as it will crush the narrow Y-axis!
    return state_in

# SISLStepper3D initialization
stepper = SISLStepper3D(physics, dt)

def unified_step_fn(curr_state, step_idx):
    t_curr = step_idx * dt
    # Advance the state. Pass None for forcing if there is no external source.
    next_state = stepper.step(curr_state, t_curr, forcing=None, bc_fn=bc_fn)
    
    # Track the maximum vertical velocity for diagnostics
    max_w = jnp.max(jnp.abs(next_state['w']))
    return next_state, max_w

sim = Simulation(step_fn=unified_step_fn, dt=dt)

# --- 4. RUN SIMULATION ---
print(f"[TEST 3D] Launching Warm Bubble Benchmark (dt={dt}s)...")
final_state = sim.run(state, t_start=0.0, t_end=t_end, chunk_steps=20)

# --- 5. VISUALIZE RESULTS ---
print("[PLOTTING] Plotting final state...")
# Extract a 2D slice down the middle of the Y-axis
th_v_slice = final_state['th_v'][:, 1, :]
th_v_bg_slice = bg_ref['th_v'][:, 1, :]
perturbation = th_v_slice - th_v_bg_slice

plt.figure(figsize=(10, 8))
# Transpose for plotting (Z on y-axis, X on x-axis)
plt.contourf(grid.x_m, grid.z_m, perturbation.T, levels=40, cmap='RdBu_r')
plt.colorbar(label='Potential Temp Perturbation (K)')
plt.title(f'Warm Bubble at T = {t_end}s')
plt.xlabel('X Distance (m)')
plt.ylabel('Altitude (m)')

plt.savefig(f'{output_dir}/rising_bubble_3d_{t_end}.png')
print(f"[PLOTTING] Saved plot to '{output_dir}/rising_bubble_3d_{t_end}.png'")