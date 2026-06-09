import os

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

from suetes.slice2d.geometry import StaggeredGrid
from suetes.slice2d.euler import VerticalSlice
from suetes.slice2d.steppers import SISLStepper
from suetes.shared.driver import Simulation

output_dir = "suetes/plots/benchmarks"
os.makedirs(output_dir, exist_ok=True)

# ====================================================================
# 1. SETUP GRID & PHYSICS
# ====================================================================
nx, nz = 200, 100 
Lx, Lz = 20000.0, 10000.0
grid = StaggeredGrid(nx, nz, Lx, Lz, h_func=lambda x: 0.0)
grid.periodic_x = True

constants = {'g': 9.81, 'cp': 1004.0, 'cvd': 717.0, 'Rd': 287.0, 'p0': 100000.0}
# Isentropic background (theta = 300K)
physics = VerticalSlice(grid, constants, damp_height=8000.0, N_bv=0.0)

# ====================================================================
# 2. INITIAL STATE (Hydrostatic + Warm Bubble)
# ====================================================================
def get_initial_state(grid, physics):
    # 1. Background Potential Temperature
    theta_bg = 300.0
    
    # 2. Hydrostatic Exner Pressure: pi(z) = 1 - (g*z)/(cp*theta)
    pi_bg = 1.0 - (physics.c['g'] * grid.Z_m) / (physics.c['cp'] * theta_bg)
    
    # 3. Density from Ideal Gas Law
    # rho = (p0 / (Rd * theta)) * pi^(cv/Rd)
    rho_bg = (physics.c['p0'] / (physics.c['Rd'] * theta_bg)) * \
             (pi_bg ** (physics.c['cvd'] / physics.c['Rd']))
    
    # 4. Bubble Perturbation
    xc, zc = 10000.0, 2000.0
    r_bubble = 2000.0
    r = jnp.sqrt((grid.X_m - xc)**2 + (grid.Z_m - zc)**2)
    
    theta_pert = jnp.where(r < r_bubble, 2.0 * jnp.cos(0.5 * jnp.pi * r / r_bubble)**2, 0.0)
    
    return {
        'u': jnp.zeros_like(grid.X_u),
        'w': jnp.zeros_like(grid.X_w),
        'rho': rho_bg,
        'pi': pi_bg,
        'th_v': theta_bg + theta_pert,
        'eta_dot': jnp.zeros_like(grid.X_w)
    }

state = get_initial_state(grid, physics)

#====================================================================
# 3. RUN SIMULATION
# ====================================================================
dt = 0.5
stepper = SISLStepper(physics, dt)

def boundary_conditions(st, forcing):
    st['w'] = st['w'].at[:, 0].set(0.0)
    st['w'] = st['w'].at[:, -1].set(0.0)
    st['eta_dot'] = st['eta_dot'].at[:, 0].set(0.0)
    st['eta_dot'] = st['eta_dot'].at[:, -1].set(0.0)
    return st

def unified_step_fn(curr_state, step_idx):
    t_curr = step_idx * dt
    next_state = stepper.step(curr_state, t_curr, forcing=None, bc_fn=boundary_conditions)
    max_w = jnp.max(jnp.abs(next_state['w']))
    return next_state, max_w

t_end = 1000.0
sim = Simulation(step_fn=unified_step_fn, dt=dt)

print(f"[TEST 2D] Running SISL Bubble Test (dt={dt}s)...")
final_state = sim.run(state, t_start=0.0, t_end=t_end, chunk_steps=100)

# ====================================================================
# 4. PLOTTING
# ====================================================================
plt.figure(figsize=(12, 5))
th_pert = final_state['th_v'] - 300.0
plt.contourf(grid.X_m / 1000.0, grid.Z_m / 1000.0, th_pert, levels=20, cmap='RdBu_r')
plt.title(fr"SISL Rising Bubble: $\Delta \theta$ at T={t_end}s (dt={dt}s)")
plt.xlabel("x (km)")
plt.ylabel("z (km)")
plt.colorbar(label="Temperature Perturbation (K)")
plt.savefig(f'{output_dir}/rising_bubble_2d_{t_end}s.png', dpi=150, bbox_inches='tight')
print("[PLOTTING]Saved rising_bubble_2d.png")