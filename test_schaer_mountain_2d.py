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
# 1. SETUP GRID
# ====================================================================
nx, nz = 600, 50            # Widened domain to prevent wave wrap-around
Lx, Lz = 300000.0, 20000.0  # 300km wide, 20km high

def schaer_h(x):
    hm = 250.0  
    a = 5000.0
    lam = 4000.0
    xc = x - 100000.0       # Center the mountain at x=100km
    envelope = hm * jnp.exp(-(xc**2)/(a**2))
    return envelope * (jnp.cos(jnp.pi * xc / lam)**2)

# Standard Gal-Chen terrain following
grid = StaggeredGrid(nx, nz, Lx, Lz, h_func=schaer_h)
grid.periodic_x = True

hx_m = schaer_h(grid.X_m[:, 0])

# ====================================================================
# 2. INITIALIZE PHYSICS
# ====================================================================
constants = {'g': 9.81, 'cp': 1004.0, 'cvd': 717.0, 'Rd': 287.0, 'p0': 100000.0}
physics = VerticalSlice(grid, constants, damp_height=12000.0, N_bv=0.01)

dt = 4.0
# Add a tiny bit of diffusion to smooth the grid-scale noise at the sharp peaks
stepper = SISLStepper(physics, dt, nu_ratio=0.0)

# ====================================================================
# 3. INITIAL STATE
# ====================================================================
u_bg = 10.0
state = {
    'u': u_bg * jnp.ones_like(grid.X_u),
    'w': jnp.zeros_like(grid.X_w),
    'eta_dot': jnp.zeros_like(grid.X_w),
    'rho': physics.c['p0'] / (physics.c['Rd'] * physics.theta_bg) * (physics.pi_bg ** (physics.c['cvd'] / physics.c['Rd'])),
    'pi': physics.pi_bg,
    'th_v': physics.theta_bg
}

# ====================================================================
# 4. BOUNDARIES & SPONGE
# ====================================================================
sponge_depth = 8
x_idx = jnp.arange(nx, dtype=jnp.float32)
dist_x = jnp.minimum(x_idx, nx - x_idx)
weight_x = jnp.where(dist_x < sponge_depth, jnp.cos(0.5 * jnp.pi * dist_x / sponge_depth)**2, 0.0)
mask_x = weight_x[:, None] # Broadcast to 2D

def boundary_conditions(st, forcing):
    ext_state = {
        'u': u_bg * jnp.ones_like(st['u']),
        'th_v': physics.theta_bg,
        'rho': physics.c['p0'] / (physics.c['Rd'] * physics.theta_bg) * (physics.pi_bg ** (physics.c['cvd'] / physics.c['Rd'])),
        'pi': physics.pi_bg
    }
    
    blended = {}
    # Strictly exclude w and eta_dot to protect the kinematic boundary constraint!
    blend_vars = ['u', 'th_v', 'pi', 'rho']
    
    for k in st.keys():
        if k in blend_vars and k in ext_state:
            m = jnp.pad(mask_x, ((0, 1), (0, 0)), mode='edge') if k == 'u' else mask_x
            blended[k] = (1.0 - m) * st[k] + m * ext_state[k]
        else:
            blended[k] = st[k]
            
    return blended

def step_fn(curr_state, step_idx):
    t_curr = step_idx * dt
    next_state = stepper.step(curr_state, t_curr, forcing=None, bc_fn=boundary_conditions)
    max_w = jnp.max(jnp.abs(next_state['w']))
    return next_state, max_w

# ====================================================================
# 5. RUN SIMULATION
# ====================================================================
t_start = 0.0
t_end = 7200.0  
print("[TEST 2D] Running Schär Mountain Test (dt={dt}, T={t_end}s)...")

sim = Simulation(step_fn=step_fn, dt=dt)
final_state = sim.run(state, t_start, t_end, chunk_steps=50)

# ====================================================================
# 6. PLOTTING
# ====================================================================
plt.figure(figsize=(12, 6))

x_plot_1d = grid.X_m[:, 0] / 1000.0 

# Match the 50km window of the 3D script. 
# Mountain is at 100km, so we plot from 75km to 125km.
x_start_idx = int(75000.0 / grid.dx)
x_end_idx = int(125000.0 / grid.dx)

contour = plt.contourf(
    grid.X_w[x_start_idx:x_end_idx, :] / 1000.0,    
    grid.Z_w[x_start_idx:x_end_idx, :] / 1000.0,    
    final_state['w'][x_start_idx:x_end_idx, :],     
    levels=jnp.linspace(-2.0, 2.0, 41),
    cmap='RdBu_r', 
    extend='both'
)

plt.fill_between(x_plot_1d[x_start_idx:x_end_idx], 0, hx_m[x_start_idx:x_end_idx] / 1000.0, color='black')
plt.colorbar(contour, label='Vertical Velocity W (m/s)')
plt.title(f"Schär Mountain Wave: Vertical Velocity at T={t_end}s")
plt.xlabel("Distance (km)")
plt.ylabel("Altitude (km)")

plt.savefig(f'{output_dir}/schaer_mountain_2d_{t_end}.png', dpi=150, bbox_inches='tight')
print("[PLOTTING] Saved plot to '{output_dir}/schaer_mountain_2d_{t_end}.png'")