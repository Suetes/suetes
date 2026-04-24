import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from suetes.core.grids import StaggeredGrid
from suetes.dynamics.euler import ICON2DSlice
from suetes.core.steppers import SISLStepper
from suetes.core.driver import Simulation

# ====================================================================
# 1. SETUP GRID & PHYSICS
# ====================================================================
nx, nz = 200, 100 
Lx, Lz = 20000.0, 10000.0
grid = StaggeredGrid(nx, nz, Lx, Lz, h_func=lambda x: 0.0)
grid.periodic_x = True

constants = {'g': 9.81, 'cp': 1004.0, 'cvd': 717.0, 'Rd': 287.0, 'p0': 100000.0}
# Isentropic background (theta = 300K)
physics = ICON2DSlice(grid, constants, damp_height=8000.0, N_bv=0.0)

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
        'th_v': theta_bg + theta_pert
    }

state = get_initial_state(grid, physics)

# ====================================================================
# 3. RUN SIMULATION
# ====================================================================
dt = 0.5  # SISL allows much larger dt than HE-VI (which used 0.15s)
stepper = SISLStepper(physics, dt)

def boundary_conditions(st, forcing):
    # Rigid wall at top and bottom
    st['w'] = st['w'].at[:, 0].set(0.0)
    st['w'] = st['w'].at[:, -1].set(0.0)
    return st

# 1000 seconds of simulation
t_end = 1000.0
sim = Simulation(stepper, forcing_fn=None, bc_fn=boundary_conditions)

print(f"Running SISL Bubble Test (dt={dt}s)...")
final_state = sim.run(state, 0.0, t_end, dt, chunk_steps=100)

# ====================================================================
# 4. PLOTTING
# ====================================================================
plt.figure(figsize=(12, 5))
th_pert = final_state['th_v'] - 300.0
plt.contourf(grid.X_m / 1000.0, grid.Z_m / 1000.0, th_pert, levels=20, cmap='RdBu_r')
plt.title(f"SISL Rising Bubble: $\Delta \\theta$ at T={t_end}s (dt={dt}s)")
plt.xlabel("x (km)")
plt.ylabel("z (km)")
plt.colorbar(label="Temperature Perturbation (K)")
plt.savefig('bubble_sisl.png', dpi=150)
print("Saved bubble_sisl.png")