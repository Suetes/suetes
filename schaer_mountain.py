import jax.numpy as jnp
import matplotlib.pyplot as plt

from suetes.core.grids import StaggeredGrid
from suetes.dynamics.euler import ICON2DSlice
from suetes.core.steppers import SISLStepper
from suetes.core.driver import Simulation

# ====================================================================
# 1. SETUP SLEVE GRID
# ====================================================================
nx, nz = 200, 60 
Lx, Lz = 100000.0, 30000.0
grid = StaggeredGrid(nx, nz, Lx, Lz, h_func=lambda x: 0.0)
grid.periodic_x = True
H_top = Lz

def schaer_sleve_components(x):
    hm = 250.0  
    a = 5000.0
    lam = 4000.0
    xc = x - 50000.0
    envelope = hm * jnp.exp(-(xc**2)/(a**2))
    
    h1 = 0.5 * envelope 
    h2 = 0.5 * envelope * jnp.cos(2.0 * jnp.pi * xc / lam) 
    return h1, h2

def sleve_decay(zeta, s, H_top):
    return jnp.sinh((H_top - zeta) / s) / jnp.sinh(H_top / s)

def apply_sleve(Z_flat, X_flat):
    h1, h2 = schaer_sleve_components(X_flat[:, 0])
    h1, h2 = h1[:, None], h2[:, None]
    s1, s2 = 15000.0, 2500.0  
    
    b1 = sleve_decay(Z_flat, s1, H_top)
    b2 = sleve_decay(Z_flat, s2, H_top)
    return Z_flat + h1 * b1 + h2 * b2

grid.Z_m = apply_sleve(grid.Z_m, grid.X_m)
grid.Z_u = apply_sleve(grid.Z_u, grid.X_u)
grid.Z_w = apply_sleve(grid.Z_w, grid.X_w)

hx_m = schaer_sleve_components(grid.X_m[:, 0])[0] + schaer_sleve_components(grid.X_m[:, 0])[1]
hx_m = hx_m[:, None]

# ====================================================================
# 2. INITIALIZE PHYSICS
# ====================================================================
constants = {'g': 9.81, 'cp': 1004.0, 'cvd': 717.0, 'Rd': 287.0, 'p0': 100000.0}
physics = ICON2DSlice(grid, constants, damp_height=22000.0, N_bv=0.01)
dt = 25.0
stepper = SISLStepper(physics, dt)

# ====================================================================
# INITIAL STATE
# ====================================================================
u_0 = 10.0
state = {
    'u': u_0 * jnp.ones_like(grid.X_u),
    'w': jnp.zeros_like(grid.X_w),
    'rho': physics.c['p0'] / (physics.c['Rd'] * physics.theta_bg) * (physics.pi_bg ** (physics.c['cvd'] / physics.c['Rd'])),
    'pi': physics.pi_bg,
    'th_v': physics.theta_bg
}

# Ensure initial vertical wind follows terrain to prevent acoustic shocks
z_xi_w = physics._get_metrics('w')['z_xi']
state['w'] = u_0 * z_xi_w

def boundary_conditions(st, forcing):
    # Kinematic bottom boundary condition
    dh_dx = physics._get_metrics('w')['z_xi'][:, 0]
    
    # Average 'u' from cell faces to the cell centers (w-faces) to match shapes
    u_at_w_face = physics.op.avg_u_to_m(st['u'])[:, 0]
    
    st['w'] = st['w'].at[:, 0].set(u_at_w_face * dh_dx)
    st['w'] = st['w'].at[:, -1].set(0.0)
    return st

print("Running Schaer Mountain Test (5 hours)...")

# 5 hours of simulation time
t_start = 0.0
t_end = 5.0 * 3600.0  

# Initialize the simulation driver
sim = Simulation(stepper, forcing_fn=None, bc_fn=boundary_conditions)

# Run the chunked simulation
# A chunk_steps of 100 is usually a good balance between fast host logging 
# and keeping the GPU busy.
final_state = sim.run(state, t_start, t_end, dt, chunk_steps=100)

# Plotting
plt.figure(figsize=(10, 5))
plt.contourf(grid.X_w / 1000.0, grid.Z_w / 1000.0, final_state['w'], levels=20, cmap='RdBu_r')
plt.plot(grid.X_m[:, 0] / 1000.0, hx_m[:, 0] / 1000.0, color='black', linewidth=2)
plt.fill_between(grid.X_m[:, 0] / 1000.0, 0, hx_m[:, 0] / 1000.0, color='gray')
plt.title("Schär Mountain Wave: Vertical Velocity (m/s)")
plt.xlabel("x (km)")
plt.ylabel("z (km)")
plt.colorbar()
plt.ylim(0, 15)
plt.xlim(25, 75)
plt.savefig('schaer_test.png', dpi=150)
print("Saved schaer_test.png")