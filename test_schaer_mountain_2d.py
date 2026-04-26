import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

from suetes.slice2d.grids import StaggeredGrid
from suetes.slice2d.euler import VerticalSlice
from suetes.slice2d.steppers import SISLStepper
from suetes.shared.driver import Simulation
from suetes.shared.transforms import Sleve  # Import the Sleve transform

# ====================================================================
# 1. SETUP SLEVE GRID
# ====================================================================
nx, nz = 200, 60 
Lx, Lz = 100000.0, 30000.0

def schaer_h(x):
    hm = 250.0  
    a = 5000.0
    lam = 4000.0
    xc = x - 50000.0
    envelope = hm * jnp.exp(-(xc**2)/(a**2))
    # True Schär topography
    return envelope * (jnp.cos(jnp.pi * xc / lam)**2)

def schaer_h1(x):
    hm = 250.0  
    a = 5000.0
    xc = x - 50000.0
    envelope = hm * jnp.exp(-(xc**2)/(a**2))
    # Large scale component for SLEVE
    return 0.5 * envelope

# Pass the transform natively into the grid!
sleve_transform = Sleve(h1_func=schaer_h1, s1=15000.0, s2=2500.0)
grid = StaggeredGrid(nx, nz, Lx, Lz, h_func=schaer_h, transform=sleve_transform)
grid.periodic_x = True

# We still need hx_m for plotting the terrain fill
hx_m = schaer_h(grid.X_m[:, 0])

# ====================================================================
# 2. INITIALIZE PHYSICS
# ====================================================================
constants = {'g': 9.81, 'cp': 1004.0, 'cvd': 717.0, 'Rd': 287.0, 'p0': 100000.0}
physics = VerticalSlice(grid, constants, damp_height=22000.0, N_bv=0.01)
dt = 10.0
stepper = SISLStepper(physics, dt, nu_ratio=0.04)

# ====================================================================
# INITIAL STATE
# ====================================================================
u_0 = 10.0
state = {
    'u': u_0 * jnp.ones_like(grid.X_u),
    'w': jnp.zeros_like(grid.X_w),
    'eta_dot': jnp.zeros_like(grid.X_w),  # NEW: Track logical vertical motion
    'rho': physics.c['p0'] / (physics.c['Rd'] * physics.theta_bg) * (physics.pi_bg ** (physics.c['cvd'] / physics.c['Rd'])),
    'pi': physics.pi_bg,
    'th_v': physics.theta_bg
}

# Ensure initial vertical wind follows terrain to prevent acoustic shocks
z_xi_w = physics._get_metrics('w')['z_xi']
state['w'] = u_0 * z_xi_w

def boundary_conditions(st, forcing):
    dh_dx = physics._get_metrics('w')['z_xi'][:, 0]
    u_at_w_face = physics.op.avg_u_to_m(st['u'])[:, 0]
    
    st['w'] = st['w'].at[:, 0].set(u_at_w_face * dh_dx)
    st['w'] = st['w'].at[:, -1].set(0.0)
    
    # NEW: Kinematic bounds for logical motion
    if 'eta_dot' in st:
        st['eta_dot'] = st['eta_dot'].at[:, 0].set(0.0)
        st['eta_dot'] = st['eta_dot'].at[:, -1].set(0.0)
    return st

print("Running Schaer Mountain Test (4 hours)...")

# 6 hours of simulation time
t_start = 0.0
t_end = 4.0 * 3600.0  

# Initialize the simulation driver
sim = Simulation(stepper, forcing_fn=None, bc_fn=boundary_conditions)

# Run the chunked simulation
# A chunk_steps of 100 is usually a good balance between fast host logging 
# and keeping the GPU busy.
final_state = sim.run(state, t_start, t_end, dt, chunk_steps=100)

# Plotting
plt.figure(figsize=(10, 5))
plt.contourf(grid.X_w / 1000.0, grid.Z_w / 1000.0, final_state['w'], levels=50, cmap='coolwarm')
plt.plot(grid.X_m[:, 0] / 1000.0, hx_m / 1000.0, color='black', linewidth=2)
plt.fill_between(grid.X_m[:, 0] / 1000.0, 0, hx_m / 1000.0, color='gray')
plt.title("Schär Mountain Wave: Vertical Velocity (m/s)")
plt.xlabel("x (km)")
plt.ylabel("z (km)")
plt.colorbar()
plt.ylim(0, 15)
plt.xlim(25, 75)
plt.savefig('test_schaer_mountain_2d.png', dpi=150)
print("Saved test_schaer_mountain_2d.png")