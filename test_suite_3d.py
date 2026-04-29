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

# --- 1. SETUP GRID & PHYSICS ---
# Using a 10km x 10km domain. We use a pseudo-2D setup (ny=3) just so it runs 
# blazingly fast on your local machine for the first test, but the math is fully 3D.
nx, ny, nz = 80, 3, 80
dx, dy, dz = 125.0, 125.0, 125.0  # 125m resolution

grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_center=0.0, lon_center=0.0)
op = CGridOperator3D(grid)

constants = {'g': 9.81, 'cp': 1004.0, 'Rd': 287.0, 'cvd': 717.0, 'p0': 100000.0}
physics = Euler3D(grid, op, constants, N_bv=0.00, damp_height=7500.0, max_damp=0.5,
                  nu_h=0.0, nu_v=0.0)

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

# 1. Update the potential temperature
state['th_v'] = bg_ref['th_v'] + bubble

# 2. UPDATE THE DENSITY to maintain pressure equilibrium!
state['rho'] = physics.c['p0'] / (physics.c['Rd'] * state['th_v']) * \
               (bg_ref['pi'] ** (physics.c['cvd'] / physics.c['Rd']))

# --- 3. BOUNDARIES & STEPPER ---
# Apply a sponge layer 5 cells deep to absorb acoustic waves at the edges
sponge = DaviesSponge(grid, sponge_depth=5)

def bc_fn(state_in, forcing):
    # Only use the lateral sponge!
    ext_state = {
        'u': jnp.zeros_like(state_in['u']),
        'v': jnp.zeros_like(state_in['v']),
        'th_v': bg_ref['th_v'],
        'rho': bg_ref['rho'],
        'pi': bg_ref['pi']
    }
    return sponge.blend(state_in, ext_state)

def forcing_fn(state, t):
    return state

# Use a 2.5 second time step. High resolution means we need a smaller dt.
dt = 2.5
stepper = SISLStepper3D(physics, dt, use_mass_fixer=False)
sim = Simulation(stepper, forcing_fn, bc_fn)

# --- 4. RUN SIMULATION ---
t_end = 600.0 # Run for 10 minutes of physical time
print("\nLaunching Warm Bubble Benchmark...")
final_state = sim.run(state, t_start=0.0, t_end=t_end, dt=dt, chunk_steps=20)

# --- 5. VISUALIZE RESULTS ---
print("\nPlotting final state...")
# Extract a 2D slice down the middle of the Y-axis
th_v_slice = final_state['th_v'][:, 1, :]
th_v_bg_slice = bg_ref['th_v'][:, 1, :]
perturbation = th_v_slice - th_v_bg_slice

plt.figure(figsize=(10, 8))
# Transpose for plotting (Z on y-axis, X on x-axis)
plt.contourf(grid.x_m, grid.z_m, perturbation.T, levels=20, cmap='magma')
plt.colorbar(label='Potential Temp Perturbation (K)')
plt.title(f'Warm Bubble at T = {t_end}s')
plt.xlabel('X Distance (m)')
plt.ylabel('Altitude (m)')
plt.savefig('warm_bubble_result.png')
print("Saved plot to 'warm_bubble_result.png'")