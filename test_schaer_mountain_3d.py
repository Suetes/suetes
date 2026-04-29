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

# --- 1. SCHÄR MOUNTAIN PROFILE ---
def schaer_mountain(x, y):
    h0 = 250.0      # Maximum height of 250m
    a = 5000.0      # Envelope half-width of 5km
    lam = 4000.0    # Wavelength of 4km
    
    # The Schaer profile combines a Gaussian envelope with a cosine wave
    return h0 * jnp.exp(-(x / a)**2) * jnp.cos(jnp.pi * x / lam)**2

# --- 2. SETUP GRID & PHYSICS ---
# 100km domain, 20km top. 500m horizontal resolution, 400m vertical.
nx, ny, nz = 200, 3, 50
dx, dy, dz = 500.0, 500.0, 400.0  

grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_center=45.0, lon_center=0.0, h_func=schaer_mountain)
op = CGridOperator3D(grid)

constants = {'g': 9.81, 'cp': 1004.0, 'Rd': 287.0, 'cvd': 717.0, 'p0': 100000.0}

# Strong stratification, with a Rayleigh sponge in the top 8km to absorb gravity waves
physics = Euler3D(grid, op, constants, N_bv=0.01, damp_height=12000.0, max_damp=0.5)

# --- 3. INITIALIZE STATE ---
# Note: We initialize U to a constant 10 m/s everywhere!
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
sponge = DaviesSponge(grid, sponge_depth=8)

def bc_fn(state_in, forcing):
    # The exterior state MUST have the 10 m/s wind to feed the domain
    ext_state = {
        'u': jnp.ones_like(state_in['u']) * u_bg,
        'v': jnp.zeros_like(state_in['v']),
        'th_v': bg_ref['th_v'],
        'rho': bg_ref['rho'],
        'pi': bg_ref['pi']
    }
    return sponge.blend(state_in, ext_state)

def forcing_fn(state, t):
    return state

# A 4.0 second time step is generally safe for 500m resolution at 10m/s
dt = 4.0
stepper = SISLStepper3D(physics, dt, use_mass_fixer=False)
sim = Simulation(stepper, forcing_fn, bc_fn)

# --- 5. RUN SIMULATION ---
# We run for 2 hours (7200s) to allow the gravity waves to reach a steady state
t_end = 3600.0 
print("\nLaunching Schär Mountain Benchmark...")
final_state = sim.run(state, t_start=0.0, t_end=t_end, dt=dt, chunk_steps=50)

# --- 6. VISUALIZE RESULTS ---
print("\nPlotting final state...")
# Extract the vertical velocity (w) down the middle of the Y-axis
w_slice = final_state['w'][:, 1, :-1] # Drop the top staggered level for plotting

# Crop the X-axis for plotting just the central 50km
x_start_idx = (nx // 4)
x_end_idx = 3 * (nx // 4)
x_plot_1d = grid.x_m[x_start_idx:x_end_idx] / 1000.0 # Convert to km

# Create 2D coordinate grids for Matplotlib
X_plot, _ = jnp.meshgrid(x_plot_1d, grid.z_m, indexing='ij')

# FIX: Extract the 2D slice of the true PHYSICAL heights
Z_plot = grid.Z_m[x_start_idx:x_end_idx, 1, :] / 1000.0 # Convert to km

plt.figure(figsize=(12, 6))
# Pass the 2D X and Z arrays to contourf so it warps the grid to match the terrain!
contour = plt.contourf(X_plot, Z_plot, w_slice[x_start_idx:x_end_idx, :], 
                       levels=jnp.linspace(-2.0, 2.0, 41), cmap='RdBu_r', extend='both')
plt.colorbar(contour, label='Vertical Velocity W (m/s)')
plt.title(f'Schär Mountain Waves at T = {t_end}s')
plt.xlabel('Distance (km)')
plt.ylabel('Altitude (km)')

# Plot the mountain profile
mountain_terrain = schaer_mountain(grid.x_m[x_start_idx:x_end_idx], 0.0) / 1000.0
plt.fill_between(x_plot_1d, 0, mountain_terrain, color='black')

plt.savefig('schaer_mountain_result.png', dpi=150, bbox_inches='tight')
print("Saved plot to 'schaer_mountain_result.png'")