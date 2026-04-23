import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from suetes.core.grids import StaggeredGrid
from suetes.dynamics.euler import ICON2DSlice

# ====================================================================
# 1. SETUP GRID AND NATIVE OPERATOR
# ====================================================================
nx, nz = 200, 200
dx, dz = 100.0, 100.0
grid = StaggeredGrid(nx, nz, nx*dx, nz*dz, h_func=lambda x: 0.0)
grid.periodic_x = True

constants = {'g': 9.81, 'cp': 1004.0, 'cvd': 717.0, 'Rd': 287.0, 'p0': 100000.0}

# Initialize native physics purely to access its configured spatial operators
physics = ICON2DSlice(grid, constants)
op = physics.op

# ====================================================================
# 2. SOLID BODY ROTATION FIELD & BUBBLE
# ====================================================================
period = 1000.0
omega = 2.0 * jnp.pi / period
xc, zc = 10000.0, 10000.0

# Prescribed steady velocity field
X_u, Z_u = grid.X_u, grid.Z_u
X_w, Z_w = grid.X_w, grid.Z_w
u_vel = -omega * (Z_u - zc)
w_vel =  omega * (X_w - xc)

# Cosine-bell anomaly
r = jnp.sqrt((grid.X_m - 10000.0)**2 + (grid.Z_m - 5000.0)**2)
q_init = jnp.where(r < 2000.0, 2.0 * jnp.cos(0.5 * jnp.pi * r / 2000.0)**2, 0.0)

# ====================================================================
# 3. NATIVE TRANSPORT LOOP
# ====================================================================
dt = 0.2 # Courant number stability limit for 2D advection
num_steps = int(period / dt)

def calculate_native_divergence(q_val):
    """Uses the native TVD limiters and operators to compute flux divergence."""
    # 1. TVD Interpolation to faces
    q_u = op.tvd_interp_m_to_u(q_val, u_vel)
    q_w = op.tvd_interp_m_to_w(q_val, w_vel)
    
    # 2. Compute Fluxes
    flux_x = u_vel * q_u
    flux_z = w_vel * q_w
    
    # 3. Boundary Conditions (Rigid lid in Z)
    flux_z = flux_z.at[:, 0].set(0.0)
    flux_z = flux_z.at[:, -1].set(0.0)
    
    # 4. Divergence using native staggered derivatives
    div_x = op.diff_x_u_to_m(flux_x)
    div_z = op.diff_z_w_to_m(flux_z)
    return div_x + div_z

@jax.jit
def run_advection(q_start):
    def step_fn(q_n, _):
        # RK2 Time Integration using native spatial operators
        k1 = -calculate_native_divergence(q_n)
        q_star = q_n + dt * k1
        
        k2 = -calculate_native_divergence(q_star)
        q_next = q_n + 0.5 * dt * (k1 + k2)
        return q_next, None

    q_final, _ = jax.lax.scan(step_fn, q_start, jnp.arange(num_steps))
    return q_final

# ====================================================================
# 4. EXECUTE & VERIFY
# ====================================================================
print(f"Running Native Transport for 1 full revolution ({num_steps} steps)...")
q_final = run_advection(q_init)

mass_init = jnp.sum(q_init) * dx * dz
mass_final = jnp.sum(q_final) * dx * dz
print(f"Mass Error: {abs(mass_init - mass_final) / mass_init * 100:.4e} %")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
c1 = ax1.imshow(q_init.T, origin='lower', extent=[0, 20, 0, 20], cmap='RdBu_r', vmin=0, vmax=2)
ax1.set_title("Initial State")
ax2.imshow(q_final.T, origin='lower', extent=[0, 20, 0, 20], cmap='RdBu_r', vmin=0, vmax=2)
ax2.set_title("After 1 Revolution (Native TVD)")
plt.savefig('advection_benchmark.png', dpi=150)
print("Saved advection_benchmark.png")