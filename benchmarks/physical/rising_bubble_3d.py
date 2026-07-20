import os
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.steppers import build_dynamical_core
from suetes.shared.driver import Simulation

output_dir = "output/plots/benchmarks"
os.makedirs(output_dir, exist_ok=True)

# =====================================================================
# CONFIGURATION SWITCHES
# =====================================================================
CORE_TYPE = "sisl"  # Toggle to "sisl" or "split-explicit"
t_end = 1000.0 

dx, dy, dz = 10.0, 10.0, 10.0  # Resolution

# Dynamically calculate grid cells to preserve a 10 km x 10 km physical domain
domain_width = 10000.0
domain_height = 10000.0
nx = int(domain_width / dx)
ny = 3
nz = int(domain_height / dz)

# Base stable timestep and reference resolution (CFL condition)
ref_dx = 125.0
ref_dt = 2.5
dt = ref_dt * (min(dx, dy, dz) / ref_dx)

if CORE_TYPE.lower() == "sisl":
    core_kwargs = {"dt": dt, "nu_div_factor": 0.05, "nu_h_factor": 0.05, "damp_height": 7500.0, "max_damp": 0.05, "N_bv": 0.0,
    "solver_tol": 1e-6, "solver_maxiter": 20, "solver_restart": 20}
elif CORE_TYPE.lower() == "split-explicit":
    core_kwargs = {"dt": dt, "ns": 12, "nu_div_factor": 0.0, "nu_h_factor": 0.0, "damp_height": 7500.0, "max_damp": 0.05, "N_bv": 0.0}

grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_center=0.0, lon_center=0.0)
op = CGridOperator3D(grid)
constants = {'g': 9.81, 'cp': 1004.0, 'Rd': 287.0, 'cvd': 717.0, 'p0': 100000.0}

# --- INITIALIZE STATE ---
# Temporary physics instance to precompute background profiles cleanly
tmp_phys = Euler3D(grid, op, constants, dt=dt, N_bv=0.0)
bg_ref = {
    'rho': tmp_phys.c['p0'] / (tmp_phys.c['Rd'] * tmp_phys.theta_bg) * \
           (tmp_phys.pi_bg ** (tmp_phys.c['cvd'] / tmp_phys.c['Rd'])),
    'pi': tmp_phys.pi_bg, 'th_v': tmp_phys.theta_bg
}

state = {
    'u': jnp.zeros((nx+1, ny, nz)), 'v': jnp.zeros((nx, ny+1, nz)), 'w': jnp.zeros((nx, ny, nz+1)),
    'pi': bg_ref['pi'], 'eta_dot': jnp.zeros((nx, ny, nz+1)), 'rho': bg_ref['rho'],
}

# Inject the +2K Cosine-squared Warm Bubble (radius 1500m)
X, Y, Z = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_m, indexing='ij')
r = jnp.sqrt(X**2 + (Z - 2000.0)**2)
bubble = jnp.where(r <= 1500.0, 2.0 * jnp.cos(0.5 * jnp.pi * r / 1500.0)**2, 0.0)

state['th_v'] = bg_ref['th_v'] + bubble
state['rho'] = constants['p0'] / (constants['Rd'] * state['th_v']) * \
               (bg_ref['pi'] ** (constants['cvd'] / constants['Rd']))

# --- FACTORY INSTANTIATION ---
stepper, dt = build_dynamical_core(
    core_type=CORE_TYPE, grid=grid, operators=op, constants=constants,
    initial_state=state, **core_kwargs
)

def unified_step_fn(curr_state, step_idx):
    t_curr = step_idx * dt
    next_state = stepper.step(curr_state, t_curr, forcing=None, bc_fn=lambda x, f: x)
    max_w = jnp.max(jnp.abs(next_state['w']))
    return next_state, max_w

sim = Simulation(step_fn=unified_step_fn, dt=dt)

if __name__ == "__main__":
    print(f"\n[BENCHMARK] Launching {CORE_TYPE.upper()} Warm Bubble (dt={dt}s)...")
    final_state = sim.run(state, t_start=0.0, t_end=t_end, chunk_steps=20)

    # --- VISUALIZE RESULTS ---
    perturbation = final_state['th_v'][:, 1, :] - bg_ref['th_v'][:, 1, :]
    max_val = float(jnp.max(jnp.abs(perturbation)))
    levels = jnp.linspace(-max_val, max_val, 101)

    plt.figure(figsize=(10, 8))
    plt.contourf(grid.x_m / 1000.0, grid.z_m / 1000.0, perturbation.T, levels=levels, cmap='RdBu_r')
    plt.colorbar(label='Potential temperature perturbation (K)')
    # plt.title(f'{CORE_TYPE.capitalize()} warm bubble at T = {t_end}s')
    plt.xlabel('Distance (km)')
    plt.ylabel('Altitude (km)')

    img_path = f'{output_dir}/rising_bubble_{CORE_TYPE.lower()}_3d_{int(dx)}m_{t_end}s.png'
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    print(f"[PLOTTING] Saved plot to '{img_path}'")

    # --- INTEGRITY CHECKS ---
    cell_volumes = grid.dx * grid.dy * (grid.Z_w[:, :, 1:] - grid.Z_w[:, :, :-1])
    mass_error = jnp.abs(jnp.sum(final_state['rho'] * cell_volumes) - jnp.sum(state['rho'] * cell_volumes)) / jnp.sum(state['rho'] * cell_volumes)
    w_sym_error = jnp.max(jnp.abs(final_state['w'] - final_state['w'][::-1, :, :]))

    print(f"\n[VALIDATION] Integrity Metrics:")
    print(f"  -> Fractional Mass Error: {mass_error:.4e}")
    print(f"  -> Max Asymmetry in W:       {w_sym_error:.4e} m/s")