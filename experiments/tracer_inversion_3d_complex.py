import os
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.boundaries import BenchmarkSponge
from suetes.regional3d.steppers import build_dynamical_core
from suetes.physics.base import PhysicsSuite
from suetes.shared.optimization import OptaxSolver

# =====================================================================
# CONFIGURATION & COMMAND-LINE ARGUMENTS
# =====================================================================
ap = argparse.ArgumentParser(description="Fully 3D Tracer Inversion Benchmark.")
ap.add_argument('--core', type=str, default='split-explicit', choices=['sisl', 'split-explicit'],
                help="Dynamical core type to run ('sisl' or 'split-explicit')")
ap.add_argument('--steps', type=int, default=100,
                help="Number of optimization steps")
args = ap.parse_args()

CORE_TYPE = args.core
total_opt_steps = args.steps

output_dir = "output/plots/tracer_inversion"
os.makedirs(output_dir, exist_ok=True)

print(f"=====================================================================")
print(f"[BENCHMARK] Running Fully 3D Tracer Inversion on CORE: {CORE_TYPE}")
print(f"=====================================================================")

# --- 1. SETUP DOMAIN & GRID ---
# nx, ny, nz = 50, 50, 20 is fully 3D with 50,000 grid points.
nx, ny, nz = 50, 50, 20
dx, dy, dz = 500.0, 500.0, 250.0

if CORE_TYPE.lower() == "sisl":
    dt = 10.0
elif CORE_TYPE.lower() == "split-explicit":
    dt = 2.0  # Stable timestep for split-explicit core
else:
    dt = 10.0

t_end = 1200.0  # 20 minutes of advection
num_steps = int(t_end / dt)

constants = {'g': 9.81, 'cp': 1004.0, 'Rd': 287.0, 'cvd': 717.0, 'p0': 100000.0}

# --- 2. DYNAMIC 3D TOPOGRAPHY (TWO PEAKS) ---
def terrain_profile(x, y):
    # Primary peak: Height 1.8 km at X = -6 km, Y = -2 km
    h0 = 1800.0
    ax, ay = 5000.0, 5000.0
    peak1 = h0 * jnp.exp(- (x + 6000.0)**2 / ax**2 - (y + 2000.0)**2 / ay**2)
    
    # Secondary peak: Height 1.0 km at X = 2 km, Y = 4 km
    h1 = 1000.0
    bx, by = 4000.0, 4000.0
    peak2 = h1 * jnp.exp(- (x - 2000.0)**2 / bx**2 - (y - 4000.0)**2 / by**2)
    
    return peak1 + peak2

grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_center=45.0, lon_center=0.0, h_func=terrain_profile)
op = CGridOperator3D(grid)

suite = PhysicsSuite()
suite.register_tracer('q_tr')

physics = Euler3D(grid, op, constants, dt=dt, N_bv=0.01, damp_height=4000.0, 
                  max_damp=0.5, nu_div_factor=0.0, nu_h_factor=0.0, physics_suite=suite)

# Reference background thermodynamics
bg_ref = {
    'rho': physics.c['p0'] / (physics.c['Rd'] * physics.theta_bg) * \
           (physics.pi_bg ** (physics.c['cvd'] / physics.c['Rd'])),
    'pi': physics.pi_bg,
    'th_v': physics.theta_bg
}

# --- 3. SHEARED & TURNING WIND FIELD ---
def get_wind_u(Z_coords):
    u_sfc, u_top = 5.0, 15.0
    H_top = nz * dz
    return u_sfc + (u_top - u_sfc) * jnp.clip(Z_coords / H_top, 0.0, 1.0)

def get_wind_v(Z_coords):
    v_max = 4.0
    H_top = nz * dz
    return v_max * jnp.sin(jnp.pi * jnp.clip(Z_coords / H_top, 0.0, 1.0))

sponge = BenchmarkSponge(nx, ny, sponge_depth=6, axes=('x', 'y'))

def bc_fn(state_in, forcing=None):
    ext_state = {
        'u': get_wind_u(grid.Z_u),
        'v': get_wind_v(grid.Z_v),
        'w': jnp.zeros_like(state_in['w']),
        'th_v': bg_ref['th_v'],
        'rho': bg_ref['rho'],
        'pi': bg_ref['pi'],
        'q_tr': jnp.zeros_like(state_in.get('q_tr', jnp.zeros_like(state_in['rho'])))
    }
    return sponge.blend(state_in, ext_state)

# --- 5. 3D GAUSSIAN TRACER plume generator ---
def create_state_with_tracer(x_km, y_km, z_km, amplitude):
    X_km = grid.x_m[:, None, None] / 1000.0
    Y_km = grid.y_m[None, :, None] / 1000.0
    Z_km = grid.Z_m / 1000.0
    
    sigma_x, sigma_y, sigma_z = 2.0, 2.0, 0.8
    q_tr = amplitude * jnp.exp(-(
        (X_km - x_km)**2 / (2.0 * sigma_x**2) +
        (Y_km - y_km)**2 / (2.0 * sigma_y**2) +
        (Z_km - z_km)**2 / (2.0 * sigma_z**2)
    ))
    
    return {
        'u': get_wind_u(grid.Z_u),
        'v': get_wind_v(grid.Z_v),
        'w': jnp.zeros((nx, ny, nz+1)),
        'pi': bg_ref['pi'],
        'eta_dot': jnp.zeros((nx, ny, nz+1)),
        'rho': bg_ref['rho'],
        'th_v': bg_ref['th_v'],
        'q_tr': q_tr
    }

# --- 6. DISCRETE MONITORING TOWERS (SENSORS) ---
# Towers downstream at different distances, lateral positions, and altitudes
# Relocated closer to the source and within the domain to avoid boundaries and sponges
sensor_locs = [
    (-2.0, -2.0, 1.0),
    (-2.0,  2.0, 1.0),
    (0.0,   0.0, 1.5),
    (2.0,  -1.0, 1.5),
    (2.0,   1.0, 2.0),
    (3.0,   0.0, 2.0)
]

sensor_indices = []
for sx, sy, sz in sensor_locs:
    grid_x = np.asarray(grid.x_m) / 1000.0
    grid_y = np.asarray(grid.y_m) / 1000.0
    grid_Z = np.asarray(grid.Z_m) / 1000.0
    
    i = np.argmin(np.abs(grid_x - sx))
    j = np.argmin(np.abs(grid_y - sy))
    k = np.argmin(np.abs(grid_Z[i, j, :] - sz))
    
    sensor_indices.append((int(i), int(j), int(k)))

print(f"[SENSORS] Configured {len(sensor_indices)} monitoring towers downstream at grid cells:")
for loc, idx in zip(sensor_locs, sensor_indices):
    print(f"  Physical position (x, y, z): {loc} km  ->  Grid Index (i, j, k): {idx}")

# --- 7. GENERATE TARGET OBSERVATIONS FROM TRUE plume ---
true_params = jnp.array([-8.0, -1.5, 2.0, 10.0]) # True (X_km, Y_km, Z_km, Amplitude)
print(f"\n[TARGET] Placing true source at X: {true_params[0]} km, Y: {true_params[1]} km, Z: {true_params[2]} km (Amp: {true_params[3]})")
true_initial_state = create_state_with_tracer(true_params[0], true_params[1], true_params[2], true_params[3])

# Configure dynamic core properties
if CORE_TYPE.lower() == "sisl":
    core_kwargs = {
        "dt": dt, "nu_div_factor": 0.0, "nu_h_factor": 0.0, 
        "damp_height": 4000.0, "max_damp": 0.5, "N_bv": 0.01
    }
elif CORE_TYPE.lower() == "split-explicit":
    core_kwargs = {
        "dt": dt, "ns": 6, "nu_div_factor": 0.0, "nu_h_factor": 0.0, 
        "damp_height": 4000.0, "max_damp": 0.5, "N_bv": 0.01
    }

# Build forward stepper
stepper_fwd, _ = build_dynamical_core(
    core_type=CORE_TYPE, grid=grid, operators=op, constants=constants,
    initial_state=true_initial_state, physics_suite=suite, **core_kwargs
)
if hasattr(stepper_fwd, "use_checkpointing"):
    stepper_fwd.use_checkpointing = False

@jax.jit
def generate_target_data(init_state):
    def scan_fn(s, _):
        next_s = stepper_fwd.step(s, 0.0, None, bc_fn)
        # Extract sensor time-series values
        vals = jnp.stack([next_s['q_tr'][i, j, k] for i, j, k in sensor_indices])
        return next_s, (vals, next_s['q_tr'])
    
    return jax.lax.scan(scan_fn, init_state, jnp.arange(num_steps))

print("[SIMULATION] Running true forward simulation to generate observations...")
t_sim_start = time.time()
true_final_state, (target_sensor_history, plume_history) = generate_target_data(true_initial_state)
jax.block_until_ready(target_sensor_history)
print(f"[SIMULATION] Done in {time.time() - t_sim_start:.2f}s")

# --- 8. ADJOINT INVERSE SOLVER ---
# Build adjoint stepper (compiled with checkpointing enabled to save memory during reverse-mode AD)
stepper_adj, _ = build_dynamical_core(
    core_type=CORE_TYPE, grid=grid, operators=op, constants=constants,
    initial_state=true_initial_state, physics_suite=suite, **core_kwargs
)
if hasattr(stepper_adj, "use_checkpointing"):
    stepper_adj.use_checkpointing = True

def objective_fn(params):
    x_km, y_km, z_km, A = params[0], params[1], params[2], params[3]
    state = create_state_with_tracer(x_km, y_km, z_km, A)
    
    def body(s, _):
        next_s = stepper_adj.step(s, 0.0, None, bc_fn)
        vals = jnp.stack([next_s['q_tr'][i, j, k] for i, j, k in sensor_indices])
        return next_s, vals
        
    final_state, sim_sensor_history = jax.lax.scan(body, state, jnp.arange(num_steps))
    
    # MSE between simulated and target sensor histories
    mse_loss = jnp.mean((sim_sensor_history - target_sensor_history)**2)
    return mse_loss, final_state

# Initial optimization parameters
# Placed intentionally far from the true source
guess_params = jnp.array([-4.0, 3.0, 3.5, 2.0]) 
print(f"[OPTIMIZATION] Initial guess source: X: {guess_params[0]} km, Y: {guess_params[1]} km, Z: {guess_params[2]} km (Amp: {guess_params[3]})")

lr_schedule = optax.cosine_decay_schedule(init_value=0.25, decay_steps=total_opt_steps, alpha=0.02)
optimizer = optax.adam(learning_rate=lr_schedule)

def constrain_bounds(p):
    # Bound the source location to the physical coordinates of the domain
    p = p.at[0].set(jnp.clip(p[0], -20.0, 20.0))
    p = p.at[1].set(jnp.clip(p[1], -20.0, 20.0))
    p = p.at[2].set(jnp.maximum(p[2], 0.1)) # Must remain above ground level (z >= 100 m)
    p = p.at[3].set(jnp.maximum(p[3], 0.0)) # Amplitude cannot be negative
    return p

solver = OptaxSolver(objective_fn, optimizer, has_aux=True)
optimal_params, history = solver.fit(guess_params, total_steps=total_opt_steps, bounds_fn=constrain_bounds)

print(f"\n=====================================================================")
print(f"[RESULTS] True Parameters:    X={true_params[0]:.2f}, Y={true_params[1]:.2f}, Z={true_params[2]:.2f}, Amp={true_params[3]:.2f}")
print(f"[RESULTS] Recovered Parameters: X={optimal_params[0]:.2f}, Y={optimal_params[1]:.2f}, Z={optimal_params[2]:.2f}, Amp={optimal_params[3]:.2f}")
print(f"=====================================================================")

# --- 9. VISUALIZATIONS ---
print("\n[PLOT] Generating visualizations...")
x_1d = np.asarray(grid.x_m) / 1000.0
y_1d = np.asarray(grid.y_m) / 1000.0
Z_m_np = np.asarray(grid.Z_m) / 1000.0

# 9a. Plume temporal evolution slices (2x4 panel grid)
# Find the slice indices closest to the true source
true_z_idx = int(np.argmin(np.abs(Z_m_np[nx//2, ny//2, :] - true_params[2])))
true_y_idx = int(np.argmin(np.abs(y_1d - true_params[1])))

# Select 4 snapshots equally spaced in time
snapshot_indices = [
    max(0, min(num_steps // 4 - 1, num_steps - 1)),
    max(0, min(num_steps // 2 - 1, num_steps - 1)),
    max(0, min(3 * num_steps // 4 - 1, num_steps - 1)),
    num_steps - 1
]
# Make unique in case num_steps is small
snapshot_indices = sorted(list(set(snapshot_indices)))
num_cols = len(snapshot_indices)

fig1, axs = plt.subplots(2, num_cols, figsize=(4 * num_cols + 1.5, 7.5), squeeze=False)

# Contour levels from 0.01 to the peak initial concentration (10.0)
levels = np.linspace(0.01, 10.0, 50)

terrain_2d = np.asarray(terrain_profile(grid.x_m[:, None], grid.y_m[None, :])) / 1000.0
terrain_y = np.asarray(terrain_profile(grid.x_m, np.asarray(grid.y_m)[true_y_idx])) / 1000.0
X_grid_padded = np.broadcast_to(x_1d[:, None], (nx, nz + 1))
Z_bottom = terrain_y
Z_slice_padded = np.concatenate([Z_bottom[:, None], Z_m_np[:, true_y_idx, :]], axis=1)

c_mappable = None

for col_idx, step_idx in enumerate(snapshot_indices):
    t_min = (step_idx + 1) * dt / 60.0
    plume_at_t = np.asarray(plume_history[step_idx])
    
    # ------------------ TOP ROW: Horizontal XY Slice ------------------
    ax_top = axs[0, col_idx]
    
    # Contour of plume concentration
    c_top = ax_top.contourf(x_1d, y_1d, plume_at_t[:, :, true_z_idx].T, levels=levels, cmap='Blues', extend='max', zorder=1)
    if c_mappable is None:
        c_mappable = c_top
        
    # Overlay terrain contours
    ax_top.contourf(x_1d, y_1d, terrain_2d.T, levels=20, cmap='binary', alpha=0.15, zorder=1.5)
    topo_contours = ax_top.contour(x_1d, y_1d, terrain_2d.T, levels=[0.2, 0.5, 0.8, 1.1, 1.4, 1.7], colors='black', alpha=0.20, linewidths=0.8, zorder=1.6)
    
    # Plot true source location
    ax_top.scatter([true_params[0]], [true_params[1]], color='green', marker='x', s=100, linewidths=2.0, label='True source' if col_idx == 0 else "", zorder=6)

    # Plot sensor locations
    sensor_x_np = [loc[0] for loc in sensor_locs]
    sensor_y_np = [loc[1] for loc in sensor_locs]
    ax_top.scatter(sensor_x_np, sensor_y_np, color='red', marker='^', s=80, label='Sensor towers', edgecolor='black', zorder=3)
    
    ax_top.set_title(f"t = {t_min:.1f} mins", fontsize=12)
    ax_top.set_xlim([-12.5, 12.5])
    ax_top.set_ylim([-12.5, 12.5])
    ax_top.set_aspect('equal')
    
    if col_idx == 0:
        ax_top.set_ylabel("y (km)", fontsize=11)
        ax_top.legend(loc='lower right', framealpha=0.9, fontsize=9)
    else:
        ax_top.set_yticklabels([])
        
    # ------------------ BOTTOM ROW: Vertical XZ Slice ------------------
    ax_bottom = axs[1, col_idx]
    
    # Pad plume at the bottom to extend contour to the terrain surface
    plume_slice_padded = np.concatenate([plume_at_t[:, true_y_idx, 0:1], plume_at_t[:, true_y_idx, :]], axis=1)
    
    # Contour of plume concentration
    ax_bottom.contourf(X_grid_padded.T, Z_slice_padded.T, plume_slice_padded.T, levels=levels, cmap='Blues', extend='max')
    
    # Plot topography cross-section
    ax_bottom.fill_between(x_1d, 0, terrain_y, color='gray', alpha=0.5, label='Terrain' if col_idx == 0 else "")
    
    # Plot true source location
    ax_bottom.scatter([true_params[0]], [true_params[2]], color='green', marker='x', s=100, linewidths=2.0, label='True source' if col_idx == 0 else "")
    
    ax_bottom.set_xlim([-12.5, 12.5])
    ax_bottom.set_ylim([0, 5.0])
    ax_bottom.set_xlabel("x (km)", fontsize=11)
    
    if col_idx == 0:
        ax_bottom.set_ylabel("Altitude z (km)", fontsize=11)
        ax_bottom.legend(loc='lower right', framealpha=0.9, fontsize=9)
    else:
        ax_bottom.set_yticklabels([])

# Add title to row panels
axs[0, 0].text(-11.5, 10.0, "Horizontal x-y slice", color='black', fontsize=11, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
axs[1, 0].text(-11.5, 4.3, "Vertical x-z slice", color='black', fontsize=11, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.suptitle(f"Forward tracer plume evolution", fontsize=16, y=0.98)
plt.tight_layout(rect=[0, 0, 0.88, 0.95])

# Add a single shared colorbar on the right
cbar_ax = fig1.add_axes([0.90, 0.15, 0.02, 0.70])
cbar = fig1.colorbar(c_mappable, cax=cbar_ax)
cbar.set_label('Tracer concentration', fontsize=11)

plt.savefig(f"{output_dir}/tracer_3d_complex_plume_{CORE_TYPE.lower()}.png", dpi=150)
plt.close()

# 9b. Plot Optimization Trajectories (Horizontal and Vertical paths)
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 6))

hx = [float(p[0]) for p in history['params']]
hy = [float(p[1]) for p in history['params']]
hz = [float(p[2]) for p in history['params']]

# Horizontal trajectory tracking (X-Y plane)
# Overlay topography filled contours and lines
terrain_2d = np.asarray(terrain_profile(grid.x_m[:, None], grid.y_m[None, :])) / 1000.0
ax3.contourf(x_1d, y_1d, terrain_2d.T, levels=20, cmap='binary', alpha=0.15, zorder=1.5)
topo_contours_tr = ax3.contour(x_1d, y_1d, terrain_2d.T, levels=[0.2, 0.5, 0.8, 1.1, 1.4, 1.7], colors='black', alpha=0.15, linewidths=0.8, zorder=1.6)
ax3.clabel(topo_contours_tr, inline=True, fmt='%1.1f km', fontsize=8)

sensor_x_np = [loc[0] for loc in sensor_locs]
sensor_y_np = [loc[1] for loc in sensor_locs]
ax3.scatter(sensor_x_np, sensor_y_np, color='red', marker='^', s=80, label='Sensor towers', edgecolor='black', zorder=3)
ax3.plot(hx, hy, marker='o', color='purple', linestyle='-', linewidth=2, markersize=4, label='Optimizer trajectory', zorder=4)
ax3.scatter([hx[0]], [hy[0]], color='orange', s=100, label='Initial guess', zorder=5)
ax3.scatter([hx[-1]], [hy[-1]], color='purple', marker='*', s=150, zorder=6, label='Recovered source')
ax3.scatter([true_params[0]], [true_params[1]], color='green', marker='x', s=120, zorder=7, label='True source')
ax3.set_title("Source location trajectory (horizontal x-y plane)")
ax3.set_xlabel("x (km)")
ax3.set_ylabel("y (km)")
ax3.set_xlim([-12.5, 12.5])
ax3.set_ylim([-12.5, 12.5])
ax3.grid(True)
ax3.legend()

# Vertical trajectory tracking (X-Z plane)
ax4.plot(hx, hz, marker='o', color='purple', linestyle='-', linewidth=2, markersize=4, label='Optimizer trajectory', zorder=4)
ax4.scatter([hx[0]], [hz[0]], color='orange', s=100, label='Initial guess', zorder=5)
ax4.scatter([hx[-1]], [hz[-1]], color='purple', marker='*', s=150, zorder=6, label='Recovered source')
ax4.scatter([true_params[0]], [true_params[2]], color='green', marker='x', s=120, zorder=7, label='True source')
# Plot terrain envelope
terrain_max = np.max(np.asarray(terrain_profile(grid.x_m[:, None], grid.y_m[None, :])), axis=1) / 1000.0
ax4.fill_between(x_1d, 0, terrain_max, color='gray', alpha=0.3, label='Max terrain profile')
ax4.set_title("Source location trajectory (vertical x-z plane)")
ax4.set_xlabel("x (km)")
ax4.set_ylabel("Altitude z (km)")
ax4.set_xlim([-12.5, 12.5])
ax4.set_ylim([0, 5.0])
ax4.grid(True)
ax4.legend()

plt.tight_layout()
plt.savefig(f"{output_dir}/tracer_3d_complex_optimization_{CORE_TYPE.lower()}.png", dpi=150)
plt.close()

# 9c. Sensor Observation Time Series comparison (Target vs Recovered)
fig3, axs = plt.subplots(2, 3, figsize=(15, 8))
axs = axs.flatten()

# Rerun forward with optimal_params to get simulated recovered history
optimal_state = create_state_with_tracer(optimal_params[0], optimal_params[1], optimal_params[2], optimal_params[3])
_, (recovered_sensor_history, _) = generate_target_data(optimal_state)

times_mins = np.arange(num_steps) * dt / 60.0

for s_idx in range(len(sensor_locs)):
    ax = axs[s_idx]
    ax.plot(times_mins, np.asarray(target_sensor_history[:, s_idx]), 'k-', linewidth=2.5, label='True observations')
    ax.plot(times_mins, np.asarray(recovered_sensor_history[:, s_idx]), 'r--', linewidth=2, label='Inferred plume')
    ax.set_title(f"Tower {s_idx+1} at {sensor_locs[s_idx]} km")
    ax.set_xlabel("Time (mins)")
    ax.set_ylabel("Tracer concentration")
    ax.grid(True)
    if s_idx == 0:
        ax.legend()

plt.suptitle(f"Time-series of concentration at monitoring towers", fontsize=16)
plt.tight_layout()
plt.savefig(f"{output_dir}/tracer_3d_complex_sensors_{CORE_TYPE.lower()}.png", dpi=150)
plt.close()

print(f"[DONE] Visualizations saved to {output_dir}/")
print(f"       - Snapshots plot: tracer_3d_complex_plume_{CORE_TYPE.lower()}.png")
print(f"       - Trajectory plot: tracer_3d_complex_optimization_{CORE_TYPE.lower()}.png")
print(f"       - Sensor series plot: tracer_3d_complex_sensors_{CORE_TYPE.lower()}.png")
