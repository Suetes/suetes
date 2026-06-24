import os
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.steppers import build_dynamical_core
from suetes.regional3d.boundaries import BenchmarkSponge
from suetes.shared.driver import Simulation

output_dir = "suetes/plots/inversion"
os.makedirs(output_dir, exist_ok=True)

# =====================================================================
# CONFIGURATION SWITCHES
# =====================================================================
CORE_TYPE = "split-explicit"  # Toggle to "sisl" or "split-explicit"

# --- 1. SCHÄR MOUNTAIN PROFILE ---
def schaer_mountain(x, y):
    h0 = 250.0      
    a = 5000.0      
    lam = 4000.0    
    return h0 * jnp.exp(-(x / a)**2) * jnp.cos(jnp.pi * x / lam)**2

# --- 2. SETUP GRID & PHYSICS ---
nx, ny, nz = 300, 3, 50  
dx, dy, dz = 500.0, 500.0, 400.0  
t_end = 7200.0  
dt = 4.0
num_steps = int(t_end / dt)

grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_center=45.0, lon_center=0.0, h_func=schaer_mountain)
op = CGridOperator3D(grid)
constants = {'g': 9.81, 'cp': 1004.0, 'Rd': 287.0, 'cvd': 717.0, 'p0': 100000.0}

physics = Euler3D(grid, op, constants, dt=dt, N_bv=0.01, damp_height=12000.0, max_damp=0.5,
                  nu_div_factor=0.0, nu_h_factor=0.0, physics_suite=None)

# --- 3. INITIALIZE STATE & BOUNDARIES ---
u_bg = 10.0

bg_ref = {
    'rho': physics.c['p0'] / (physics.c['Rd'] * physics.theta_bg) * \
           (physics.pi_bg ** (physics.c['cvd'] / physics.c['Rd'])),
    'pi': physics.pi_bg,
    'th_v': physics.theta_bg
}

initial_state = {
    'u': jnp.ones((nx+1, ny, nz)) * u_bg,
    'v': jnp.zeros((nx, ny+1, nz)),
    'w': jnp.zeros((nx, ny, nz+1)),
    'pi': bg_ref['pi'],
    'eta_dot': jnp.zeros((nx, ny, nz+1)),
    'rho': bg_ref['rho'],
    'th_v': bg_ref['th_v']
}

x_sponge = BenchmarkSponge(nx=nx, sponge_depth=10, axes=('x',))

def bc_fn(state_in, forcing=None):
    ext_state = {
        'u': jnp.ones_like(state_in['u']) * u_bg,
        'v': jnp.zeros_like(state_in['v']),
        'th_v': bg_ref['th_v'],
        'rho': bg_ref['rho'],
        'pi': bg_ref['pi']
    }
    return x_sponge.blend(state_in, ext_state)

# --- 4. BUILD DYNAMICAL CORE ---
if CORE_TYPE.lower() == "sisl":
    core_kwargs = {
        "dt": dt, "nu_div_factor": 0.0, "nu_h_factor": 0.0, 
        "damp_height": 12000.0, "max_damp": 0.5, "N_bv": 0.01
    }
elif CORE_TYPE.lower() == "split-explicit":
    core_kwargs = {
        "dt": dt, "ns": 6, "nu_div_factor": 0.0, "nu_h_factor": 0.0, 
        "damp_height": 12000.0, "max_damp": 0.5, "N_bv": 0.01
    }

stepper, _ = build_dynamical_core(
    core_type=CORE_TYPE, grid=grid, operators=op, constants=constants,
    initial_state=initial_state, **core_kwargs
)

sim = Simulation(step_fn=stepper.step, dt=dt)

# --- 5. THE ADJOINT METRIC FUNCTION ---
x_t0, x_t1 = 190, 220
z_t0, z_t1 = 8, 20

def compute_forecast_metric(x0_state):
    # Utilize the shared differentiable driver
    final_state = sim.run_differentiable(x0_state, 0.0, t_end, bc_fn=bc_fn, chunk_steps=50)
    
    w_final = final_state['w']
    target_energy = w_final[x_t0:x_t1, 1, z_t0:z_t1] ** 2
    return jnp.sum(target_energy)

# --- 6. RUN SENSITIVITIES ---
print(f"\n[ADJOINT] Compiling and running ideal forward/backward trajectories ({CORE_TYPE.upper()})...")
start_time = time.time()

adjoint_fn = jax.value_and_grad(compute_forecast_metric)
J_0, sensitivities = adjoint_fn(initial_state)

J_0 = J_0.block_until_ready()
jax.tree_util.tree_map(lambda x: x.block_until_ready(), sensitivities)

sensitivities_cpu = jax.device_get(sensitivities)
print(f"[ADJOINT] Completed in {(time.time() - start_time):.2f} seconds.")
print(f"[ADJOINT] Baseline target energy (J_0): {float(J_0):.4e}")

# --- 7. VISUALIZE THE COMET TAIL ---
print("\n[PLOT] Generating sensitivities plots...")

# Run a quick pure forward pass just to get the final state for plotting the box
# We use the fast, uncheckpointed loop for this
def fast_forward(state, _):
    return stepper.step(state, 0.0, None, bc_fn), None
final_fwd_state, _ = jax.lax.scan(fast_forward, initial_state, jnp.arange(num_steps))

# Axis setup
x_plot_1d = grid.x_m / 1000.0
mountain_terrain = schaer_mountain(grid.x_m, 0.0) / 1000.0

def plot_field(data, title, filename, cmap='RdBu_r', add_box=False):
    plt.figure(figsize=(14, 6))
    vmax = float(jnp.max(jnp.abs(data)))
    
    nz_data = data.shape[1]
    
    if nz_data == nz + 1:
        # W-grid (Interfaces): Goes all the way to the surface
        Z_plot_curr = grid.Z_w[:, 1, :] / 1000.0  
        X_plot_curr, _ = jnp.meshgrid(x_plot_1d, jnp.arange(nz_data), indexing='ij')
        data_to_plot = data
    else:
        # Mass-grid (Centers): Pad the bottom to stretch colors to the terrain
        data_to_plot = jnp.concatenate([data[:, :1], data], axis=1)
        Z_plot_curr = jnp.concatenate([grid.Z_w[:, 1, :1] / 1000.0, grid.Z_m[:, 1, :] / 1000.0], axis=1)
        X_plot_curr, _ = jnp.meshgrid(x_plot_1d, jnp.arange(nz_data + 1), indexing='ij')

    contour = plt.contourf(X_plot_curr, Z_plot_curr, data_to_plot, levels=jnp.linspace(-vmax, vmax, 41), cmap=cmap, extend='both')
    plt.colorbar(contour)
    plt.title(title)
    plt.xlabel('Distance (km)')
    plt.ylabel('Altitude (km)')
    plt.fill_between(x_plot_1d, 0, mountain_terrain, color='black')

    if add_box:
        Z_w_plot = grid.Z_w[:, 1, :] / 1000.0
        rect_x = x_plot_1d[x_t0]
        rect_z = Z_w_plot[x_t0, z_t0]
        width = x_plot_1d[x_t1] - x_plot_1d[x_t0]
        height = Z_w_plot[x_t0, z_t1] - Z_w_plot[x_t0, z_t0]
        rect = patches.Rectangle((rect_x, rect_z), width, height, linewidth=2, edgecolor='k', facecolor='none', linestyle='--')
        plt.gca().add_patch(rect)
        plt.text(rect_x + width/2, rect_z + height + 0.2, 'Target $J$', color='k', ha='center')

    plt.xlim([-50, 50])
    plt.ylim([0, 12])
    plt.savefig(f'{output_dir}/{filename}', dpi=150, bbox_inches='tight')
    plt.close()

# Forward State at T=2h (Where the waves are)
plot_field(final_fwd_state['w'][:, 1, :], 
           rf'Forward state $w$ at $T = {t_end/3600}h$', 
           f'schaer_fwd_w_{CORE_TYPE.lower()}.png', add_box=True)

# Sensitivity to Temperature at T=0 (Where the signal came from)
plot_field(sensitivities_cpu['th_v'][:, 1, :], 
           rf'Adjoint sensitivity ($\partial J / \partial \theta_v$) at T = 0 ({CORE_TYPE.upper()})', 
           f'schaer_adj_th_v_{CORE_TYPE.lower()}.png')

# Sensitivity to Horizontal Wind at T=0
u_sens_m = 0.5 * (sensitivities_cpu['u'][:-1, 1, :] + sensitivities_cpu['u'][1:, 1, :])
plot_field(u_sens_m, 
           rf'Adjoint sensitivity ($\partial J / \partial u$) at T = 0 ({CORE_TYPE.upper()})', 
           f'schaer_adj_u_{CORE_TYPE.lower()}.png')

print("[PLOT] Success! Check the inversion folder.")


# --- 8. THE PERTURBATION EXPERIMENT ---
print("\n[EXPERIMENT] Testing optimal vs. random perturbations...")

sens_th_v = sensitivities_cpu['th_v']

# Create the optimal perturbation (Gradient Ascent)
max_sens = jnp.max(jnp.abs(sens_th_v))
optimal_pert = (sens_th_v / max_sens) * 1.0

# 1. Gather configurations to test
configs_to_test = [("Optimal perturbation", optimal_pert)]

# Generate 3 Random Perturbations with the EXACT same L2 norm
key = jax.random.PRNGKey(42)
target_norm = jnp.linalg.norm(optimal_pert)

for i in range(1, 4):
    key, subkey = jax.random.split(key)
    random_pert_raw = jax.random.normal(subkey, optimal_pert.shape)
    random_pert = random_pert_raw * (target_norm / jnp.linalg.norm(random_pert_raw))
    configs_to_test.append((f"Random perturbation {i}", random_pert))

# 2. Function to evaluate a given perturbation
def evaluate_perturbation(pert_th_v):
    perturbed_state = initial_state.copy()
    
    # Apply the perturbation to the initial temperature
    perturbed_state['th_v'] = initial_state['th_v'] + pert_th_v
    
    # Update density (rho) using the Equation of State to maintain acoustic balance
    perturbed_state['rho'] = constants['p0'] / (constants['Rd'] * perturbed_state['th_v']) * \
                             (perturbed_state['pi'] ** (constants['cvd'] / constants['Rd']))
                             
    # Run the fast forward solver
    final_pert_state, _ = jax.lax.scan(fast_forward, perturbed_state, jnp.arange(num_steps))
    
    # Calculate J
    w_final = final_pert_state['w']
    target_energy = w_final[x_t0:x_t1, 1, z_t0:z_t1] ** 2
    return float(jnp.sum(target_energy)), final_pert_state

# 3. Pre-calculate the baseline and optimal vmax for a unified color scale
baseline_w = final_fwd_state['w'][:, 1, :]
_, final_opt_state = evaluate_perturbation(optimal_pert)
opt_diff_w = final_opt_state['w'][:, 1, :] - baseline_w
vmax_shared = float(jnp.max(jnp.abs(opt_diff_w))) * 1.1

# 4. Run and Plot all configurations
fig, axs = plt.subplots(2, 2, figsize=(20, 12))
axs = axs.flatten()

Z_w_plot = grid.Z_w[:, 1, :] / 1000.0
X_plot_curr, _ = jnp.meshgrid(x_plot_1d, jnp.arange(nz + 1), indexing='ij')

for idx, (title, pert_th_v) in enumerate(configs_to_test):
    print(f"Evaluating {title}...")
    J_val, final_pert_state = evaluate_perturbation(pert_th_v)
    print(f"  -> {title}: J = {J_val:.6e} (Amplification: {((J_val / float(J_0)) - 1.0) * 100:.1f}%)")
    
    # Calculate the difference field: perturbed - baseline
    w_diff = final_pert_state['w'][:, 1, :] - baseline_w
    
    ax = axs[idx]
    
    contour = ax.contourf(X_plot_curr, Z_w_plot, w_diff, 
                          levels=jnp.linspace(-vmax_shared, vmax_shared, 41), 
                          cmap='RdBu_r', extend='both')
    
    ax.fill_between(x_plot_1d, 0, mountain_terrain, color='black')
    
    # Draw Target Box
    rect_x = x_plot_1d[x_t0]
    rect_z = Z_w_plot[x_t0, z_t0]
    width = x_plot_1d[x_t1] - x_plot_1d[x_t0]
    height = Z_w_plot[x_t0, z_t1] - Z_w_plot[x_t0, z_t0]
    rect = patches.Rectangle((rect_x, rect_z), width, height, 
                             linewidth=2, edgecolor='k', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    
    ax.set_title(f"{title}\nTarget energy (J): {J_val:.2f}")
    ax.set_xlim([-50, 50])
    ax.set_ylim([0, 12])
    
    if idx >= 2: ax.set_xlabel('Distance (km)')
    if idx % 2 == 0: ax.set_ylabel('Altitude (km)')

fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
fig.colorbar(contour, cax=cbar_ax, label=r'Change in $w$ (m/s)')

plt.savefig(f'{output_dir}/schaer_adjoint_validation_{CORE_TYPE.lower()}.png', dpi=150, bbox_inches='tight')
print(f"\nValidation complete. Saved to {output_dir}/schaer_adjoint_validation_{CORE_TYPE.lower()}.png")

# --- 9. THE FORWARD PERTURBATION COMPARISON PLOT ---
print("\n[PLOT] Generating forward states comparison plot...")
fig2, axs2 = plt.subplots(2, 2, figsize=(20, 12))
axs2 = axs2.flatten()

# Pre-calculate the maximum vertical velocity across the optimal run to define a unified color scale
vmax_fwd_shared = float(jnp.max(jnp.abs(final_opt_state['w'][:, 1, :]))) * 1.1

for idx, (title, pert_th_v) in enumerate(configs_to_test):
    J_val, final_pert_state = evaluate_perturbation(pert_th_v)
    
    ax = axs2[idx]
    contour2 = ax.contourf(X_plot_curr, Z_w_plot, final_pert_state['w'][:, 1, :], 
                           levels=jnp.linspace(-vmax_fwd_shared, vmax_fwd_shared, 41), 
                           cmap='RdBu_r', extend='both')
    
    ax.fill_between(x_plot_1d, 0, mountain_terrain, color='black')
    
    # Draw Target Box
    rect = patches.Rectangle((rect_x, rect_z), width, height, 
                             linewidth=2, edgecolor='k', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    
    ax.set_title(f"{title}\nTarget energy (J): {J_val:.2f}")
    ax.set_xlim([-50, 50])
    ax.set_ylim([0, 12])
    
    if idx >= 2: ax.set_xlabel('Distance (km)')
    if idx % 2 == 0: ax.set_ylabel('Altitude (km)')

fig2.subplots_adjust(right=0.92)
cbar_ax2 = fig2.add_axes([0.94, 0.15, 0.02, 0.7])
fig2.colorbar(contour2, cax=cbar_ax2, label=r'Vertical velocity $w$ (m/s)')

plt.savefig(f'{output_dir}/schaer_forward_validation_{CORE_TYPE.lower()}.png', dpi=150, bbox_inches='tight')
print(f"Forward comparison complete. Saved to {output_dir}/schaer_forward_validation_{CORE_TYPE.lower()}.png")

# --- 10. THE INITIAL PERTURBATION COMPARISON PLOT ---
print("\n[PLOT] Generating initial perturbation comparison plot...")
fig3, axs3 = plt.subplots(2, 2, figsize=(20, 12))
axs3 = axs3.flatten()

# Pre-calculate a unified color scale for the perturbations
vmax_pert_shared = max(float(jnp.max(jnp.abs(p))) for _, p in configs_to_test) * 1.1

# For mass points, pad the bottom to align properly with the terrain
Z_m_plot = jnp.concatenate([grid.Z_w[:, 1, :1] / 1000.0, grid.Z_m[:, 1, :] / 1000.0], axis=1)
X_m_plot_curr, _ = jnp.meshgrid(x_plot_1d, jnp.arange(nz + 1), indexing='ij')

for idx, (title, pert_th_v) in enumerate(configs_to_test):
    data_2d = pert_th_v[:, 1, :]
    data_padded = jnp.concatenate([data_2d[:, :1], data_2d], axis=1)
    
    ax = axs3[idx]
    contour3 = ax.contourf(X_m_plot_curr, Z_m_plot, data_padded, 
                           levels=jnp.linspace(-vmax_pert_shared, vmax_pert_shared, 41), 
                           cmap='RdBu_r', extend='both')
    
    ax.fill_between(x_plot_1d, 0, mountain_terrain, color='black')
    
    # Draw Target Box at T=0 (for reference, though target metric is at T=t_end)
    rect = patches.Rectangle((rect_x, rect_z), width, height, 
                             linewidth=2, edgecolor='k', facecolor='none', linestyle=':')
    ax.add_patch(rect)
    ax.text(rect_x + width/2, rect_z + height + 0.2, 'Target area', color='k', ha='center', fontsize=9)
    
    ax.set_title(f"{title} (\\delta \\theta_v$ at $T=0$)")
    ax.set_xlim([-50, 50])
    ax.set_ylim([0, 12])
    
    if idx >= 2: ax.set_xlabel('Distance (km)')
    if idx % 2 == 0: ax.set_ylabel('Altitude (km)')

fig3.subplots_adjust(right=0.92)
cbar_ax3 = fig3.add_axes([0.94, 0.15, 0.02, 0.7])
fig3.colorbar(contour3, cax=cbar_ax3, label=r'Temperature perturbation $\delta \theta_v$ (K)')

plt.savefig(f'{output_dir}/schaer_perturbation_validation_{CORE_TYPE.lower()}.png', dpi=150, bbox_inches='tight')
print(f"Perturbation comparison complete. Saved to {output_dir}/schaer_perturbation_validation_{CORE_TYPE.lower()}.png")