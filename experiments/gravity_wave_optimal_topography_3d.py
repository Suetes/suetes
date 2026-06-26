import os
import numpy as np

import jax
jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp
import optax

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.steppers import build_dynamical_core
from suetes.regional3d.boundaries import BenchmarkSponge
from suetes.shared.driver import Simulation
from suetes.shared.optimization import OptaxSolver

output_dir = "output/plots/inversion"
os.makedirs(output_dir, exist_ok=True)

# =====================================================================
# CONFIGURATION SWITCHES
# =====================================================================
CORE_TYPE = "split-explicit"  # Toggle to "sisl" or "split-explicit"

# --- 1. SETUP PARAMETERS ---
nx, ny, nz = 300, 3, 50
dx, dy, dz = 500.0, 500.0, 400.0  
dt = 4.0
t_end = 1800.0  
num_steps = int(t_end / dt)
u_bg = 10.0
constants = {'g': 9.81, 'cp': 1004.0, 'Rd': 287.0, 'cvd': 717.0, 'p0': 100000.0}

num_rbfs = 8
mu_rbf = jnp.linspace(-15000.0, 10000.0, num_rbfs) 
sigma_rbf = 2500.0

x_sponge = BenchmarkSponge(nx=nx, sponge_depth=10, axes=('x',))

# --- 2. THE OBJECTIVE FUNCTION ---
total_dirt_budget = 1500.0  

def objective_fn(z_params):
    weights = jax.nn.sigmoid(z_params)
    A_params = total_dirt_budget * (weights / jnp.sum(weights))

    def h_func(x, y):
        h = jnp.zeros_like(x)
        for i in range(num_rbfs):
            h += A_params[i] * jnp.exp(-((x - mu_rbf[i])**2) / (2 * sigma_rbf**2))
        return jnp.maximum(h, 0.0)

    grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_center=45.0, lon_center=0.0, h_func=h_func)
    op = CGridOperator3D(grid)
    
    # Temporarily instantiate physics to establish the background reference state
    tmp_phys = Euler3D(grid, op, constants, dt=dt, N_bv=0.01)
    bg_ref = {
        'rho': tmp_phys.c['p0'] / (tmp_phys.c['Rd'] * tmp_phys.theta_bg) * \
               (tmp_phys.pi_bg ** (tmp_phys.c['cvd'] / tmp_phys.c['Rd'])),
        'pi': tmp_phys.pi_bg,
        'th_v': tmp_phys.theta_bg
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
        initial_state=state, **core_kwargs
    )
    
    # Inject checkpointing for reverse-mode autodiff memory savings
    if CORE_TYPE.lower() == "sisl":
        stepper.use_checkpointing = True

    def bc_fn(state_in, forcing=None):
        ext_state = {
            'u': jnp.ones_like(state_in['u']) * u_bg,
            'v': jnp.zeros_like(state_in['v']),
            'th_v': bg_ref['th_v'],
            'rho': bg_ref['rho'],
            'pi': bg_ref['pi']
        }
        return x_sponge.blend(state_in, ext_state)

    sim = Simulation(step_fn=stepper.step, dt=dt)
    
    final_state = sim.run_differentiable(state, 0.0, t_end, bc_fn=bc_fn, chunk_steps=50)
    
    w_target = final_state['w'][180:200, 1, 8:20] 
    J_energy = jnp.sum(w_target ** 2)
    
    return -J_energy, (final_state['w'], A_params)

# --- 3. THE OPTIMIZATION LOOP ---
key = jax.random.PRNGKey(42)
z_params = jax.random.normal(key, (num_rbfs,)) * 0.1
total_steps = 20

lr_schedule = optax.cosine_decay_schedule(init_value=0.5, decay_steps=total_steps, alpha=0.01)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0), 
    optax.scale_by_adam(),           
    optax.scale_by_schedule(lr_schedule), 
    optax.scale(-1.0)                     
)

print(f"\n[OPTIMIZATION] Launching {CORE_TYPE.upper()} inverse topography optimization...")
solver = OptaxSolver(objective_fn, optimizer, has_aux=True)
optimal_z, history = solver.fit(
    z_params, 
    total_steps=total_steps, 
    patience=2, 
    metric_name="Energy", 
    maximize=True
)

# Retrieve the best state from history since energy maximization can oscillate
energies = [-loss for loss in history['loss']]
best_idx = np.argmax(energies)
best_energy = energies[best_idx]
best_A = history['aux'][best_idx][1]
history_A = [aux[1] for aux in history['aux']]

print(f"\nOptimization complete. Best target energy: {best_energy:.2f} at step {best_idx+1}")

# --- 4. VISUALIZE THE GENERATED MOUNTAIN ---
print("\n[PLOT] Saving optimal design...")
x_1d = jnp.linspace(-nx*dx/2, nx*dx/2, nx) / 1000.0

plt.figure(figsize=(10, 5))
plt.title(f"Evolution of optimal topography ({CORE_TYPE.capitalize()})")

# Plot history in fading blue
for i, A_step in enumerate(history_A):
    h = jnp.zeros_like(x_1d)
    for j in range(num_rbfs):
        h += A_step[j] * jnp.exp(-((x_1d*1000.0 - mu_rbf[j])**2) / (2 * sigma_rbf**2))
    h = jnp.maximum(h, 0.0)
    alpha = (i + 1) / len(history_A)
    plt.plot(x_1d, h, color='blue', alpha=alpha*0.5)

# Plot the best mountain in bold
h_best = jnp.zeros_like(x_1d)
for j in range(num_rbfs):
    h_best += best_A[j] * jnp.exp(-((x_1d*1000.0 - mu_rbf[j])**2) / (2 * sigma_rbf**2))
h_best = jnp.maximum(h_best, 0.0)
plt.plot(x_1d, h_best, color='red', linewidth=2, label=f'Best (E={best_energy:.1f}J)')

plt.xlabel("Distance (km)")
plt.ylabel("Elevation (m)")
plt.xlim([-25, 25])
plt.legend()
plt.savefig(f"{output_dir}/inverse_topography_evolution_{CORE_TYPE.lower()}.png", dpi=150)
print(f"Done! Check 'inverse_topography_evolution_{CORE_TYPE.lower()}.png'.")

# --- 5. VALIDATION: OPTIMAL VS. RANDOM MOUNTAIN ALLOCATIONS ---
print("\n[VALIDATION] Running forward simulations for comparison...")

# 1. Gather the configurations to test
configs_to_test = []

# A. The Optimal Configuration
optimal_A = best_A 
configs_to_test.append(("Optimal topography", optimal_A))

# B. Generate 3 Random Configurations (enforcing the 1500m budget)
key = jax.random.PRNGKey(42)
for i in range(1, 4):
    key, subkey = jax.random.split(key)
    rand_z = jax.random.normal(subkey, (num_rbfs,))
    rand_A = total_dirt_budget * jax.nn.softmax(rand_z)
    configs_to_test.append((f"Random topography {i}", rand_A))

# 2. Evaluation Wrapper 
def evaluate_topography(A_params):
    def h_func(x, y):
        h = jnp.zeros_like(x)
        for i in range(num_rbfs):
            h += A_params[i] * jnp.exp(-((x - mu_rbf[i])**2) / (2 * sigma_rbf**2))
        return jnp.maximum(h, 0.0)

    grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_center=45.0, lon_center=0.0, h_func=h_func)
    op = CGridOperator3D(grid)
    
    tmp_phys = Euler3D(grid, op, constants, dt=dt, N_bv=0.01)
    bg_ref = {
        'rho': tmp_phys.c['p0'] / (tmp_phys.c['Rd'] * tmp_phys.theta_bg) * \
               (tmp_phys.pi_bg ** (tmp_phys.c['cvd'] / tmp_phys.c['Rd'])),
        'pi': tmp_phys.pi_bg,
        'th_v': tmp_phys.theta_bg
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
        initial_state=state, **core_kwargs
    )
    
    if CORE_TYPE.lower() == "sisl":
        stepper.use_checkpointing = False

    def bc_fn(state_in, forcing=None):
        ext_state = {
            'u': jnp.ones_like(state_in['u']) * u_bg,
            'v': jnp.zeros_like(state_in['v']),
            'th_v': bg_ref['th_v'],
            'rho': bg_ref['rho'],
            'pi': bg_ref['pi']
        }
        return x_sponge.blend(state_in, ext_state)

    @jax.jit
    def fast_forward(s):
        def scan_fn(state, _):
            return stepper.step(state, 0.0, None, bc_fn), None
        return jax.lax.scan(scan_fn, s, jnp.arange(num_steps))[0]

    final_state = fast_forward(state)
    
    w_target = final_state['w'][180:200, 1, 8:20] 
    J_energy = jnp.sum(w_target ** 2)
    
    return float(J_energy), final_state['w'], grid

# 3. Run and Plot
fig, axs = plt.subplots(2, 2, figsize=(20, 12))
axs = axs.flatten()

x_plot_1d = jnp.linspace(-nx*dx/2, nx*dx/2, nx) / 1000.0

for idx, (title, A_params) in enumerate(configs_to_test):
    print(f"Evaluating {title}...")
    J_val, w_field, eval_grid = evaluate_topography(A_params)
    
    # Reconstruct the terrain for plotting
    h_terrain = jnp.zeros_like(x_plot_1d)
    for j in range(num_rbfs):
        h_terrain += A_params[j] * jnp.exp(-((x_plot_1d*1000.0 - mu_rbf[j])**2) / (2 * sigma_rbf**2))
    h_terrain = jnp.maximum(h_terrain, 0.0)

    # Plotting logic
    ax = axs[idx]
    w_slice = w_field[:, 1, :]
    
    # Pad for Mass/W grid differences
    data_to_plot = w_slice
    Z_plot_curr = eval_grid.Z_w[:, 1, :] / 1000.0  
    X_plot_curr, _ = jnp.meshgrid(x_plot_1d, jnp.arange(nz + 1), indexing='ij')

    vmax = 1.5 # Fixed color scale for fair visual comparison
    contour = ax.contourf(X_plot_curr, Z_plot_curr, data_to_plot, levels=jnp.linspace(-vmax, vmax, 41), cmap='RdBu_r', extend='both')
    
    ax.fill_between(x_plot_1d, 0, h_terrain / 1000.0, color='black')
    
    # Draw Target Box
    rect_x = x_plot_1d[180]
    rect_z = Z_plot_curr[180, 8]
    width = x_plot_1d[200] - x_plot_1d[180]
    height = Z_plot_curr[180, 20] - Z_plot_curr[180, 8]
    rect = patches.Rectangle((rect_x, rect_z), width, height, linewidth=2, edgecolor='k', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    
    ax.set_title(f"{title}\nTarget energy (J): {J_val:.2f}", fontweight='bold')
    ax.set_xlim([-30, 40])
    ax.set_ylim([0, 12])
    if idx >= 2: ax.set_xlabel('Distance (km)')
    if idx % 2 == 0: ax.set_ylabel('Altitude (km)')

plt.tight_layout()
plt.savefig(f'{output_dir}/inverse_topography_validation_{CORE_TYPE.lower()}.png', dpi=150, bbox_inches='tight')
print(f"Validation complete. Saved to {output_dir}/inverse_topography_validation_{CORE_TYPE.lower()}.png")