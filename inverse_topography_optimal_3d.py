import os
import time
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import optax

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.steppers import SISLStepper3D

output_dir = "suetes/plots/inversion"
os.makedirs(output_dir, exist_ok=True)

# --- 1. SETUP PARAMETERS ---
nx, ny, nz = 300, 3, 50
dx, dy, dz = 500.0, 500.0, 400.0  
dt = 4.0
t_end = 1800.0  # 30 mins (gives waves time to reach the target box)
num_steps = int(t_end / dt)
u_bg = 10.0
constants = {'g': 9.81, 'cp': 1004.0, 'Rd': 287.0, 'cvd': 717.0, 'p0': 100000.0}

# Define the Radial Basis Functions for our trainable mountain
num_rbfs = 8
mu_rbf = jnp.linspace(-15000.0, 10000.0, num_rbfs) # Centers from -15km to +10km
sigma_rbf = 2500.0

# Global Sponge Mask
sponge_depth = 10
x_idx = jnp.arange(nx, dtype=jnp.float32)
dist_x = jnp.minimum(x_idx, nx - x_idx)
weight_x = jnp.where(dist_x < sponge_depth, jnp.cos(0.5 * jnp.pi * dist_x / sponge_depth)**2, 0.0)
mask_x = weight_x[:, None, None]

# --- 2. THE OBJECTIVE FUNCTION ---
total_dirt_budget = 1500.0  # Total allocated amplitude in meters

# We now pass unconstrained logits (z_params) into the objective function
def objective_fn(z_params):
    
    # 1. Sigmoid-Normalize: Enforces budget without the aggressive sparsity of Softmax
    weights = jax.nn.sigmoid(z_params)
    A_params = total_dirt_budget * (weights / jnp.sum(weights))

    # 2. Construct the Differentiable Topography 
    def h_func(x, y):
        h = jnp.zeros_like(x)
        for i in range(num_rbfs):
            h += A_params[i] * jnp.exp(-((x - mu_rbf[i])**2) / (2 * sigma_rbf**2))
        return jnp.maximum(h, 0.0)

    grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_center=45.0, lon_center=0.0, h_func=h_func)
    op = CGridOperator3D(grid)
    physics = Euler3D(grid, op, constants, dt=dt, N_bv=0.01, damp_height=12000.0, 
                      max_damp=0.5, nu_div_factor=0.0, nu_h_factor=0.0, physics_suite=None)

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

    stepper = SISLStepper3D(physics, dt, use_checkpointing=True)

    def bc_fn(state_in, forcing):
        ext_state = {
            'u': jnp.ones_like(state_in['u']) * u_bg,
            'v': jnp.zeros_like(state_in['v']),
            'th_v': bg_ref['th_v'],
            'rho': bg_ref['rho'],
            'pi': bg_ref['pi']
        }
        blended = {}
        blend_vars = ['u', 'v', 'th_v', 'pi', 'rho'] 
        for k in state_in.keys():
            if k in blend_vars and k in ext_state:
                m = jnp.pad(mask_x, ((0, 1), (0, 0), (0, 0)), mode='edge') if k == 'u' else mask_x
                blended[k] = (1.0 - m) * state_in[k] + m * ext_state[k]
            else:
                blended[k] = state_in[k]
        return blended

    # 3. Time Integration Loop
    chunk_size = 50 
    num_chunks = num_steps // chunk_size
    
    @jax.checkpoint
    def scan_chunk(curr_state, chunk_idx):
        def inner_scan_fn(s, step_offset):
            t_curr = (chunk_idx * chunk_size + step_offset) * dt
            return stepper.step(s, t_curr, forcing=None, bc_fn=bc_fn), None
        chunk_final_state, _ = jax.lax.scan(inner_scan_fn, curr_state, jnp.arange(chunk_size))
        return chunk_final_state, None

    final_state, _ = jax.lax.scan(scan_chunk, state, jnp.arange(num_chunks))
    
    # 4. Target Wave Evaluation
    # Use the same 2D target box you use in your validation script
    w_target = final_state['w'][180:200, 1, 8:20] 
    
    # Calculate Total Kinetic Energy in the target box
    J_energy = jnp.sum(w_target ** 2)
    
    # Return NEGATIVE energy so we can minimize it using standard descent
    return -J_energy, (final_state['w'], A_params)

# --- 3. THE OPTIMIZATION LOOP (ENERGY MAXIMIZATION WITH OPTAX) ---
print("\n[JAX] Compiling Inverse Design Model... (This will take a minute)")
grad_fn = jax.jit(jax.value_and_grad(objective_fn, has_aux=True))

key = jax.random.PRNGKey(42)
z_params = jax.random.normal(key, (num_rbfs,)) * 0.1

total_steps = 30

# 1. Cosine Decay Schedule: Starts at 0.5, decays down to 1% of that (0.005) by step 30
lr_schedule = optax.cosine_decay_schedule(init_value=0.5, decay_steps=total_steps, alpha=0.01)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0), 
    optax.scale_by_adam(),           
    optax.scale_by_schedule(lr_schedule), # Apply the decay schedule
    optax.scale(-1.0)                     # Negative scale for descent
)
opt_state = optimizer.init(z_params)

history_A = []
history_Energy = []

# Tracker for the best model
best_energy = -np.inf
best_A = None

print("\n[OPTIMIZATION] Starting Downstream Energy Maximization with Optax...")
for i in range(total_steps): 
    start = time.time()
    
    (neg_J_energy, (w_final, A_actual)), grads = grad_fn(z_params)
    energy_val = -float(neg_J_energy)
    
    # Track the best configuration
    if energy_val > best_energy:
        best_energy = energy_val
        best_A = A_actual
    
    updates, opt_state = optimizer.update(grads, opt_state, z_params)
    z_params = optax.apply_updates(z_params, updates)
    
    history_A.append(A_actual)
    history_Energy.append(energy_val)
    
    # Early Stopping Logic
    if energy_val < best_energy:
        patience_counter += 1
    else:
        patience_counter = 0 # Reset if we find a new best
        
    if patience_counter >= 3: # If it drops for 3 steps in a row, bail out.
        print(f"\n[EARLY STOPPING] Optimizer overshot at step {i+1}. Halting to save compute.")
        break
    
    # Optional: fetch current learning rate for logging
    current_lr = lr_schedule(i)
    
    print(f"Step {i+1:02d} | Energy: {energy_val:.2f} | Best: {best_energy:.2f} | LR: {current_lr:.3f} | Max Peak: {float(jnp.max(A_actual)):.1f}m | Time: {time.time()-start:.1f}s")

print(f"\nOptimization Complete. Best Target Energy Achieved: {best_energy:.2f}")


# --- 4. VISUALIZE THE GENERATED MOUNTAIN ---
print("\n[PLOT] Saving optimal design...")
x_1d = jnp.linspace(-nx*dx/2, nx*dx/2, nx) / 1000.0

plt.figure(figsize=(10, 5))
plt.title("Evolution of Optimal Topography")

# Plot history in fading blue
for i, A_step in enumerate(history_A):
    h = jnp.zeros_like(x_1d)
    for j in range(num_rbfs):
        h += A_step[j] * jnp.exp(-((x_1d*1000.0 - mu_rbf[j])**2) / (2 * sigma_rbf**2))
    h = jnp.maximum(h, 0.0)
    alpha = (i + 1) / len(history_A)
    plt.plot(x_1d, h, color='blue', alpha=alpha*0.5)

# Plot the definitive BEST mountain in bold
h_best = jnp.zeros_like(x_1d)
for j in range(num_rbfs):
    h_best += best_A[j] * jnp.exp(-((x_1d*1000.0 - mu_rbf[j])**2) / (2 * sigma_rbf**2))
h_best = jnp.maximum(h_best, 0.0)
plt.plot(x_1d, h_best, color='red', linewidth=2, label=f'Best (E={best_energy:.1f}J)')

plt.xlabel("Distance (km)")
plt.ylabel("Elevation (m)")
plt.xlim([-25, 25])
plt.legend()
plt.savefig(f"{output_dir}/inverse_topography_evolution.png", dpi=150)
print("Done! Check 'inverse_topography_evolution.png'.")

# --- 5. VALIDATION: OPTIMAL VS. RANDOM DIRT ALLOCATIONS ---
print("\n[VALIDATION] Running forward simulations for comparison...")

# 1. Gather the configurations to test
configs_to_test = []

# A. The Optimal Configuration
optimal_A = best_A 
configs_to_test.append(("Optimal Topography", optimal_A))

# B. Generate 3 Random Configurations (strictly enforcing the 1500m budget)
key = jax.random.PRNGKey(42)
for i in range(1, 4):
    key, subkey = jax.random.split(key)
    rand_z = jax.random.normal(subkey, (num_rbfs,))
    rand_A = total_dirt_budget * jax.nn.softmax(rand_z)
    configs_to_test.append((f"Random Topography {i}", rand_A))

# 2. Evaluation Wrapper (Eager execution, no JAX tracing needed here)
def evaluate_topography(A_params):
    def h_func(x, y):
        h = jnp.zeros_like(x)
        for i in range(num_rbfs):
            h += A_params[i] * jnp.exp(-((x - mu_rbf[i])**2) / (2 * sigma_rbf**2))
        return jnp.maximum(h, 0.0)

    grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_center=45.0, lon_center=0.0, h_func=h_func)
    op = CGridOperator3D(grid)
    physics = Euler3D(grid, op, constants, dt=dt, N_bv=0.01, damp_height=12000.0, 
                      max_damp=0.5, nu_div_factor=0.0, nu_h_factor=0.0, physics_suite=None)

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

    stepper = SISLStepper3D(physics, dt, use_checkpointing=False) # Fast forward mode

    def bc_fn(state_in, forcing):
        ext_state = {
            'u': jnp.ones_like(state_in['u']) * u_bg,
            'v': jnp.zeros_like(state_in['v']),
            'th_v': bg_ref['th_v'],
            'rho': bg_ref['rho'],
            'pi': bg_ref['pi']
        }
        blended = {}
        blend_vars = ['u', 'v', 'th_v', 'pi', 'rho'] 
        for k in state_in.keys():
            if k in blend_vars and k in ext_state:
                m = jnp.pad(mask_x, ((0, 1), (0, 0), (0, 0)), mode='edge') if k == 'u' else mask_x
                blended[k] = (1.0 - m) * state_in[k] + m * ext_state[k]
            else:
                blended[k] = state_in[k]
        return blended

    def fast_forward(s, _):
        return stepper.step(s, 0.0, None, bc_fn), None

    final_state, _ = jax.lax.scan(fast_forward, state, jnp.arange(num_steps))
    
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
    
    ax.set_title(f"{title}\nTarget Energy (J): {J_val:.2f}", fontweight='bold')
    ax.set_xlim([-30, 40])
    ax.set_ylim([0, 12])
    if idx >= 2: ax.set_xlabel('Distance (km)')
    if idx % 2 == 0: ax.set_ylabel('Altitude (km)')

plt.tight_layout()
plt.savefig(f'{output_dir}/inverse_topography_validation.png', dpi=150, bbox_inches='tight')
print(f"Validation complete. Saved to {output_dir}/inverse_topography_validation.png")