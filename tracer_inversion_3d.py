import os
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.steppers import SISLStepper3D
from suetes.regional3d.physics import PhysicsSuite

output_dir = "suetes/plots/tracer_inversion"
os.makedirs(output_dir, exist_ok=True)

# --- 1. SETUP DOMAIN & PHYSICS ---
nx, ny, nz = 200, 3, 40  
dx, dy, dz = 400.0, 400.0, 250.0  
dt = 5.0
t_end = 2400.0  # 40 mins of advection
num_steps = int(t_end / dt)
u_bg = 10.0

constants = {'g': 9.81, 'cp': 1004.0, 'Rd': 287.0, 'cvd': 717.0, 'p0': 100000.0}

# Introduce irregular topography: A 2km high Gaussian mountain
def terrain_profile(x, y):
    h0 = 2000.0    # 2 km height
    a = 4000.0     # 4 km width spread
    center = -5000.0 # Positioned at X = -5 km
    return h0 * jnp.exp(-((x - center) / a)**2)

grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_center=45.0, lon_center=0.0, h_func=terrain_profile)
op = CGridOperator3D(grid)

suite = PhysicsSuite()
suite.register_tracer('q_tr')

physics = Euler3D(grid, op, constants, dt=dt, N_bv=0.01, damp_height=8000.0, 
                  max_damp=0.5, nu_div_factor=0.0, nu_h_factor=0.0, physics_suite=suite)

# --- 2. STATE GENERATOR & BOUNDARY CONDITIONS ---
bg_ref = {
    'rho': physics.c['p0'] / (physics.c['Rd'] * physics.theta_bg) * \
           (physics.pi_bg ** (physics.c['cvd'] / physics.c['Rd'])),
    'pi': physics.pi_bg,
    'th_v': physics.theta_bg
}

sponge_depth = 10
x_idx = jnp.arange(nx, dtype=jnp.float32)
dist_x = jnp.minimum(x_idx, nx - x_idx)
mask_x = jnp.where(dist_x < sponge_depth, jnp.cos(0.5 * jnp.pi * dist_x / sponge_depth)**2, 0.0)[:, None, None]

def bc_fn(state_in, forcing):
    ext_state = {
        'u': jnp.ones_like(state_in['u']) * u_bg,
        'v': jnp.zeros_like(state_in['v']),
        'th_v': bg_ref['th_v'],
        'rho': bg_ref['rho'],
        'pi': bg_ref['pi'],
        'q_tr': jnp.zeros_like(state_in.get('q_tr', jnp.zeros_like(state_in['rho'])))
    }
    blended = {}
    blend_vars = ['u', 'v', 'th_v', 'pi', 'rho', 'q_tr'] 
    for k in state_in.keys():
        if k in blend_vars and k in ext_state:
            m = jnp.pad(mask_x, ((0, 1), (0, 0), (0, 0)), mode='edge') if k == 'u' else mask_x
            blended[k] = (1.0 - m) * state_in[k] + m * ext_state[k]
        else:
            blended[k] = state_in[k]
    return blended

def create_state_with_tracer(x_km, z_km, amplitude):
    X_km = grid.x_m[:, None, None] / 1000.0
    Z_km = grid.Z_m / 1000.0
    
    sigma_x, sigma_z = 2.0, 1.0
    q_tr = amplitude * jnp.exp(-((X_km - x_km)**2 / (2 * sigma_x**2) + (Z_km - z_km)**2 / (2 * sigma_z**2)))
    
    return {
        'u': jnp.ones((nx+1, ny, nz)) * u_bg,
        'v': jnp.zeros((nx, ny+1, nz)),
        'w': jnp.zeros((nx, ny, nz+1)),
        'pi': bg_ref['pi'],
        'eta_dot': jnp.zeros((nx, ny, nz+1)),
        'rho': bg_ref['rho'],
        'th_v': bg_ref['th_v'],
        'q_tr': q_tr
    }

# --- 3. GENERATE "TRUE" TARGET DATA AND SNAPSHOTS ---
print("[SIMULATION] Generating true target observations over topography...")
stepper_fwd = SISLStepper3D(physics, dt, use_checkpointing=False)

true_params = {'x': -15.0, 'z': 3.5, 'A': 10.0}
true_state = create_state_with_tracer(true_params['x'], true_params['z'], true_params['A'])

# Modified fast_forward to return the tracer slice at each step for the snapshot plot
def fast_forward(s, _):
    next_s = stepper_fwd.step(s, 0.0, None, bc_fn)
    return next_s, next_s['q_tr'][:, 1, :]

true_final_state, q_tr_history = jax.lax.scan(fast_forward, true_state, jnp.arange(num_steps))

# Sensor Array at x = +10 km
sensor_idx_x = int((10.0 - (grid.x_m[0]/1000.0)) / (dx/1000.0))
target_sensor_profile = true_final_state['q_tr'][sensor_idx_x, 1, :]

# Extract snapshots for visualization
snapshot_indices = [num_steps // 4, num_steps // 2, 3 * num_steps // 4, num_steps - 1]
snapshots = [true_state['q_tr'][:, 1, :]] + [q_tr_history[i] for i in snapshot_indices]
times_mins = [0.0] + [(i * dt) / 60.0 for i in snapshot_indices]

# --- 4. THE INVERSE PROBLEM ---
def objective_fn(params):
    x_km, z_km, A = params[0], params[1], params[2]
    
    state = create_state_with_tracer(x_km, z_km, A)
    stepper_adj = SISLStepper3D(physics, dt, use_checkpointing=True)
    
    chunk_size = 40 
    num_chunks = num_steps // chunk_size
    
    @jax.checkpoint
    def scan_chunk(curr_state, chunk_idx):
        def inner_scan_fn(s, step_offset):
            t_curr = (chunk_idx * chunk_size + step_offset) * dt
            return stepper_adj.step(s, t_curr, forcing=None, bc_fn=bc_fn), None
        chunk_final, _ = jax.lax.scan(inner_scan_fn, curr_state, jnp.arange(chunk_size))
        return chunk_final, None

    final_state, _ = jax.lax.scan(scan_chunk, state, jnp.arange(num_chunks))
    
    # Pure MSE Loss (removed the regularization penalty)
    simulated_sensor_profile = final_state['q_tr'][sensor_idx_x, 1, :]
    mse_loss = jnp.mean((simulated_sensor_profile - target_sensor_profile)**2)
    
    return mse_loss, final_state

print("\n[JAX] Compiling Inverse Advection Model...")
grad_fn = jax.jit(jax.value_and_grad(objective_fn, has_aux=True))

# Start with a terrible guess
guess_params = jnp.array([-5.0, 7.0, 2.0]) 

# Use the Cosine Decay Schedule to prevent overshooting the target
total_opt_steps = 60
lr_schedule = optax.cosine_decay_schedule(init_value=0.5, decay_steps=total_opt_steps, alpha=0.02)

optimizer = optax.adam(learning_rate=lr_schedule)
opt_state = optimizer.init(guess_params)

history_params = [guess_params]

print("\n[OPTIMIZATION] Recovering Tracer Source Coordinates...")
for i in range(total_opt_steps): 
    start = time.time()
    
    (loss, final_sim_state), grads = grad_fn(guess_params)
    updates, opt_state = optimizer.update(grads, opt_state, guess_params)
    guess_params = optax.apply_updates(guess_params, updates)
    
    # Bounds protection
    guess_params = guess_params.at[1].set(jnp.maximum(guess_params[1], 0.1))
    guess_params = guess_params.at[2].set(jnp.maximum(guess_params[2], 0.0))
    
    history_params.append(guess_params)
    current_lr = lr_schedule(i)
    
    print(f"Step {i+1:02d} | Loss: {loss:.4f} | Guess X: {guess_params[0]:.2f}km, Z: {guess_params[1]:.2f}km, A: {guess_params[2]:.2f} | LR: {current_lr:.3f} | Time: {time.time()-start:.1f}s")

# --- 5. VISUALIZATION ---
print("\n[PLOT] Generating visualizations...")

x_plot_1d = grid.x_m / 1000.0
Z_plot = grid.Z_m[:, 1, :] / 1000.0
X_plot, _ = jnp.meshgrid(x_plot_1d, jnp.arange(nz), indexing='ij')
terrain_plot = terrain_profile(grid.x_m, 0) / 1000.0

# 5a. Plot Forward Snapshots
fig1, axs1 = plt.subplots(5, 1, figsize=(12, 18), sharex=True)
fig1.suptitle("Tracer plume evolution over topography", fontsize=16)

for idx, ax in enumerate(axs1):
    levels = jnp.linspace(0.01, 10.0, 100)
    contour = ax.contourf(X_plot, Z_plot, snapshots[idx], levels=levels, cmap='Blues', extend='max')
    ax.set_facecolor('white')
    ax.fill_between(x_plot_1d, 0, terrain_plot, color='black')
    ax.axvline(x=10.0, color='blue', linestyle='--', linewidth=1.5)
    ax.set_title(f"T = {times_mins[idx]:.1f} mins")
    ax.set_ylabel("Altitude (km)")
    ax.set_ylim([0, 10])

axs1[-1].set_xlabel("Distance (km)")
plt.tight_layout()
plt.savefig(f"{output_dir}/tracer_snapshots.png", dpi=150)

# 5b. Plot Optimization Trajectory
fig2, axs2 = plt.subplots(2, 1, figsize=(12, 10))

# Trajectory Plot
axs2[0].contourf(X_plot, Z_plot, true_state['q_tr'][:, 1, :], levels=20, cmap='Greys', alpha=0.3)
axs2[0].fill_between(x_plot_1d, 0, terrain_plot, color='black', alpha=0.7)
axs2[0].axvline(x=10.0, color='blue', linestyle='--', linewidth=2, label='Sensor array')

hx = [p[0] for p in history_params]
hz = [p[1] for p in history_params]
axs2[0].plot(hx, hz, marker='o', color='purple', linestyle='-', linewidth=2, markersize=5, label='Optimizer trajectory')
axs2[0].scatter([hx[0]], [hz[0]], color='orange', s=100, label='Initial guess')
axs2[0].scatter([hx[-1]], [hz[-1]], color='blue', s=100, zorder=5, label='Inferred source location')
axs2[0].scatter([true_params['x']], [true_params['z']], color='red', marker='x', s=100, zorder=6, label='True source location')

axs2[0].set_title("Source location optimization")
axs2[0].set_xlim([-30, 30])
axs2[0].set_ylim([0, 10])
axs2[0].set_ylabel("Altitude (km)")
axs2[0].legend()

# Sensor Profile Plot
final_guessed_state = create_state_with_tracer(guess_params[0], guess_params[1], guess_params[2])
final_sim, _ = jax.lax.scan(fast_forward, final_guessed_state, jnp.arange(num_steps))
recovered_sensor_profile = final_sim['q_tr'][sensor_idx_x, 1, :]

axs2[1].plot(target_sensor_profile, grid.Z_m[sensor_idx_x, 1, :]/1000.0, 'k-', linewidth=3, label='True target profile')
axs2[1].plot(recovered_sensor_profile, grid.Z_m[sensor_idx_x, 1, :]/1000.0, 'r--', linewidth=2, label='Recovered profile')
axs2[1].set_title("Vertical tracer concentration at sensor array (T=40 mins)")
axs2[1].set_xlabel("Tracer concentration")
axs2[1].set_ylabel("Altitude (km)")
axs2[1].legend()

plt.tight_layout()
plt.savefig(f"{output_dir}/tracer_source_inversion_terrain.png", dpi=150)
print(f"Done! Check '{output_dir}/tracer_snapshots.png' and '{output_dir}/tracer_source_inversion_terrain.png'")