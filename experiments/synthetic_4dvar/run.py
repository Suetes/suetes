# Case runner.
import argparse
from pathlib import Path
import time
import numpy as np
from scipy.ndimage import gaussian_filter

import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import xarray as xr

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.steppers import build_dynamical_core
from suetes.shared.optimization import OptaxSolver
from suetes.shared.experiment import save_plot_dataset, add_experiment_args, setup_experiment_directories

parser = argparse.ArgumentParser(description="Run the synthetic 4D-Var experiment")
add_experiment_args(parser)
parser.add_argument("--no-render", action="store_true")
args = parser.parse_args()
data_dir, figure_dir = setup_experiment_directories(
    args, kind="experiments", case="synthetic_4dvar"
)

# =====================================================================
# CONFIGURATION
# =====================================================================
CORE_TYPE = "split-explicit"
OBSERVATION_MODE = "full_3d"  # Options: "full_3d", "surface_only", "sparse_mix"

nx, ny, nz = 100, 3, 20
dx, dy, dz = 2000.0, 2000.0, 400.0
dt = 10.0
assimilation_window = 10800.0  # 3 hours
obs_interval = 1800.0  # Take an observation every 30 mins
num_steps = int(assimilation_window / dt)
steps_per_obs = int(obs_interval / dt)

constants = {"g": 9.81, "cp": 1004.0, "Rd": 287.0, "cvd": 717.0, "p0": 100000.0}


# --- 1. SETUP THE GRID & BASE STATE ---
def h_func(x, y):
    return jnp.zeros_like(x)  # Flat terrain for clean test


grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_center=45.0, lon_center=0.0, h_func=h_func)
op = CGridOperator3D(grid)

tmp_phys = Euler3D(grid, op, constants, dt=dt, N_bv=0.01)
bg_ref = {
    "rho": tmp_phys.c["p0"]
    / (tmp_phys.c["Rd"] * tmp_phys.theta_bg)
    * (tmp_phys.pi_bg ** (tmp_phys.c["cvd"] / tmp_phys.c["Rd"])),
    "pi": tmp_phys.pi_bg,
    "th_v": tmp_phys.theta_bg,
}

# --- 2. SENSOR CONFIGURATION MASKS ---
# Create mask base: shape (nx, ny, nz)
surface_mask = jnp.zeros((nx, ny, nz)).at[:, :, 0].set(1.0)

# Radiosondes: Stationed at x-grid index 30 and 70, sampling all vertical levels
sonde_mask = jnp.zeros((nx, ny, nz))
sonde_mask = sonde_mask.at[30, :, :].set(1.0)
sonde_mask = sonde_mask.at[50, :, :].set(1.0)
sonde_mask = sonde_mask.at[70, :, :].set(1.0)

# Commercial Aircraft: Slicing diagonally across the domain at cruise altitudes
aircraft_mask = jnp.zeros((nx, ny, nz))
for z_level in range(5, 15):
    x_idx = int(20 + z_level * 4)
    if x_idx < nx:
        aircraft_mask = aircraft_mask.at[x_idx, :, z_level].set(1.0)

# Combine masks based on the chosen mode
if OBSERVATION_MODE == "sparse_mix":
    H_mask = jnp.clip(surface_mask + sonde_mask + aircraft_mask, 0.0, 1.0)
elif OBSERVATION_MODE == "surface_only":
    H_mask = surface_mask
elif OBSERVATION_MODE == "full_3d":
    H_mask = jnp.ones((nx, ny, nz))

# --- 3. CREATE THE "TRUE" INITIAL STATE ---
x_1d = jnp.linspace(-nx * dx / 2, nx * dx / 2, nx)
X_3d, Y_3d, Z_3d = jnp.meshgrid(x_1d, jnp.arange(ny), jnp.arange(nz), indexing="ij")

# Inject a cold anomaly (-5K) in the center to drive dynamics
true_th_v = bg_ref["th_v"] - 5.0 * jnp.exp(
    -((X_3d - 0.0) ** 2 / (2 * 10000.0**2)) - ((Z_3d - 2000.0) ** 2 / (2 * 2000.0**2))
)

true_initial_state = {
    "u": jnp.ones((nx + 1, ny, nz)) * 5.0,  # 5 m/s background wind
    "v": jnp.zeros((nx, ny + 1, nz)),
    "w": jnp.zeros((nx, ny, nz + 1)),
    "pi": bg_ref["pi"],
    "eta_dot": jnp.zeros((nx, ny, nz + 1)),
    "rho": bg_ref["rho"],
    "th_v": true_th_v,
}

# --- 4. CREATE THE "SABOTAGED" FIRST GUESS ---
# We blur the true initial temperature heavily using standard SciPy
th_v_np = np.array(true_initial_state["th_v"])
blurred_th_v_np = gaussian_filter(th_v_np, sigma=(10.0, 0.0, 3.0))

# Convert back to a JAX array for the autodiff loop
blurred_th_v = jnp.array(blurred_th_v_np)

# --- 5. GENERATE SYNTHETIC OBSERVATIONS (The Forward Truth Run) ---
print(f"\n[TRUTH RUN] Generating synthetic observations over {assimilation_window / 60:.0f} mins...")
stepper, _ = build_dynamical_core(
    core_type=CORE_TYPE, grid=grid, operators=op, constants=constants, initial_state=true_initial_state, dt=dt, ns=4
)


def dummy_bc(state_in, forcing=None):
    return state_in


@jax.jit
def run_forward(start_state):
    def scan_fn(state, step_idx):
        next_state = stepper.step(state, step_idx * dt, None, dummy_bc)
        return next_state, next_state["th_v"]

    final_state, trajectory = jax.lax.scan(scan_fn, start_state, jnp.arange(num_steps))
    return final_state, trajectory


_, true_trajectory_full = run_forward(true_initial_state)
# Use static slicing to extract observations cleanly for JAX
true_obs = true_trajectory_full[::steps_per_obs]
print(f"[TRUTH RUN] Saved {len(true_obs)} observation snapshots.")


# --- 6. BACKGROUND ERROR COVARIANCE (B-MATRIX) OPERATORS ---
def string_gaussian_kernel(sigma, radius):
    """Generates a 1D Gaussian kernel."""
    x = jnp.arange(-radius, radius + 1)
    kernel = jnp.exp(-0.5 * (x / sigma) ** 2)
    return kernel / jnp.sum(kernel)


def apply_b_half(chi, sigma_x=5.0, sigma_z=2.5):
    """Applies an anisotropic B^{1/2} filter to spread information spatially."""
    radius_x = int(3 * sigma_x)
    radius_z = int(3 * sigma_z)

    kernel_x = string_gaussian_kernel(sigma_x, radius_x)
    kernel_z = string_gaussian_kernel(sigma_z, radius_z)

    def smooth_1d(field_1d, kernel):
        res = jnp.convolve(field_1d, kernel, mode="same")
        # Robustly handle cases where the kernel is larger than the domain
        diff = res.shape[0] - field_1d.shape[0]
        start = diff // 2
        return res[start : start + field_1d.shape[0]]

    def smooth_x(field_1d):
        return smooth_1d(field_1d, kernel_x)

    def smooth_z(field_1d):
        return smooth_1d(field_1d, kernel_z)

    # Smooth along X-axis
    chi_T_x = jnp.transpose(chi, (1, 2, 0))  # Swap X to the end: (ny, nz, nx)
    smoothed_x = jax.vmap(jax.vmap(smooth_x))(chi_T_x)
    chi_smoothed_x = jnp.transpose(smoothed_x, (2, 0, 1))  # Back to (nx, ny, nz)

    # Smooth along Z-axis (already at the end, so just vmap directly)
    chi_smoothed_z = jax.vmap(jax.vmap(smooth_z))(chi_smoothed_x)

    return chi_smoothed_z


# --- 7. THE 4D-VAR OBJECTIVE FUNCTION ---
def objective_fn(chi):
    # Transform control variable to physical space via B^{1/2}
    th_v_perturbation = apply_b_half(chi, sigma_x=5.0, sigma_z=2.5)

    current_th_v_guess = blurred_th_v + th_v_perturbation

    # Build the state explicitly
    guess_state = {
        "u": true_initial_state["u"],
        "v": true_initial_state["v"],
        "w": true_initial_state["w"],
        "pi": true_initial_state["pi"],
        "eta_dot": true_initial_state["eta_dot"],
        "rho": true_initial_state["rho"],
        "th_v": current_th_v_guess,
    }

    def scan_fn(state, step_idx):
        next_state = stepper.step(state, step_idx * dt, None, dummy_bc)
        return next_state, next_state["th_v"]

    # Run the guess forward
    _, guess_trajectory_full = jax.lax.scan(scan_fn, guess_state, jnp.arange(num_steps))
    guess_obs = guess_trajectory_full[::steps_per_obs]

    # Data discrepancy loss (Masked MSE)
    squared_error = (guess_obs - true_obs) ** 2
    masked_error = squared_error * H_mask[None, ...]
    data_loss = jnp.sum(masked_error) / jnp.sum(H_mask * len(guess_obs))

    # Background energy penalty (Regularization in the control space)
    reg_loss = jnp.mean(chi**2)

    alpha = 1e-6
    total_loss = data_loss + alpha * reg_loss

    return total_loss, th_v_perturbation


# --- 8. THE OPTIMIZATION LOOP ---
# Start with zero prior adjustments
initial_chi = jnp.zeros((nx, ny, nz))
total_steps = 150

lr_schedule = optax.cosine_decay_schedule(init_value=0.1, decay_steps=total_steps)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.scale_by_adam(),
    optax.scale_by_schedule(lr_schedule),
    optax.scale(-1.0),  # Flips to minimize
)

print(f"\n[OPTIMIZATION] Launching 4D-Var. Mode: {OBSERVATION_MODE.upper()}")
solver = OptaxSolver(objective_fn, optimizer, has_aux=True)

optimal_chi, history = solver.fit(initial_chi, total_steps=total_steps, metric_name="Loss", maximize=False)

best_idx = np.argmin(history["loss"])
best_perturbation = history["aux"][best_idx]
best_recovered_th_v = blurred_th_v + best_perturbation

# --- 9. VISUALIZE THE RECONSTRUCTION ---
print("\n[PLOT] Saving reconstruction results...")
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Plot middle slice (y=1)
vmax = 5.0
vmin = -5.0

anom_true = true_initial_state["th_v"][:, 1, :] - bg_ref["th_v"][:, 1, :]
anom_blurred = blurred_th_v[:, 1, :] - bg_ref["th_v"][:, 1, :]
anom_recovered = best_recovered_th_v[:, 1, :] - bg_ref["th_v"][:, 1, :]
artifact = xr.Dataset(
    data_vars={
        "theta_anomaly": (("state", "x", "z"), np.stack([anom_blurred, anom_recovered, anom_true])),
        "optimization_loss": ("optimization_step", np.asarray(history["loss"])),
    },
    coords={
        "state": ["first_guess", "recovered", "truth"],
        "x": np.asarray(grid.x_m),
        "z": np.asarray(grid.z_m),
        "optimization_step": np.arange(1, len(history["loss"]) + 1),
    },
    attrs={"core": CORE_TYPE, "observation_mode": OBSERVATION_MODE, "assimilation_window_s": assimilation_window},
)
artifact_path = save_plot_dataset(artifact, data_dir / "artifact.nc", experiment="synthetic_4dvar")
print(f"[OUTPUT] Saved {artifact_path}")
if args.no_render:
    raise SystemExit(0)

im0 = axs[0].contourf(
    X_3d[:, 1, :] / 1000, Z_3d[:, 1, :], anom_blurred, cmap="RdBu_r", levels=np.linspace(vmin, vmax, 21)
)
axs[0].set_title("First Guess (Blurred)")
axs[0].set_xlabel("Distance (km)")
axs[0].set_ylabel("Grid Level (z)")

axs[1].contourf(X_3d[:, 1, :] / 1000, Z_3d[:, 1, :], anom_recovered, cmap="RdBu_r", levels=np.linspace(vmin, vmax, 21))
axs[1].set_title(f"Recovered 4D-Var ({OBSERVATION_MODE})")
axs[1].set_xlabel("Distance (km)")

axs[2].contourf(X_3d[:, 1, :] / 1000, Z_3d[:, 1, :], anom_true, cmap="RdBu_r", levels=np.linspace(vmin, vmax, 21))
axs[2].set_title("True Initial State")
axs[2].set_xlabel("Distance (km)")

plt.colorbar(im0, ax=axs, orientation="horizontal", fraction=0.05, pad=0.15, label="th_v Anomaly (K)")
plt.savefig(figure_dir / "4dvar_reconstruction.png", dpi=150)
print(f"Done! Check '4dvar_reconstruction_{OBSERVATION_MODE}.png'.")
