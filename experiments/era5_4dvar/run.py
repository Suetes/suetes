# Case runner.
"""
Multivariate 4D-Var Downscaling Benchmark (NAM-22).
Recovers high-resolution dynamics from a blurred first guess using
sparse surface and vertical column observations. Includes memory-efficient
gradient checkpointing and C-grid staggered observation masks.
"""

import os
import argparse
from pathlib import Path
import time
import numpy as np
from scipy.ndimage import gaussian_filter

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import xarray as xr

from suetes.preprocessing.era5downloader import ERA5Manager
from suetes.preprocessing.processor import ERA5Processor
from suetes.preprocessing.topography import TopographyProcessor
from suetes.preprocessing.era2suetes import BoundaryProcessor, TimeManager
from suetes.shared.transforms import SleveSimple
from suetes.shared.optimization import OptaxSolver

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.boundaries import DaviesSponge
from suetes.regional3d.steppers import build_dynamical_core

from suetes.physics.base import PhysicsSuite
from suetes.physics.forcing import NewtonianRelaxation
from suetes.physics.surface import McFarlaneSurfaceDrag
from suetes.physics.turbulence import McFarlaneVerticalDiffusion
from suetes.shared.artifacts import ArtifactLayout, save_plot_dataset

# =====================================================================
# 0. CONFIGURATION
# =====================================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
DATA_DIR = "inputs"
ACTIVE_DOMAIN = "nam22"
CORE_TYPE = "split-explicit"

# Grid specs matching your NAM-22 setup
nx, ny, nz = 64, 64, 32
dx, dy, dz = 22000.0, 22000.0, 500.0
dt = 40.0
sponge_depth = 15

# Assimilation Window parameters
assimilation_hours = 6.0
sim_time_seconds = assimilation_hours * 3600.0
num_steps = int(sim_time_seconds / dt)
obs_interval = 1800.0  # Take an observation every 30 mins
steps_per_obs = int(obs_interval / dt)
num_obs_windows = num_steps // steps_per_obs

constants = {"g": 9.81, "Rd": 287.0, "cp": 1004.0, "cvd": 717.0, "p0": 100000.0, "epsilon": 0.622}
PROGNOSTIC_VARS = ["u", "v", "w", "pi", "rho", "th_v"]

# Characteristic standard deviations for preconditioning (physical scaling)
SIGMAS = {"u": 10.0, "v": 10.0, "w": 0.5, "pi": 0.01, "rho": 0.05, "th_v": 5.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--name", default="default")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        layout = ArtifactLayout(
            kind="experiments", case="era5_4dvar", execution=args.name, output_root=args.output_root
        ).create()
        data_dir, figure_dir = layout.data, layout.figures
    else:
        data_dir = figure_dir = args.output_dir
        data_dir.mkdir(parents=True, exist_ok=True)
    print(f"[CONFIG] Starting 4D-Var Benchmark on {ACTIVE_DOMAIN.upper()}")

    # ---------------------------------------------------------
    # 1. SETUP GRID, TOPOGRAPHY & ERA5 FORCING
    # ---------------------------------------------------------
    lat_c, lon_c = 47.5, -97.0
    dynamic_bbox = ERA5Manager.calculate_required_bbox(lat_c, lon_c, nx, ny, dx, dy, buffer_deg=3.0)
    manager = ERA5Manager(data_dir=DATA_DIR, pressure_levels="buffered")

    cache_prefix = f"{ACTIVE_DOMAIN}_202507_Nx{nx}_Ny{ny}"
    sl_file, pl_file = manager.download_regional_subset(
        year="2025", month="07", days=["18", "19"], area=dynamic_bbox, prefix=cache_prefix
    )

    base_grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_c, lon_c)
    topo_proc = TopographyProcessor(era5_sl_path=sl_file, gebco_path=os.path.join(DATA_DIR, "gebco_data.nc"))
    h_func = topo_proc.process_and_blend(base_grid, sponge_depth=sponge_depth, smooth_sigma=2.0)

    grid = RegionalGrid3D(
        nx, ny, nz, dx, dy, dz, lat_c, lon_c, h_func=h_func, transform=SleveSimple(scale_s=10000.0, n=1.0)
    )
    operators = CGridOperator3D(grid)

    era5_proc = ERA5Processor(pl_path=pl_file, sl_path=sl_file)
    raw_t0 = era5_proc.get_stitched_state(time_idx=0)
    bridge = BoundaryProcessor(grid, raw_t0["latitude"], raw_t0["longitude"], constants)

    num_era5_states = int(assimilation_hours) + 2

    state_cache_path = os.path.join(DATA_DIR, f"{cache_prefix}_native_{int(assimilation_hours)}h.pkl")

    bc_states_native = bridge.build_or_load_timeseries(
        era5_proc=era5_proc, num_states=num_era5_states, cache_path=state_cache_path, coarsen_window=None
    )
    time_manager = TimeManager(bc_states_native, [float(i * 3600.0) for i in range(num_era5_states)], grid)

    # ---------------------------------------------------------
    # 2. PHYSICS & DYNAMICAL CORE
    # ---------------------------------------------------------
    initial_state_true = dict(bc_states_native[0])
    initial_state_true["theta_surf"] = initial_state_true["theta_skt"]
    initial_state_true["target_th_v"] = initial_state_true["th_v"]

    sponge = DaviesSponge(grid, operators, sponge_depth=sponge_depth, dt=dt, tau_bndy_factor=10.0)

    physics_suite = PhysicsSuite()
    z_0_field = (
        bridge.process_static(raw_t0)["land_fraction"] * 0.1
        + (1.0 - bridge.process_static(raw_t0)["land_fraction"]) * 1e-4
    )
    physics_suite.add_tendency_scheme(
        McFarlaneSurfaceDrag(
            grid,
            operators,
            constants,
            z_0=z_0_field,
            epsilon=jnp.zeros_like(z_0_field),
            theta_surf=initial_state_true["theta_surf"],
        )
    )
    physics_suite.add_tendency_scheme(
        McFarlaneVerticalDiffusion(grid, operators, constants, epsilon=jnp.zeros_like(z_0_field)[..., None])
    )
    physics_suite.add_tendency_scheme(NewtonianRelaxation(tau_relax_hours=6.0))

    stepper, _ = build_dynamical_core(
        core_type=CORE_TYPE,
        grid=grid,
        operators=operators,
        constants=constants,
        initial_state=initial_state_true,
        physics_suite=physics_suite,
        interior_mask=sponge.get_interior_mask(),
        dt=dt,
        ns=3,
        damp_height=12000.0,
        max_damp=3.0,
    )

    # ---------------------------------------------------------
    # 3. REUSABLE TRAJECTORY GENERATOR (MEMORY EFFICIENT)
    # ---------------------------------------------------------
    def run_trajectory(start_state):
        @jax.checkpoint
        def advance_one_obs_window(state, start_step_idx):
            def inner_step(s, i):
                t_curr = (start_step_idx + i) * dt
                bc_state_t = time_manager.get_forcing(t_curr)

                # Maintain surface boundary conditions
                s["theta_surf"] = bc_state_t["theta_skt"]
                s["target_th_v"] = bc_state_t["th_v"]

                def bc_fn(s_next, _):
                    return sponge.blend(s_next, bc_state_t)

                next_s = stepper.step(s, t_curr, None, bc_fn)
                return next_s, None

            state_at_obs, _ = jax.lax.scan(inner_step, state, jnp.arange(steps_per_obs))
            obs_state = {k: state_at_obs[k] for k in PROGNOSTIC_VARS}
            return state_at_obs, obs_state

        start_indices = jnp.arange(num_obs_windows) * steps_per_obs
        final_state, obs_trajectory = jax.lax.scan(advance_one_obs_window, start_state, start_indices)
        return final_state, obs_trajectory

    # ---------------------------------------------------------
    # 4. CREATE GROUND TRUTH TRAJECTORY
    # ---------------------------------------------------------
    print("\n[TRUTH] Running native resolution nature run to generate observations...")
    # Wrap in JIT for execution speed
    run_trajectory_jit = jax.jit(run_trajectory)
    _, true_obs_trajectory = run_trajectory_jit(initial_state_true)

    # ---------------------------------------------------------
    # 5. DENSE SATELLITE & SPARSE C-GRID MASKS
    # ---------------------------------------------------------
    H_masks = {}

    for k in PROGNOSTIC_VARS:
        shape = initial_state_true[k].shape
        H_masks[k] = jnp.zeros(shape)

        if k == "th_v":
            # SATELLITE: Observe EVERY grid point in the domain
            H_masks[k] = H_masks[k].at[:, :, :].set(1.0)
        else:
            # SURFACE: Keep winds and pressure sparse (every 5th point)
            H_masks[k] = H_masks[k].at[5:-5:5, 5:-5:5, 0].set(1.0)

    # ---------------------------------------------------------
    # 6. BLURRED FIRST GUESS & B-MATRIX TRANSFORM
    # ---------------------------------------------------------
    blurred_background = {}
    for k in PROGNOSTIC_VARS:
        var_np = np.array(initial_state_true[k])

        # 1. Massive spatial blur (destroys almost all structural information)
        sabotaged_field = gaussian_filter(var_np, sigma=(15.0, 15.0, 1.0))

        # 2. Add systematic physical biases
        if k == "th_v":
            sabotaged_field -= 2.0  # Cold bias of 2 Kelvin everywhere
        elif k in ["u", "v"]:
            sabotaged_field *= 0.5  # Cut the wind speeds in half (momentum deficit)

        blurred_background[k] = jnp.array(sabotaged_field)

    def string_gaussian_kernel(sigma, radius):
        x = jnp.arange(-radius, radius + 1)
        kernel = jnp.exp(-0.5 * (x / sigma) ** 2)
        return kernel / jnp.sum(kernel)

    def apply_b_half(chi, sigma_h=3.0, sigma_z=1.0):
        """Applies separable 1D Gaussian smoothing safely across any grid shape."""
        rad_h = int(3 * sigma_h)
        rad_z = int(3 * sigma_z)

        ker_h = string_gaussian_kernel(sigma_h, rad_h)
        ker_z = string_gaussian_kernel(sigma_z, rad_z)

        def smooth_1d(field_1d, kernel):
            res = jnp.convolve(field_1d, kernel, mode="same")
            start = (res.shape[0] - field_1d.shape[0]) // 2
            return res[start : start + field_1d.shape[0]]

        # --- Helper: Smooths along the LAST axis of an ND array ---
        def smooth_last_axis(arr, kernel):
            orig_shape = arr.shape
            # Flatten all dimensions except the last one
            arr_flat = arr.reshape(-1, orig_shape[-1])
            # Vmap the 1D smoother over the flattened batch dimension
            smoothed_flat = jax.vmap(lambda f: smooth_1d(f, kernel))(arr_flat)
            # Restore the original ND shape
            return smoothed_flat.reshape(orig_shape)

        # 1. Smooth along X (dim 0): Transpose X to last, smooth, transpose back
        chi_x = jnp.transpose(chi, (1, 2, 0))
        chi_x = smooth_last_axis(chi_x, ker_h)
        chi_x = jnp.transpose(chi_x, (2, 0, 1))

        # 2. Smooth along Y (dim 1): Transpose Y to last, smooth, transpose back
        chi_y = jnp.transpose(chi_x, (0, 2, 1))
        chi_y = smooth_last_axis(chi_y, ker_h)
        chi_y = jnp.transpose(chi_y, (0, 2, 1))

        # 3. Smooth along Z (dim 2): Z is already the last dimension
        chi_z = smooth_last_axis(chi_y, ker_z)

        return chi_z

    def transform_to_physical(chi_dict):
        perturbations = {}
        for k in PROGNOSTIC_VARS:
            # th_v gets sigma_h = 0.5 (almost no horizontal smoothing)
            # Winds get 1.5 (moderate smoothing)
            # pi gets 2.5 (heavy smoothing to maintain acoustic balance)
            if k == "th_v":
                current_sigma = 0.5
            elif k == "pi":
                current_sigma = 2.5
            else:
                current_sigma = 1.5

            smoothed = apply_b_half(chi_dict[k], sigma_h=current_sigma, sigma_z=0.5)
            perturbations[k] = smoothed * SIGMAS[k]
        return perturbations

    # ---------------------------------------------------------
    # 7. OBJECTIVE FUNCTION
    # ---------------------------------------------------------
    @jax.jit
    def objective_fn(chi_dict):
        perturbations = transform_to_physical(chi_dict)

        guess_state = {**blurred_background}
        for k in PROGNOSTIC_VARS:
            guess_state[k] = blurred_background[k] + perturbations[k]

        guess_state["theta_surf"] = initial_state_true["theta_surf"]
        guess_state["target_th_v"] = initial_state_true["target_th_v"]
        guess_state["eta_dot"] = initial_state_true["eta_dot"]

        # --- Dynamic Balance Penalty (Initial Tendencies) ---
        bc_state_t0 = time_manager.get_forcing(0.0)

        def bc_fn_t0(s, _):
            return sponge.blend(s, bc_state_t0)

        step_1_state = stepper.step(guess_state, 0.0, None, bc_fn_t0)

        dw_dt = (step_1_state["w"] - guess_state["w"]) / dt
        dpi_dt = (step_1_state["pi"] - guess_state["pi"]) / dt
        balance_penalty = jnp.mean(dw_dt**2) + 100.0 * jnp.mean(dpi_dt**2)

        # --- Forward Integration (Checkpointed) ---
        _, guess_obs_trajectory = run_trajectory(guess_state)

        # --- Losses ---
        data_loss = 0.0
        for k in PROGNOSTIC_VARS:
            # 1. Get the squared error everywhere
            sq_err = (guess_obs_trajectory[k] - true_obs_trajectory[k]) ** 2

            # 2. Mask it (applies 0.0 to unobserved regions)
            masked_sq_err = sq_err * H_masks[k]

            # 3. Count the actual number of observations (spatial points * time windows)
            num_time_windows = guess_obs_trajectory[k].shape[0]
            num_active_sensors = jnp.sum(H_masks[k]) * num_time_windows

            # 4. Calculate the true MSE at the sensor locations
            # (Adding a small epsilon to avoid division by zero)
            mse = jnp.sum(masked_sq_err) / (num_active_sensors + 1e-8)

            # Normalize by characteristic variance
            data_loss += mse / (SIGMAS[k] ** 2)

        reg_loss = sum(jnp.mean(chi**2) for chi in chi_dict.values())

        alpha_b = 0.001  # Dropped from 0.05
        alpha_bal = 0.01  # Soften the tendency penalty slightly

        total_loss = data_loss + (alpha_b * reg_loss) + (alpha_bal * balance_penalty)
        return total_loss, guess_state

    # ---------------------------------------------------------
    # 8. OPTIMIZATION LOOP
    # ---------------------------------------------------------
    print("\n[OPTIMIZATION] Initializing Adjoint-Free 4D-Var...")

    chi_init = {k: jnp.zeros_like(initial_state_true[k]) for k in PROGNOSTIC_VARS}

    total_steps = 100
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.001, peak_value=0.01, warmup_steps=20, decay_steps=total_steps, end_value=0.0001
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.1),  # Tightened from 1.0
        optax.scale_by_adam(),
        optax.scale_by_schedule(lr_schedule),
        optax.scale(-1.0),
    )

    solver = OptaxSolver(objective_fn, optimizer, has_aux=True)

    optimal_chi, history = solver.fit(chi_init, total_steps=total_steps, metric_name="Cost Function", maximize=False)

    best_idx = np.argmin(history["loss"])
    recovered_state = history["aux"][best_idx]
    variables = ["th_v", "u", "pi"]
    dataset = xr.Dataset(
        data_vars={
            "surface_field": (
                ("state", "variable", "x", "y"),
                np.stack(
                    [
                        np.stack([np.asarray(blurred_background[var][:, :, 0]) for var in variables]),
                        np.stack([np.asarray(recovered_state[var][:, :, 0]) for var in variables]),
                        np.stack([np.asarray(initial_state_true[var][:, :, 0]) for var in variables]),
                    ]
                ),
            ),
            "optimization_loss": ("optimization_step", np.asarray(history["loss"])),
        },
        coords={
            "state": ["first_guess", "recovered", "truth"],
            "variable": variables,
            "x": np.arange(nx),
            "y": np.arange(ny),
            "optimization_step": np.arange(1, len(history["loss"]) + 1),
        },
        attrs={"domain": ACTIVE_DOMAIN, "core": CORE_TYPE, "assimilation_hours": assimilation_hours},
    )
    artifact_path = save_plot_dataset(dataset, data_dir / "artifact.nc", experiment="era5_4dvar")
    print(f"[OUTPUT] Saved {artifact_path}")
    if args.no_render:
        return

    # ---------------------------------------------------------
    # 9. VISUALIZATION
    # ---------------------------------------------------------
    print("\n[PLOT] Saving Multi-Variate Reconstructions...")
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    z_idx = 0

    for i, var in enumerate(["th_v", "u", "pi"]):
        vmin = np.min(initial_state_true[var][:, :, z_idx])
        vmax = np.max(initial_state_true[var][:, :, z_idx])

        axs[i, 0].imshow(blurred_background[var][:, :, z_idx].T, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
        axs[i, 0].set_title(f"Blurred Guess ({var})")

        axs[i, 1].imshow(recovered_state[var][:, :, z_idx].T, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
        axs[i, 1].set_title(f"4D-Var Recovered ({var})")

        im = axs[i, 2].imshow(
            initial_state_true[var][:, :, z_idx].T, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis"
        )
        axs[i, 2].set_title(f"True ERA5 ({var})")
        plt.colorbar(im, ax=axs[i, 2])

    plt.tight_layout()
    figure_path = figure_dir / "multivariate_4dvar_recovery.png"
    plt.savefig(figure_path, dpi=150)
    print(f"Benchmark Complete. Check {figure_path}")


if __name__ == "__main__":
    main()
