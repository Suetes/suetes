#!/usr/bin/env python3
"""Matched dual-core comparison for the 3-D Schär mountain-wave test."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from suetes.regional3d.euler import Euler3D
from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.steppers import build_dynamical_core

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import xarray as xr
from suetes.shared.artifacts import ArtifactLayout, save_plot_dataset


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output"
CONSTANTS = {
    "g": 9.81, "cp": 1004.0, "Rd": 287.0,
    "cvd": 717.0, "p0": 100_000.0,
}
NX, NY, NZ = 200, 3, 50
DX, DY, DZ = 500.0, 500.0, 400.0
DOMAIN_X, DOMAIN_Z = NX * DX, NZ * DZ
RELAXATION_WIDTH = 8 * DX
EVALUATION_HALF_WIDTH = 20_000.0
N_BV = 0.01
U_BACKGROUND = 10.0


def mountain_numpy(x):
    return 250.0 * np.exp(-(x / 5000.0) ** 2) * np.cos(np.pi * x / 4000.0) ** 2


def result_path(output_dir, core, dx, dz, dt, t_end, stabilization, snapshot_interval):
    suffix = "" if stabilization == 0.0 else f"_nu{stabilization:g}"
    return output_dir / (
        f"schaer_mountain_{core.replace('-', '_')}_dx{dx:g}_dz{dz:g}"
        f"_dt{dt:g}_t{t_end:g}"
        f"{suffix}_snap{snapshot_interval:g}.npz"
    )


def run_core(core, dx, dz, dt, t_end, stabilization, snapshot_interval, output):

    nx = round(DOMAIN_X / dx)
    nz = round(DOMAIN_Z / dz)
    ny = NY
    dy = dx
    if not np.isclose(nx * dx, DOMAIN_X):
        raise ValueError(f"dx={dx:g} must divide the {DOMAIN_X:g} m x-domain")
    if not np.isclose(nz * dz, DOMAIN_Z):
        raise ValueError(f"dz={dz:g} must divide the {DOMAIN_Z:g} m z-domain")

    num_steps = round(t_end / dt)
    snapshot_steps = round(snapshot_interval / dt)
    num_snapshots = round(t_end / snapshot_interval)
    if not np.isclose(num_steps * dt, t_end):
        raise ValueError("t_end must be an integer multiple of dt")
    if not np.isclose(snapshot_steps * dt, snapshot_interval):
        raise ValueError("snapshot_interval must be an integer multiple of dt")
    if not np.isclose(num_snapshots * snapshot_interval, t_end):
        raise ValueError("t_end must be an integer multiple of snapshot_interval")

    def mountain_jax(x, y):
        del y
        return (
            250.0 * jnp.exp(-(x / 5000.0) ** 2)
            * jnp.cos(jnp.pi * x / 4000.0) ** 2
        )

    grid = RegionalGrid3D(
        nx, ny, nz, dx, dy, dz,
        lat_center=45.0, lon_center=0.0, h_func=mountain_jax,
    )
    operators = CGridOperator3D(grid)
    background_physics = Euler3D(
        grid, operators, CONSTANTS, dt=dt, N_bv=N_BV
    )
    theta_bg = background_physics.theta_bg
    pi_bg = background_physics.pi_bg
    rho_bg = (
        CONSTANTS["p0"] / (CONSTANTS["Rd"] * theta_bg)
        * pi_bg ** (CONSTANTS["cvd"] / CONSTANTS["Rd"])
    )
    state = {
        "u": U_BACKGROUND * jnp.ones((nx + 1, ny, nz)),
        "v": jnp.zeros((nx, ny + 1, nz)),
        "w": jnp.zeros((nx, ny, nz + 1)),
        "eta_dot": jnp.zeros((nx, ny, nz + 1)),
        "pi": pi_bg,
        "rho": rho_bg,
        "th_v": theta_bg,
    }

    # Keep the Davies-style relaxation width fixed in physical space.
    relaxation_cells = round(RELAXATION_WIDTH / dx)
    index = jnp.arange(nx)
    boundary_distance = jnp.minimum(index, nx - 1 - index)
    mask_x = jnp.where(
        boundary_distance < relaxation_cells,
        jnp.cos(0.5 * jnp.pi * boundary_distance / relaxation_cells) ** 2,
        0.0,
    )[:, None, None]

    def boundary_conditions(state_in, forcing):
        del forcing
        blended = {}
        for key, value in state_in.items():
            if key in ("u", "v", "th_v", "pi", "rho"):
                mask = (
                    jnp.pad(mask_x, ((0, 1), (0, 0), (0, 0)), mode="edge")
                    if key == "u" else mask_x
                )
                if key == "u":
                    exterior = U_BACKGROUND * jnp.ones_like(value)
                elif key == "v":
                    exterior = jnp.zeros_like(value)
                else:
                    exterior = {"th_v": theta_bg, "pi": pi_bg, "rho": rho_bg}[key]
                blended[key] = (1.0 - mask) * value + mask * exterior
            else:
                blended[key] = value
        return blended

    common = {
        "dt": dt, "alpha": 0.5,
        "nu_div_factor": stabilization, "nu_h_factor": stabilization,
        "damp_height": 12_000.0, "max_damp": 0.5, "N_bv": N_BV,
    }
    if core == "sisl":
        kwargs = {
            **common, "solver_tol": 1.0e-8,
            "solver_maxiter": 30, "solver_restart": 30,
        }
    elif core == "split-explicit":
        kwargs = {**common, "ns": 6}
    else:
        raise ValueError(f"Unknown core {core}")

    stepper, actual_dt = build_dynamical_core(
        core_type=core, grid=grid, operators=operators,
        constants=CONSTANTS, initial_state=state, **kwargs,
    )

    def step_fn(current_state, step_index):
        next_state = stepper.step(
            current_state, step_index * actual_dt,
            forcing=None, bc_fn=boundary_conditions,
        )
        return next_state, jnp.max(jnp.abs(next_state["w"]))

    target_chunk_steps = max(1, round(200.0 / actual_dt))
    chunk_steps = math.gcd(snapshot_steps, target_chunk_steps)
    chunks_per_snapshot = snapshot_steps // chunk_steps

    def advance_chunk(current_state, start_step):
        def scan_step(scan_state, offset):
            return step_fn(scan_state, start_step + offset)
        return jax.lax.scan(scan_step, current_state, jnp.arange(chunk_steps))

    advance_chunk = jax.jit(advance_chunk)
    print(
        f"\n[BENCHMARK] {core.upper()} Schär: {nx}x{ny}x{nz}, "
        f"dt={actual_dt:g} s, steps={num_steps}"
    )
    print(
        f"[BENCHMARK] Compiling {chunk_steps * actual_dt:g} s internal chunk "
        f"({chunks_per_snapshot} chunks per snapshot)..."
    )
    warmup_state, _ = advance_chunk(state, 0)
    jax.tree_util.tree_leaves(warmup_state)[0].block_until_ready()
    del warmup_state

    y_mid = ny // 2
    times = [0.0]
    w_snapshots = [np.asarray(state["w"][:, y_mid, :])]
    flux_snapshots = [np.zeros(nz)]
    current_state = state
    for snapshot in range(num_snapshots):
        max_w = 0.0
        for chunk in range(chunks_per_snapshot):
            start_step = snapshot * snapshot_steps + chunk * chunk_steps
            current_state, metrics = advance_chunk(current_state, start_step)
            jax.tree_util.tree_leaves(current_state)[0].block_until_ready()
            max_w = max(max_w, float(jnp.max(jnp.abs(metrics))))

        u_center = 0.5 * (
            current_state["u"][1:, y_mid, :]
            + current_state["u"][:-1, y_mid, :]
        )
        w_center = 0.5 * (
            current_state["w"][:, y_mid, 1:]
            + current_state["w"][:, y_mid, :-1]
        )
        momentum_flux = jnp.sum(
            current_state["rho"][:, y_mid, :]
            * (u_center - U_BACKGROUND) * w_center * dx,
            axis=0,
        )
        time_now = (snapshot + 1) * snapshot_interval
        times.append(time_now)
        w_snapshots.append(np.asarray(current_state["w"][:, y_mid, :]))
        flux_snapshots.append(np.asarray(momentum_flux))
        print(
            f"    Snapshot: {time_now:7.1f}s / {t_end:.1f}s "
            f"| Max W: {max_w:.4f} m/s"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        w_snapshots=np.stack(w_snapshots),
        momentum_flux_snapshots=np.stack(flux_snapshots),
        snapshot_times=np.asarray(times),
        x=np.asarray(grid.x_m),
        z_w=np.asarray(grid.Z_w[:, y_mid, :]),
        z_m=np.asarray(grid.Z_m[:, y_mid, :]),
        terrain=mountain_numpy(np.asarray(grid.x_m)),
        dx=dx, dz=dz, dt=float(actual_dt), t_end=t_end,
        stabilization=stabilization,
        snapshot_interval=snapshot_interval,
    )
    print(f"[BENCHMARK] Saved {core} result to {output}")


def load_pair(sisl_path, split_path):
    with np.load(sisl_path) as archive:
        sisl = {key: archive[key] for key in archive.files}
    with np.load(split_path) as archive:
        split = {key: archive[key] for key in archive.files}
    for key in ("x", "z_w", "z_m", "snapshot_times"):
        if not np.allclose(sisl[key], split[key]):
            raise ValueError(f"Core results have different {key}")
    return sisl, split


def snapshot_index(result, comparison_time):
    matches = np.flatnonzero(np.isclose(result["snapshot_times"], comparison_time))
    if len(matches) != 1:
        raise ValueError(
            f"No snapshot at {comparison_time:g} s; available: "
            f"{result['snapshot_times']}"
        )
    return int(matches[0])


def agreement(a, b):
    difference = a - b
    if np.linalg.norm(a) == 0.0 and np.linalg.norm(b) == 0.0:
        return 0.0, 1.0, 0.0
    mean_norm = np.linalg.norm(0.5 * (a + b))
    relative_l2 = np.linalg.norm(difference) / max(mean_norm, np.finfo(float).tiny)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    cosine = np.vdot(a, b) / max(denominator, np.finfo(float).tiny)
    return float(relative_l2), float(cosine), float(np.max(np.abs(difference)))


def latex_scientific(value, precision=2):
    """Format a scalar as a math-text scientific-notation expression."""
    mantissa, exponent = f"{value:.{precision}e}".split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def plot_comparison(sisl_path, split_path, output_dir, comparison_time):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    sisl, split = load_pair(sisl_path, split_path)
    index = snapshot_index(sisl, comparison_time)
    # Diagnose the established mountain-wave response near the terrain.  The
    # initially unbalanced flow emits a downstream startup packet, centred near
    # x = U*t, which is deliberately excluded from this stationary-wave window.
    window = np.abs(sisl["x"]) <= EVALUATION_HALF_WIDTH
    w_sisl = sisl["w_snapshots"][index][window]
    w_split = split["w_snapshots"][index][window]
    difference = w_sisl - w_split
    relative_l2, cosine, maximum = agreement(w_sisl, w_split)
    main_limit = max(np.max(np.abs(w_sisl)), np.max(np.abs(w_split)))
    difference_limit = max(maximum, np.finfo(float).eps)
    x_grid = np.broadcast_to(
        (sisl["x"][window] / 1000.0)[:, None], w_sisl.shape
    )
    z_grid = sisl["z_w"][window] / 1000.0

    fig, axes = plt.subplots(
        1, 3, figsize=(18, 5.4), sharex=True, sharey=True,
        constrained_layout=True,
    )
    images = []
    for axis, field, title in zip(
        axes[:2], (w_sisl, w_split), ("(a) SISL", "(b) Split-Explicit")
    ):
        image = axis.contourf(
            x_grid, z_grid, field,
            levels=np.linspace(-main_limit, main_limit, 61),
            cmap="RdBu_r", extend="both",
        )
        images.append(image)
        axis.set_title(title, fontsize=14)
    difference_image = axes[2].contourf(
        x_grid, z_grid, difference,
        levels=np.linspace(-difference_limit, difference_limit, 61),
        cmap="RdBu_r", extend="both",
    )
    axes[2].set_title("(c) SISL $-$ Split-Explicit", fontsize=14)
    axes[2].text(
        0.03, 0.97,
        rf"relative $L_2={latex_scientific(relative_l2)}$" + "\n"
        + rf"cosine $={cosine:.5f}$" + "\n"
        + rf"$L_\infty={latex_scientific(maximum)}$ m s$^{{-1}}$",
        transform=axes[2].transAxes, ha="left", va="top", fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "none"},
    )
    terrain_x = sisl["x"][window] / 1000.0
    terrain_z = sisl["terrain"][window] / 1000.0
    for axis in axes:
        axis.fill_between(terrain_x, 0.0, terrain_z, color="black")
        axis.set_xlabel("Horizontal distance (km)")
        axis.set_ylim(0.0, 20.0)
    axes[0].set_ylabel("Height (km)")
    main_colorbar = fig.colorbar(
        images[0], ax=axes[:2], shrink=0.91,
        label=r"Vertical velocity $w$ (m s$^{-1}$)",
    )
    difference_colorbar = fig.colorbar(
        difference_image, ax=axes[2], shrink=0.91,
        label=r"Difference in $w$ (m s$^{-1}$)",
    )
    for colorbar in (main_colorbar, difference_colorbar):
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        colorbar.formatter = formatter
        colorbar.update_ticks()
    fig.suptitle(
        rf"Schär mountain waves at $t={comparison_time:g}$ s "
        rf"($\Delta x={float(sisl['dx']):g}$ m, "
        rf"$\Delta z={float(sisl['dz']):g}$ m, "
        rf"$\Delta t={float(sisl['dt']):g}$ s)"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if float(sisl["stabilization"]) == 0.0 else f"_nu{float(sisl['stabilization']):g}"
    output_path = output_dir / f"schaer_mountain_dual_core_t{comparison_time:g}{suffix}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    flux_l2, flux_cosine, flux_max = agreement(
        sisl["momentum_flux_snapshots"][index],
        split["momentum_flux_snapshots"][index],
    )
    print("\n--- Schär dual-core comparison ---")
    print(
        "Vertical-velocity evaluation : "
        f"|x| <= {EVALUATION_HALF_WIDTH / 1000:g} km "
        "(downstream startup packet excluded)"
    )
    print(f"Vertical velocity relative L2 : {relative_l2:.6e}")
    print(f"Vertical velocity cosine      : {cosine:.6e}")
    print(f"Vertical velocity max diff    : {maximum:.6e} m/s")
    print(f"Momentum-flux relative L2     : {flux_l2:.6e}")
    print(f"Momentum-flux cosine          : {flux_cosine:.6e}")
    print(f"Momentum-flux max diff        : {flux_max:.6e} kg/s^2")
    print(f"Figure saved to {output_path}")
    return output_path


def plot_evolution(sisl_path, split_path, output_dir):
    import matplotlib.pyplot as plt

    sisl, split = load_pair(sisl_path, split_path)
    times = sisl["snapshot_times"]
    wave_l2, wave_cosine, wave_max = [], [], []
    flux_l2, flux_cosine = [], []
    window = np.abs(sisl["x"]) <= EVALUATION_HALF_WIDTH
    for w_sisl, w_split, f_sisl, f_split in zip(
        sisl["w_snapshots"], split["w_snapshots"],
        sisl["momentum_flux_snapshots"], split["momentum_flux_snapshots"],
    ):
        metrics = agreement(w_sisl[window], w_split[window])
        wave_l2.append(metrics[0]); wave_cosine.append(metrics[1]); wave_max.append(metrics[2])
        metrics = agreement(f_sisl, f_split)
        flux_l2.append(metrics[0]); flux_cosine.append(metrics[1])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    axes[0].plot(times, wave_l2, "o-", label=r"relative $L_2$")
    axes[0].plot(times, wave_max, "s--", label=r"$L_\infty$ (m s$^{-1}$)")
    axes[0].set_title("(a) Vertical-velocity difference")
    axes[0].set_ylabel("Difference metric")
    axes[0].legend()
    axes[1].plot(times, wave_cosine, "o-", label="$w$ cosine")
    axes[1].plot(times, flux_cosine, "s--", label="momentum-flux cosine")
    axes[1].plot(times, flux_l2, "^-.", label="momentum-flux relative $L_2$")
    axes[1].axhline(1.0, color="0.35", lw=1.0)
    axes[1].set_title("(b) Wave and flux agreement")
    axes[1].set_ylabel("Similarity or relative difference")
    axes[1].legend()
    for axis in axes:
        axis.set_xlabel("Simulation time (s)")
        axis.grid(True, ls="--", alpha=0.4)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if float(sisl["stabilization"]) == 0.0 else f"_nu{float(sisl['stabilization']):g}"
    output_path = output_dir / f"schaer_mountain_cross_core_evolution{suffix}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("\n time[s]   w_rel_L2   w_cosine   flux_rel_L2   flux_cosine")
    for values in zip(times, wave_l2, wave_cosine, flux_l2, flux_cosine):
        print(f" {values[0]:7.1f}   {values[1]:8.5f}   {values[2]:8.5f}   {values[3]:11.5f}   {values[4]:11.5f}")
    print(f"Evolution figure saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dx", type=float, default=DX)
    parser.add_argument("--dz", type=float, default=DZ)
    parser.add_argument("--dt", type=float, default=4.0)
    parser.add_argument("--t-end", type=float, default=7200.0)
    parser.add_argument("--snapshot-interval", type=float, default=1800.0)
    parser.add_argument("--comparison-time", type=float)
    parser.add_argument("--stabilization", type=float, default=0.0)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help="Root containing benchmark, experiment, run, and verification bundles",
    )
    parser.add_argument("--name", default="default", help="Execution label")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path,
        help="Explicit flat directory (legacy compatibility override)",
    )
    parser.add_argument("--worker-core", choices=("sisl", "split-explicit"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.worker_core:
        if args.worker_output is None:
            raise ValueError("--worker-output is required in worker mode")
        run_core(
            args.worker_core, args.dx, args.dz, args.dt, args.t_end, args.stabilization,
            args.snapshot_interval, args.worker_output,
        )
        return

    if args.output_dir is None:
        layout = ArtifactLayout(
            kind="benchmarks",
            case="schaer_mountain_3d",
            execution=args.name,
            output_root=args.output_root,
        ).create()
        data_dir = layout.data
        figure_dir = layout.figures
    else:
        data_dir = args.output_dir
        figure_dir = args.output_dir

    paths = {
        core: result_path(
            data_dir, core, args.dx, args.dz, args.dt, args.t_end,
            args.stabilization, args.snapshot_interval,
        )
        for core in ("sisl", "split-explicit")
    }
    environment = dict(
        os.environ,
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        MPLCONFIGDIR=str(Path("/tmp") / "suetes-matplotlib"),
    )
    for core, path in paths.items():
        if args.reuse and path.exists():
            print(f"Reusing {path}")
            continue
        command = [
            sys.executable, "-u", str(Path(__file__).resolve()),
            "--worker-core", core, "--worker-output", str(path),
            "--dx", str(args.dx), "--dz", str(args.dz),
            "--dt", str(args.dt), "--t-end", str(args.t_end),
            "--snapshot-interval", str(args.snapshot_interval),
            "--stabilization", str(args.stabilization),
        ]
        subprocess.run(command, check=True, env=environment)

    sisl, split = load_pair(paths["sisl"], paths["split-explicit"])
    dataset = xr.Dataset(
        data_vars={
            "vertical_velocity": (
                ("core", "time", "x", "z_interface"),
                np.stack([sisl["w_snapshots"], split["w_snapshots"]]),
            ),
            "momentum_flux": (
                ("core", "time", "z_level"),
                np.stack([
                    sisl["momentum_flux_snapshots"],
                    split["momentum_flux_snapshots"],
                ]),
            ),
            "physical_height_interface": (
                ("x", "z_interface"), sisl["z_w"]
            ),
            "physical_height_level": (("x", "z_level"), sisl["z_m"]),
            "terrain_height": ("x", sisl["terrain"]),
        },
        coords={
            "core": ["sisl", "split-explicit"],
            "time": sisl["snapshot_times"],
            "x": sisl["x"],
            "z_interface": np.arange(sisl["z_w"].shape[-1]),
            "z_level": np.arange(sisl["z_m"].shape[-1]),
        },
        attrs={
            "dx_m": args.dx, "dz_m": args.dz, "dt_s": args.dt,
            "t_end_s": args.t_end, "stabilization": args.stabilization,
            "snapshot_interval_s": args.snapshot_interval,
        },
    )
    artifact = save_plot_dataset(
        dataset,
        data_dir / "artifact.nc",
        experiment="schaer_mountain_dual_core_3d",
    )
    print(f"Saved plot-ready dual-core artifact to {artifact}")
    if not args.no_render:
        comparison_time = (
            args.t_end if args.comparison_time is None else args.comparison_time
        )
        plot_comparison(
            paths["sisl"], paths["split-explicit"],
            figure_dir, comparison_time,
        )
        plot_evolution(paths["sisl"], paths["split-explicit"], figure_dir)


if __name__ == "__main__":
    main()
