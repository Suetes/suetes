#!/usr/bin/env python3
"""Compare the SISL and Split-Explicit cores on a rising thermal bubble.

The two integrations use the same grid, large time step, initial condition,
Rayleigh damping, and explicit stabilization.  Each core runs in a fresh
process so that XLA allocations are released before the other core starts.
The parent process then plots both solutions on a shared colour scale and
their signed difference on a separate symmetric scale.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from suetes.regional3d.euler import Euler3D
from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.steppers import build_dynamical_core
from suetes.shared.artifacts import ArtifactLayout, save_plot_dataset

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output"
DOMAIN_WIDTH = 10_000.0
DOMAIN_HEIGHT = 10_000.0
CONSTANTS = {"g": 9.81, "cp": 1004.0, "Rd": 287.0, "cvd": 717.0, "p0": 100_000.0}


def result_path(
    output_dir: Path, core: str, dx: float, dt: float, t_end: float, stabilization: float, snapshot_interval: float
) -> Path:
    core_name = core.replace("-", "_")
    stabilization_suffix = "" if stabilization == 0.0 else f"_nu{stabilization:g}"
    return output_dir / (
        f"rising_bubble_{core_name}_dx{dx:g}_dt{dt:g}_t{t_end:g}{stabilization_suffix}_snap{snapshot_interval:g}.npz"
    )


def run_core(
    core: str, dx: float, dt: float, t_end: float, stabilization: float, snapshot_interval: float, output: Path
) -> None:

    nx = round(DOMAIN_WIDTH / dx)
    nz = round(DOMAIN_HEIGHT / dx)
    ny = 3
    if not np.isclose(nx * dx, DOMAIN_WIDTH) or not np.isclose(nz * dx, DOMAIN_HEIGHT):
        raise ValueError("dx must divide the 10 km domain exactly")
    num_steps = round(t_end / dt)
    if not np.isclose(num_steps * dt, t_end):
        raise ValueError("t_end must be an integer multiple of dt")
    snapshot_steps = round(snapshot_interval / dt)
    if not np.isclose(snapshot_steps * dt, snapshot_interval):
        raise ValueError("snapshot_interval must be an integer multiple of dt")
    num_segments = round(t_end / snapshot_interval)
    if not np.isclose(num_segments * snapshot_interval, t_end):
        raise ValueError("t_end must be an integer multiple of snapshot_interval")

    grid = RegionalGrid3D(nx, ny, nz, dx, dx, dx, lat_center=0.0, lon_center=0.0)
    operators = CGridOperator3D(grid)
    background_physics = Euler3D(grid, operators, CONSTANTS, dt=dt, N_bv=0.0)
    theta_bg = background_physics.theta_bg
    pi_bg = background_physics.pi_bg
    rho_bg = CONSTANTS["p0"] / (CONSTANTS["Rd"] * theta_bg) * pi_bg ** (CONSTANTS["cvd"] / CONSTANTS["Rd"])

    state = {
        "u": jnp.zeros((nx + 1, ny, nz), dtype=jnp.float64),
        "v": jnp.zeros((nx, ny + 1, nz), dtype=jnp.float64),
        "w": jnp.zeros((nx, ny, nz + 1), dtype=jnp.float64),
        "eta_dot": jnp.zeros((nx, ny, nz + 1), dtype=jnp.float64),
        "pi": pi_bg,
        "rho": rho_bg,
    }

    x, _, z = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_m, indexing="ij")
    radius = jnp.sqrt(x**2 + (z - 2000.0) ** 2)
    theta_prime = jnp.where(radius <= 1500.0, 2.0 * jnp.cos(0.5 * jnp.pi * radius / 1500.0) ** 2, 0.0)
    state["th_v"] = theta_bg + theta_prime
    state["rho"] = CONSTANTS["p0"] / (CONSTANTS["Rd"] * state["th_v"]) * pi_bg ** (CONSTANTS["cvd"] / CONSTANTS["Rd"])

    common_kwargs = {
        "dt": dt,
        "alpha": 0.5,
        "nu_div_factor": stabilization,
        "nu_h_factor": stabilization,
        "damp_height": 7500.0,
        "max_damp": 0.05,
        "N_bv": 0.0,
    }
    if core == "sisl":
        core_kwargs = {**common_kwargs, "solver_tol": 1.0e-8, "solver_maxiter": 30, "solver_restart": 30}
    elif core == "split-explicit":
        core_kwargs = {**common_kwargs, "ns": 12}
    else:
        raise ValueError(f"Unknown core: {core}")

    stepper, actual_dt = build_dynamical_core(
        core_type=core, grid=grid, operators=operators, constants=CONSTANTS, initial_state=state, **core_kwargs
    )

    def step_fn(current_state, step_index):
        next_state = stepper.step(current_state, step_index * actual_dt, forcing=None, bc_fn=lambda value, field: value)
        return next_state, jnp.max(jnp.abs(next_state["w"]))

    print(f"\n[BENCHMARK] {core.upper()}: {nx}x{ny}x{nz}, dx={dx:g} m, dt={actual_dt:g} s, steps={num_steps}")
    y_mid = ny // 2

    # Keep the compiled scan modest even when snapshots are far apart. A
    # 200-step Split-Explicit scan contains all RK stages and acoustic
    # subcycles and can overwhelm XLA during compilation. Choose a chunk near
    # 20 simulated seconds that divides each snapshot interval exactly.
    target_chunk_steps = max(1, round(20.0 / actual_dt))
    chunk_steps = math.gcd(snapshot_steps, target_chunk_steps)
    chunks_per_snapshot = snapshot_steps // chunk_steps

    def advance_chunk(current_state, start_step):
        def scan_step(scan_state, step_offset):
            return step_fn(scan_state, start_step + step_offset)

        return jax.lax.scan(scan_step, current_state, jnp.arange(chunk_steps))

    advance_chunk = jax.jit(advance_chunk)
    print(
        f"[BENCHMARK] Compiling {chunk_steps * actual_dt:g} s internal chunk "
        f"({chunks_per_snapshot} chunks per snapshot)..."
    )
    warmup_state, _ = advance_chunk(state, 0)
    jax.tree_util.tree_leaves(warmup_state)[0].block_until_ready()
    del warmup_state

    snapshot_times = [0.0]
    theta_snapshots = [np.asarray(state["th_v"][:, y_mid, :] - theta_bg[:, y_mid, :])]
    final_state = state
    for segment in range(num_segments):
        max_w = 0.0
        for chunk in range(chunks_per_snapshot):
            start_step = segment * snapshot_steps + chunk * chunk_steps
            final_state, metrics = advance_chunk(final_state, start_step)
            jax.tree_util.tree_leaves(final_state)[0].block_until_ready()
            max_w = max(max_w, float(jnp.max(jnp.abs(metrics))))
        current_time = (segment + 1) * snapshot_interval
        snapshot_times.append(current_time)
        theta_snapshots.append(np.asarray(final_state["th_v"][:, y_mid, :] - theta_bg[:, y_mid, :]))
        print(f"    Snapshot: {current_time:6.1f}s / {t_end:.1f}s | Max W: {max_w:.4f} m/s")

    final_theta_prime = final_state["th_v"][:, y_mid, :] - theta_bg[:, y_mid, :]
    cell_volumes = grid.dx * grid.dy * (grid.Z_w[:, :, 1:] - grid.Z_w[:, :, :-1]) / grid.m_factors["m"][..., None] ** 2
    initial_mass = jnp.sum(state["rho"] * cell_volumes, dtype=jnp.float64)
    final_mass = jnp.sum(final_state["rho"] * cell_volumes, dtype=jnp.float64)
    mass_error = jnp.abs(final_mass - initial_mass) / initial_mass
    theta_symmetry_error = jnp.max(jnp.abs(final_theta_prime - final_theta_prime[::-1, :]))

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        theta_prime=np.asarray(final_theta_prime),
        theta_prime_snapshots=np.stack(theta_snapshots),
        snapshot_times=np.asarray(snapshot_times),
        x=np.asarray(grid.x_m),
        z=np.asarray(grid.z_m),
        mass_error=float(mass_error),
        theta_symmetry_error=float(theta_symmetry_error),
        dx=dx,
        dt=actual_dt,
        t_end=t_end,
        stabilization=stabilization,
        snapshot_interval=snapshot_interval,
    )
    print(f"[BENCHMARK] Saved {core} result to {output}")


def plot_comparison(sisl_path: Path, split_path: Path, output_dir: Path, comparison_time: float) -> Path:
    from matplotlib.ticker import ScalarFormatter

    def latex_scientific(value: float, precision: int = 2) -> str:
        mantissa, exponent = f"{value:.{precision}e}".split("e")
        return rf"{mantissa}\times10^{{{int(exponent)}}}"

    with np.load(sisl_path) as archive:
        sisl = {key: archive[key] for key in archive.files}
    with np.load(split_path) as archive:
        split = {key: archive[key] for key in archive.files}
    # Backward compatibility with results generated before stabilization was
    # made configurable; those runs used zero for both factors.
    sisl.setdefault("stabilization", np.asarray(0.0))
    split.setdefault("stabilization", np.asarray(0.0))

    for coordinate in ("x", "z"):
        if not np.allclose(sisl[coordinate], split[coordinate]):
            raise ValueError(f"Core results use different {coordinate} coordinates")
    for scalar in ("dx", "dt", "t_end", "stabilization"):
        if not np.isclose(float(sisl[scalar]), float(split[scalar])):
            raise ValueError(f"Core results use different {scalar} values")

    def field_at_time(result: dict[str, np.ndarray]) -> np.ndarray:
        if "theta_prime_snapshots" not in result:
            if np.isclose(comparison_time, float(result["t_end"])):
                return result["theta_prime"]
            raise ValueError("Stored result contains no intermediate snapshots")
        matches = np.flatnonzero(np.isclose(result["snapshot_times"], comparison_time))
        if len(matches) != 1:
            raise ValueError(
                f"No stored snapshot at t={comparison_time:g} s; available times are {result['snapshot_times']}"
            )
        return result["theta_prime_snapshots"][matches[0]]

    theta_sisl = field_at_time(sisl)
    theta_split = field_at_time(split)
    difference = theta_sisl - theta_split
    reference_norm = np.linalg.norm(0.5 * (theta_sisl + theta_split))
    relative_l2 = np.linalg.norm(difference) / max(reference_norm, np.finfo(float).tiny)
    max_difference = float(np.max(np.abs(difference)))
    cosine_similarity = float(
        np.vdot(theta_sisl, theta_split) / (np.linalg.norm(theta_sisl) * np.linalg.norm(theta_split))
    )

    def anomaly_moments(field: np.ndarray) -> tuple[float, float, float, float]:
        positive = np.maximum(field, 0.0)
        anomaly_sum = float(np.sum(positive))
        centroid = float(np.sum(positive * sisl["z"][None, :]) / anomaly_sum)
        vertical_spread = float(np.sqrt(np.sum(positive * (sisl["z"][None, :] - centroid) ** 2) / anomaly_sum))
        return anomaly_sum, centroid, vertical_spread, float(np.max(field))

    sisl_moments = anomaly_moments(theta_sisl)
    split_moments = anomaly_moments(theta_split)

    x_km = sisl["x"] / 1000.0
    z_km = sisl["z"] / 1000.0
    main_limit = float(max(np.max(np.abs(theta_sisl)), np.max(np.abs(theta_split))))
    difference_limit = max(max_difference, np.finfo(float).eps)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), sharex=True, sharey=True, constrained_layout=True)
    main_images = []
    for axis, field, title in zip(axes[:2], (theta_sisl, theta_split), ("(a) SISL", "(b) Split-Explicit")):
        image = axis.contourf(
            x_km, z_km, field.T, levels=np.linspace(-main_limit, main_limit, 61), cmap="RdBu_r", extend="both"
        )
        main_images.append(image)
        axis.set_title(title, fontsize=14)

    difference_image = axes[2].contourf(
        x_km,
        z_km,
        difference.T,
        levels=np.linspace(-difference_limit, difference_limit, 61),
        cmap="coolwarm",
        extend="both",
    )
    axes[2].set_title("(c) SISL $-$ Split-Explicit", fontsize=14)
    axes[2].text(
        0.03,
        0.97,
        rf"relative $L_2={latex_scientific(relative_l2)}$" + "\n" + rf"$L_\infty={latex_scientific(max_difference)}$ K",
        transform=axes[2].transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
    )

    for axis in axes:
        axis.set_xlabel("Horizontal distance (km)", fontsize=11)
        axis.set_aspect("equal")
        axis.set_xlim(x_km[0], x_km[-1])
        axis.set_ylim(z_km[0], z_km[-1])
    axes[0].set_ylabel("Height (km)", fontsize=11)

    main_colorbar = fig.colorbar(
        main_images[0], ax=axes[:2], shrink=0.91, label=r"Virtual potential-temperature perturbation $\theta_v'$ (K)"
    )
    difference_colorbar = fig.colorbar(
        difference_image, ax=axes[2], shrink=0.91, label=r"Difference in $\theta_v'$ (K)"
    )
    for colorbar in (main_colorbar, difference_colorbar):
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        colorbar.formatter = formatter
        colorbar.update_ticks()

    dx = float(sisl["dx"])
    dt = float(sisl["dt"])
    t_end = float(sisl["t_end"])
    stabilization = float(sisl["stabilization"])
    stabilization_text = (
        "no explicit stabilization" if stabilization == 0.0 else rf"stabilization factor $={stabilization:g}$"
    )
    fig.suptitle(
        rf"Rising thermal bubble at $t={comparison_time:g}$ s "
        rf"($\Delta x={dx:g}$ m, $\Delta t={dt:g}$ s; " + stabilization_text + ")",
        fontsize=14,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stabilization_suffix = "" if stabilization == 0.0 else f"_nu{stabilization:g}"
    output_path = output_dir / (
        f"rising_bubble_dual_core_3d_dx{dx:g}_dt{dt:g}_t{comparison_time:g}{stabilization_suffix}.png"
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\n--- Dual-core comparison ---")
    print(f"Relative L2 difference : {relative_l2:.6e}")
    print(f"Maximum point difference: {max_difference:.6e} K")
    print(f"Cosine similarity      : {cosine_similarity:.6e}")
    print(f"Positive-anomaly integral ratio (SISL/Split): {sisl_moments[0] / split_moments[0]:.6e}")
    print(f"Anomaly centroid difference (SISL-Split)    : {sisl_moments[1] - split_moments[1]:.3f} m")
    print(f"Vertical-spread difference (SISL-Split)     : {sisl_moments[2] - split_moments[2]:.3f} m")
    print(f"Peak-amplitude ratio (SISL/Split)           : {sisl_moments[3] / split_moments[3]:.6e}")
    for core, result in (("SISL", sisl), ("Split-Explicit", split)):
        print(
            f"{core:14s} | mass drift={float(result['mass_error']):.3e} "
            f"| theta symmetry error={float(result['theta_symmetry_error']):.3e} K"
        )
    print(f"Figure saved to {output_path}")
    return output_path


def plot_evolution(sisl_path: Path, split_path: Path, output_dir: Path) -> Path:
    """Plot the growth and character of cross-core differences over time."""
    with np.load(sisl_path) as archive:
        sisl = {key: archive[key] for key in archive.files}
    with np.load(split_path) as archive:
        split = {key: archive[key] for key in archive.files}
    if "theta_prime_snapshots" not in sisl or "theta_prime_snapshots" not in split:
        raise ValueError("Evolution plotting requires snapshot-enabled result files")
    if not np.allclose(sisl["snapshot_times"], split["snapshot_times"]):
        raise ValueError("The two cores have different snapshot times")

    times = sisl["snapshot_times"]
    z = sisl["z"]
    relative_l2 = []
    cosine = []
    max_difference = []
    centroid_difference = []
    spread_difference = []
    integral_ratio = []
    peak_ratio = []

    def moments(field):
        positive = np.maximum(field, 0.0)
        total = float(np.sum(positive))
        centroid = float(np.sum(positive * z[None, :]) / total)
        spread = float(np.sqrt(np.sum(positive * (z[None, :] - centroid) ** 2) / total))
        return total, centroid, spread, float(np.max(field))

    for theta_sisl, theta_split in zip(sisl["theta_prime_snapshots"], split["theta_prime_snapshots"]):
        difference = theta_sisl - theta_split
        mean_norm = np.linalg.norm(0.5 * (theta_sisl + theta_split))
        relative_l2.append(np.linalg.norm(difference) / max(mean_norm, np.finfo(float).tiny))
        denominator = np.linalg.norm(theta_sisl) * np.linalg.norm(theta_split)
        cosine.append(np.vdot(theta_sisl, theta_split) / max(denominator, np.finfo(float).tiny))
        max_difference.append(np.max(np.abs(difference)))
        sisl_moments = moments(theta_sisl)
        split_moments = moments(theta_split)
        centroid_difference.append(sisl_moments[1] - split_moments[1])
        spread_difference.append(sisl_moments[2] - split_moments[2])
        integral_ratio.append(sisl_moments[0] / split_moments[0])
        peak_ratio.append(sisl_moments[3] / split_moments[3])

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), constrained_layout=True)
    axes[0].plot(times, relative_l2, "o-", color="#0072B2", label=r"relative $L_2$")
    axes[0].plot(times, max_difference, "s--", color="#D55E00", label=r"$L_\infty$ (K)")
    axes[0].set_title("(a) Pointwise difference")
    axes[0].set_ylabel("Difference metric")
    axes[0].legend()

    axes[1].plot(times, centroid_difference, "o-", color="#009E73", label="centroid difference")
    axes[1].plot(times, spread_difference, "s--", color="#CC79A7", label="spread difference")
    axes[1].axhline(0.0, color="0.35", lw=1.0)
    axes[1].set_title("(b) Vertical structure")
    axes[1].set_ylabel("SISL $-$ Split-Explicit (m)")
    axes[1].legend()

    axes[2].plot(times, integral_ratio, "o-", color="#56B4E9", label="positive-anomaly integral")
    axes[2].plot(times, peak_ratio, "s--", color="#E69F00", label="peak amplitude")
    axes[2].plot(times, cosine, "^-.", color="#000000", label="cosine similarity")
    axes[2].axhline(1.0, color="0.35", lw=1.0)
    axes[2].set_title("(c) Bulk and structural agreement")
    axes[2].set_ylabel("SISL/Split ratio or similarity")
    axes[2].legend(fontsize=9)

    for axis in axes:
        axis.set_xlabel("Simulation time (s)")
        axis.grid(True, ls="--", alpha=0.4)

    dx = float(sisl["dx"])
    dt = float(sisl["dt"])
    stabilization = float(sisl.get("stabilization", 0.0))
    stabilization_suffix = "" if stabilization == 0.0 else f"_nu{stabilization:g}"
    fig.suptitle(
        rf"Cross-core bubble evolution ($\Delta x={dx:g}$ m, "
        rf"$\Delta t={dt:g}$ s, stabilization factor $={stabilization:g}$)"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (f"rising_bubble_cross_core_evolution_dx{dx:g}_dt{dt:g}{stabilization_suffix}.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\n--- Cross-core evolution ---")
    print(" time[s]   rel_L2   cosine   centroid[m]   integral_ratio   peak_ratio")
    for values in zip(times, relative_l2, cosine, centroid_difference, integral_ratio, peak_ratio):
        print(
            f" {values[0]:7.1f}  {values[1]:7.4f}  {values[2]:7.4f}  "
            f"{values[3]:11.3f}  {values[4]:14.6f}  {values[5]:10.6f}"
        )
    print(f"Evolution figure saved to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dx", type=float, default=50.0)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--t-end", type=float, default=1000.0)
    parser.add_argument("--snapshot-interval", type=float, default=200.0)
    parser.add_argument(
        "--comparison-time", type=float, help="Snapshot time for the three-panel comparison (default: t_end)"
    )
    parser.add_argument(
        "--stabilization", type=float, default=0.0, help="Common nu_h_factor and nu_div_factor for both cores"
    )
    parser.add_argument("--reuse", action="store_true", help="Reuse matching NPZ files")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root containing benchmark, experiment, run, and verification bundles",
    )
    parser.add_argument("--name", default="default", help="Execution label")
    parser.add_argument(
        "--no-render", action="store_true", help="Only produce numerical artifacts; render them in a separate command"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Explicit bundle directory (legacy override of --output-root and --name)"
    )
    parser.add_argument("--worker-core", choices=("sisl", "split-explicit"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dt = args.dt if args.dt is not None else 2.5 * args.dx / 125.0

    if args.worker_core:
        if args.worker_output is None:
            raise ValueError("--worker-output is required in worker mode")
        run_core(
            args.worker_core, args.dx, dt, args.t_end, args.stabilization, args.snapshot_interval, args.worker_output
        )
        return

    if args.output_dir is None:
        layout = ArtifactLayout(
            kind="benchmarks", case="rising_bubble_3d", execution=args.name, output_root=args.output_root
        ).create()
        data_dir = layout.data
        figure_dir = layout.figures
    else:
        # Preserve the old flat --output-dir contract during the pilot.
        data_dir = args.output_dir
        figure_dir = args.output_dir

    paths = {
        core: result_path(data_dir, core, args.dx, dt, args.t_end, args.stabilization, args.snapshot_interval)
        for core in ("sisl", "split-explicit")
    }
    worker_environment = dict(
        os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false", MPLCONFIGDIR=str(Path("/tmp") / "suetes-matplotlib")
    )
    for core, path in paths.items():
        if args.reuse and path.exists():
            print(f"Reusing {path}")
            continue
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--worker-core",
            core,
            "--worker-output",
            str(path),
            "--dx",
            str(args.dx),
            "--dt",
            str(dt),
            "--t-end",
            str(args.t_end),
            "--stabilization",
            str(args.stabilization),
            "--snapshot-interval",
            str(args.snapshot_interval),
        ]
        subprocess.run(command, check=True, env=worker_environment)

    with np.load(paths["sisl"]) as source:
        sisl = {key: source[key] for key in source.files}
    with np.load(paths["split-explicit"]) as source:
        split = {key: source[key] for key in source.files}
    dataset = xr.Dataset(
        data_vars={
            "theta_perturbation": (
                ("core", "time", "x", "z"),
                np.stack([sisl["theta_prime_snapshots"], split["theta_prime_snapshots"]]),
            ),
            "mass_drift": ("core", [float(sisl["mass_error"]), float(split["mass_error"])]),
            "symmetry_error": ("core", [float(sisl["theta_symmetry_error"]), float(split["theta_symmetry_error"])]),
        },
        coords={"core": ["sisl", "split-explicit"], "time": sisl["snapshot_times"], "x": sisl["x"], "z": sisl["z"]},
        attrs={
            "dx_m": args.dx,
            "dt_s": dt,
            "t_end_s": args.t_end,
            "stabilization": args.stabilization,
            "snapshot_interval_s": args.snapshot_interval,
        },
    )
    artifact = save_plot_dataset(dataset, data_dir / "artifact.nc", experiment="rising_bubble_dual_core_3d")
    print(f"Saved plot-ready dual-core artifact to {artifact}")
    if not args.no_render:
        comparison_time = args.t_end if args.comparison_time is None else args.comparison_time
        plot_comparison(paths["sisl"], paths["split-explicit"], figure_dir, comparison_time)
        plot_evolution(paths["sisl"], paths["split-explicit"], figure_dir)


if __name__ == "__main__":
    main()
