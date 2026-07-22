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
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from suetes.regional3d.euler import Euler3D
from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.steppers import build_dynamical_core
from suetes.shared.driver import Simulation

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "output" / "benchmark_data"
DEFAULT_PLOT_DIR = REPO_ROOT / "output" / "plots" / "benchmarks"
DOMAIN_WIDTH = 10_000.0
DOMAIN_HEIGHT = 10_000.0
CONSTANTS = {
    "g": 9.81, "cp": 1004.0, "Rd": 287.0,
    "cvd": 717.0, "p0": 100_000.0,
}


def result_path(
    data_dir: Path,
    core: str,
    dx: float,
    dt: float,
    t_end: float,
    stabilization: float,
) -> Path:
    core_name = core.replace("-", "_")
    stabilization_suffix = (
        "" if stabilization == 0.0 else f"_nu{stabilization:g}"
    )
    return data_dir / (
        f"rising_bubble_{core_name}_dx{dx:g}_dt{dt:g}_t{t_end:g}"
        f"{stabilization_suffix}.npz"
    )


def run_core(
    core: str,
    dx: float,
    dt: float,
    t_end: float,
    stabilization: float,
    output: Path,
) -> None:

    nx = round(DOMAIN_WIDTH / dx)
    nz = round(DOMAIN_HEIGHT / dx)
    ny = 3
    if not np.isclose(nx * dx, DOMAIN_WIDTH) or not np.isclose(
        nz * dx, DOMAIN_HEIGHT
    ):
        raise ValueError("dx must divide the 10 km domain exactly")
    num_steps = round(t_end / dt)
    if not np.isclose(num_steps * dt, t_end):
        raise ValueError("t_end must be an integer multiple of dt")

    grid = RegionalGrid3D(
        nx, ny, nz, dx, dx, dx, lat_center=0.0, lon_center=0.0
    )
    operators = CGridOperator3D(grid)
    background_physics = Euler3D(
        grid, operators, CONSTANTS, dt=dt, N_bv=0.0
    )
    theta_bg = background_physics.theta_bg
    pi_bg = background_physics.pi_bg
    rho_bg = (
        CONSTANTS["p0"] / (CONSTANTS["Rd"] * theta_bg)
        * pi_bg ** (CONSTANTS["cvd"] / CONSTANTS["Rd"])
    )

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
    theta_prime = jnp.where(
        radius <= 1500.0,
        2.0 * jnp.cos(0.5 * jnp.pi * radius / 1500.0) ** 2,
        0.0,
    )
    state["th_v"] = theta_bg + theta_prime
    state["rho"] = (
        CONSTANTS["p0"] / (CONSTANTS["Rd"] * state["th_v"])
        * pi_bg ** (CONSTANTS["cvd"] / CONSTANTS["Rd"])
    )

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
        core_kwargs = {
            **common_kwargs,
            "solver_tol": 1.0e-8,
            "solver_maxiter": 30,
            "solver_restart": 30,
        }
    elif core == "split-explicit":
        core_kwargs = {**common_kwargs, "ns": 12}
    else:
        raise ValueError(f"Unknown core: {core}")

    stepper, actual_dt = build_dynamical_core(
        core_type=core,
        grid=grid,
        operators=operators,
        constants=CONSTANTS,
        initial_state=state,
        **core_kwargs,
    )

    def step_fn(current_state, step_index):
        next_state = stepper.step(
            current_state,
            step_index * actual_dt,
            forcing=None,
            bc_fn=lambda value, field: value,
        )
        return next_state, jnp.max(jnp.abs(next_state["w"]))

    simulation = Simulation(step_fn=step_fn, dt=actual_dt)
    chunk_steps = max(1, round(100.0 / actual_dt))
    print(
        f"\n[BENCHMARK] {core.upper()}: {nx}x{ny}x{nz}, "
        f"dx={dx:g} m, dt={actual_dt:g} s, steps={num_steps}"
    )
    final_state = simulation.run(
        state, t_start=0.0, t_end=t_end, chunk_steps=chunk_steps
    )

    y_mid = ny // 2
    final_theta_prime = final_state["th_v"][:, y_mid, :] - theta_bg[:, y_mid, :]
    cell_volumes = (
        grid.dx * grid.dy * (grid.Z_w[:, :, 1:] - grid.Z_w[:, :, :-1])
        / grid.m_factors["m"][..., None] ** 2
    )
    initial_mass = jnp.sum(state["rho"] * cell_volumes, dtype=jnp.float64)
    final_mass = jnp.sum(final_state["rho"] * cell_volumes, dtype=jnp.float64)
    mass_error = jnp.abs(final_mass - initial_mass) / initial_mass
    theta_symmetry_error = jnp.max(
        jnp.abs(final_theta_prime - final_theta_prime[::-1, :])
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        theta_prime=np.asarray(final_theta_prime),
        x=np.asarray(grid.x_m),
        z=np.asarray(grid.z_m),
        mass_error=float(mass_error),
        theta_symmetry_error=float(theta_symmetry_error),
        dx=dx,
        dt=actual_dt,
        t_end=t_end,
        stabilization=stabilization,
    )
    print(f"[BENCHMARK] Saved {core} result to {output}")


def plot_comparison(sisl_path: Path, split_path: Path, output_dir: Path,) -> Path:

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

    theta_sisl = sisl["theta_prime"]
    theta_split = split["theta_prime"]
    difference = theta_sisl - theta_split
    reference_norm = np.linalg.norm(0.5 * (theta_sisl + theta_split))
    relative_l2 = np.linalg.norm(difference) / max(reference_norm, np.finfo(float).tiny)
    max_difference = float(np.max(np.abs(difference)))
    cosine_similarity = float(
        np.vdot(theta_sisl, theta_split)
        / (np.linalg.norm(theta_sisl) * np.linalg.norm(theta_split))
    )

    def anomaly_moments(field: np.ndarray) -> tuple[float, float, float, float]:
        positive = np.maximum(field, 0.0)
        anomaly_sum = float(np.sum(positive))
        centroid = float(
            np.sum(positive * sisl["z"][None, :]) / anomaly_sum
        )
        vertical_spread = float(
            np.sqrt(
                np.sum(positive * (sisl["z"][None, :] - centroid) ** 2)
                / anomaly_sum
            )
        )
        return anomaly_sum, centroid, vertical_spread, float(np.max(field))

    sisl_moments = anomaly_moments(theta_sisl)
    split_moments = anomaly_moments(theta_split)

    x_km = sisl["x"] / 1000.0
    z_km = sisl["z"] / 1000.0
    main_limit = float(max(np.max(np.abs(theta_sisl)), np.max(np.abs(theta_split))))
    difference_limit = max(max_difference, np.finfo(float).eps)

    fig, axes = plt.subplots(
        1, 3, figsize=(18, 5.4), sharex=True, sharey=True,
        constrained_layout=True,
    )
    main_images = []
    for axis, field, title in zip(
        axes[:2],
        (theta_sisl, theta_split),
        ("(a) SISL", "(b) Split-Explicit"),
    ):
        image = axis.pcolormesh(
            x_km, z_km, field.T,
            cmap="RdBu_r", vmin=-main_limit, vmax=main_limit,
            shading="auto", rasterized=True,
        )
        main_images.append(image)
        axis.set_title(title, fontsize=14)

    difference_image = axes[2].pcolormesh(
        x_km, z_km, difference.T,
        cmap="coolwarm", vmin=-difference_limit, vmax=difference_limit,
        shading="auto", rasterized=True,
    )
    axes[2].set_title("(c) SISL $-$ Split-Explicit", fontsize=14)
    axes[2].text(
        0.03, 0.97,
        rf"relative $L_2={relative_l2:.2e}$" + "\n"
        + rf"$L_\infty={max_difference:.2e}$ K",
        transform=axes[2].transAxes,
        ha="left", va="top", fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
    )

    for axis in axes:
        axis.set_xlabel("Horizontal distance (km)", fontsize=11)
        axis.set_aspect("equal")
        axis.set_xlim(x_km[0], x_km[-1])
        axis.set_ylim(z_km[0], z_km[-1])
    axes[0].set_ylabel("Height (km)", fontsize=11)

    fig.colorbar(
        main_images[0], ax=axes[:2], shrink=0.91,
        label=r"Virtual potential-temperature perturbation $\theta_v'$ (K)",
    )
    fig.colorbar(
        difference_image, ax=axes[2], shrink=0.91,
        label=r"Difference in $\theta_v'$ (K)",
    )

    dx = float(sisl["dx"])
    dt = float(sisl["dt"])
    t_end = float(sisl["t_end"])
    stabilization = float(sisl["stabilization"])
    stabilization_text = (
        "no explicit stabilization"
        if stabilization == 0.0
        else rf"stabilization factor $={stabilization:g}$"
    )
    fig.suptitle(
        rf"Rising thermal bubble at $t={t_end:g}$ s "
        rf"($\Delta x={dx:g}$ m, $\Delta t={dt:g}$ s; "
        + stabilization_text + ")",
        fontsize=14,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stabilization_suffix = (
        "" if stabilization == 0.0 else f"_nu{stabilization:g}"
    )
    output_path = output_dir / (
        f"rising_bubble_dual_core_3d_{dx:g}m_{t_end:g}s"
        f"{stabilization_suffix}.png"
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\n--- Dual-core comparison ---")
    print(f"Relative L2 difference : {relative_l2:.6e}")
    print(f"Maximum point difference: {max_difference:.6e} K")
    print(f"Cosine similarity      : {cosine_similarity:.6e}")
    print(
        "Positive-anomaly integral ratio (SISL/Split): "
        f"{sisl_moments[0] / split_moments[0]:.6e}"
    )
    print(
        "Anomaly centroid difference (SISL-Split)    : "
        f"{sisl_moments[1] - split_moments[1]:.3f} m"
    )
    print(
        "Vertical-spread difference (SISL-Split)     : "
        f"{sisl_moments[2] - split_moments[2]:.3f} m"
    )
    print(
        "Peak-amplitude ratio (SISL/Split)           : "
        f"{sisl_moments[3] / split_moments[3]:.6e}"
    )
    for core, result in (("SISL", sisl), ("Split-Explicit", split)):
        print(
            f"{core:14s} | mass drift={float(result['mass_error']):.3e} "
            f"| theta symmetry error={float(result['theta_symmetry_error']):.3e} K"
        )
    print(f"Figure saved to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dx", type=float, default=50.0)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--t-end", type=float, default=1000.0)
    parser.add_argument(
        "--stabilization", type=float, default=0.0,
        help="Common nu_h_factor and nu_div_factor for both cores",
    )
    parser.add_argument("--reuse", action="store_true", help="Reuse matching NPZ files")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOT_DIR)
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
            args.worker_core, args.dx, dt, args.t_end,
            args.stabilization, args.worker_output,
        )
        return

    paths = {
        core: result_path(
            args.data_dir, core, args.dx, dt, args.t_end,
            args.stabilization,
        )
        for core in ("sisl", "split-explicit")
    }
    worker_environment = dict(
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
            "--worker-core", core,
            "--worker-output", str(path),
            "--dx", str(args.dx),
            "--dt", str(dt),
            "--t-end", str(args.t_end),
            "--stabilization", str(args.stabilization),
        ]
        subprocess.run(command, check=True, env=worker_environment)

    plot_comparison(paths["sisl"], paths["split-explicit"], args.output_dir)


if __name__ == "__main__":
    main()
