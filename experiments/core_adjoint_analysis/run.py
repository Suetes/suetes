# Case runner.
"""
core_adjoint_analysis.py

Comprehensive qualitative and quantitative diagnostics for reverse-mode adjoint gradients
in the dual dynamical core (SISL vs. Split-Explicit).

Generates figures for the paper/appendix:
  - Adjoint Gradient Field Comparison (2D cross-sections & horizontal profiles):
    Showcases how SISL produces smooth, physically well-conditioned sensitivity fields
    compared to explicit acoustic ripples in Split-Explicit.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
from pathlib import Path
import time
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.steppers import build_dynamical_core
from suetes.shared.experiment import save_plot_dataset, add_experiment_args, setup_experiment_directories


def build_experiment_case(nx=96, ny=3, nz=32, dx=100.0):
    """Sets up grid, constants, and a warm bubble initial condition."""
    grid = RegionalGrid3D(nx, ny, nz, dx, dx, dx, lat_center=0.0, lon_center=0.0)
    op = CGridOperator3D(grid)
    constants = {"g": 9.81, "cp": 1004.0, "Rd": 287.0, "cvd": 717.0, "p0": 100000.0}

    tmp_phys = Euler3D(grid, op, constants, dt=1.0, N_bv=0.0)
    bg_ref = {
        "rho": tmp_phys.c["p0"]
        / (tmp_phys.c["Rd"] * tmp_phys.theta_bg)
        * (tmp_phys.pi_bg ** (tmp_phys.c["cvd"] / tmp_phys.c["Rd"])),
        "pi": tmp_phys.pi_bg,
        "th_v": tmp_phys.theta_bg,
    }

    state = {
        "u": jnp.zeros((nx + 1, ny, nz)),
        "v": jnp.zeros((nx, ny + 1, nz)),
        "w": jnp.zeros((nx, ny, nz + 1)),
        "pi": bg_ref["pi"],
        "eta_dot": jnp.zeros((nx, ny, nz + 1)),
        "rho": bg_ref["rho"],
    }

    X, Y, Z = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_m, indexing="ij")
    x_c, z_c = 0.0, 1500.0
    r = jnp.sqrt((X - x_c) ** 2 + (Z - z_c) ** 2)
    bubble = jnp.where(r <= 1000.0, 3.0 * jnp.cos(0.5 * jnp.pi * r / 1000.0) ** 2, 0.0)

    state["th_v"] = bg_ref["th_v"] + bubble
    state["rho"] = (
        constants["p0"] / (constants["Rd"] * state["th_v"]) * (bg_ref["pi"] ** (constants["cvd"] / constants["Rd"]))
    )

    return grid, op, constants, state


def compute_adjoint_gradient(core_type, grid, op, constants, initial_state, T_val=30.0, **core_kwargs):
    """
    Computes the reverse-mode adjoint gradient dL / d(th_v_0) for a target objective L.
    Objective L: Final updraft kinetic energy + temperature variance at time T.
    """
    dt = core_kwargs.get("dt", 1.0)
    num_steps = max(1, int(T_val / dt))
    use_checkpoint = core_type == "split-explicit"

    def forward_objective(st):
        stepper, actual_dt = build_dynamical_core(
            core_type=core_type, grid=grid, operators=op, constants=constants, initial_state=st, **core_kwargs
        )

        def scan_step(curr_st, step_idx):
            next_st = stepper.step(curr_st, step_idx * actual_dt, None, lambda x, f: x)
            return next_st, None

        if use_checkpoint and core_type == "split-explicit":
            scan_step = jax.checkpoint(scan_step)

        final_st, _ = jax.lax.scan(scan_step, st, jnp.arange(num_steps))

        # Objective L: Integrated kinetic energy of vertical updrafts + thermal perturbation
        w_c = 0.5 * (final_st["w"][:, :, :-1] + final_st["w"][:, :, 1:])
        th_pert = final_st["th_v"] - 300.0
        loss = 0.5 * jnp.sum(final_st["rho"] * w_c**2 + 0.1 * th_pert**2) * grid.dx * grid.dy * grid.dz
        return loss

    fn_grad = jax.jit(jax.grad(forward_objective))
    grad_st = fn_grad(initial_state)
    return grad_st["th_v"]


def run_gradient_field_comparison(data_dir: Path, figure_dir: Path, *, render: bool = True):
    print("\n==========================================================================")
    print("Experiment: Adjoint gradient field comparison (SISL vs. Split-Explicit)")
    print("==========================================================================")

    grid, op, constants, state = build_experiment_case(nx=96, ny=3, nz=32, dx=100.0)

    # 1. Split-Explicit (dt = 1.0s, ns = 12)
    print("  [1/2] Computing Split-Explicit reverse adjoint gradient...")
    t0 = time.perf_counter()
    grad_se = compute_adjoint_gradient(
        "split-explicit",
        grid,
        op,
        constants,
        state,
        T_val=30.0,
        dt=1.0,
        ns=12,
        damp_height=7500.0,
        max_damp=0.05,
        alpha=0.55,
    )
    grad_se.block_until_ready()
    print(f"        Done in {time.perf_counter() - t0:.2f}s")

    # 2. SISL (dt = 10.0s, 30 GMRES iterations)
    print("  [2/2] Computing SISL reverse adjoint gradient...")
    t0 = time.perf_counter()
    grad_sisl = compute_adjoint_gradient(
        "sisl",
        grid,
        op,
        constants,
        state,
        T_val=30.0,
        dt=10.0,
        solver_tol=1e-6,
        solver_maxiter=30,
        solver_restart=30,
        damp_height=7500.0,
        max_damp=0.05,
        alpha=0.55,
    )
    grad_sisl.block_until_ready()
    # Quantitative consistency metrics
    g_se_flat = np.array(grad_se).flatten()
    g_sisl_flat = np.array(grad_sisl).flatten()
    cos_sim = np.dot(g_se_flat, g_sisl_flat) / (np.linalg.norm(g_se_flat) * np.linalg.norm(g_sisl_flat) + 1e-15)
    pearson_corr = np.corrcoef(g_se_flat, g_sisl_flat)[0, 1]
    print(f"\n  [CONSISTENCY] Split-Explicit versus SISL adjoint gradients:")
    print(f"    - Cosine Similarity:     {cos_sim:.6f}")
    print(f"    - Pearson Correlation:   {pearson_corr:.6f}\n")

    # Extract central y-slice (y = 1)
    g_se_2d = np.array(grad_se[:, 1, :])
    g_sisl_2d = np.array(grad_sisl[:, 1, :])
    dataset = xr.Dataset(
        data_vars={"adjoint_gradient": (("core", "x", "z"), np.stack([g_se_2d, g_sisl_2d]))},
        coords={"core": ["split-explicit", "sisl"], "x": np.asarray(grid.x_m), "z": np.asarray(grid.z_m)},
        attrs={
            "cosine_similarity": float(cos_sim),
            "pearson_correlation": float(pearson_corr),
            "profile_height_m": 1500.0,
        },
    )
    artifact = save_plot_dataset(dataset, data_dir / "artifact.nc", experiment="core_adjoint_analysis")
    print(f"  [OUTPUT] Saved {artifact}")
    if not render:
        return

    # Plotting
    fig = plt.figure(figsize=(18, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.25)

    x_km = grid.x_m / 1000.0
    z_km = grid.z_m / 1000.0
    X, Z = np.meshgrid(x_km, z_km, indexing="ij")

    vmax = max(np.max(np.abs(g_se_2d)), np.max(np.abs(g_sisl_2d))) * 0.85

    # Panel (a): Split-Explicit
    ax0 = fig.add_subplot(gs[0])
    cf0 = ax0.pcolormesh(X, Z, g_se_2d, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="gouraud")
    ax0.set_title(
        r"(a) Split-Explicit adjoint sensitivity $\frac{\partial \mathcal{L}}{\partial \theta_{v,0}}$", fontsize=14
    )
    ax0.set_xlabel("Horizontal distance $x$ (km)", fontsize=11)
    ax0.set_ylabel("Altitude $z$ (km)", fontsize=11)

    # Panel (b): SISL
    ax1 = fig.add_subplot(gs[1])
    cf1 = ax1.pcolormesh(X, Z, g_sisl_2d, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="gouraud")
    ax1.set_title(r"(b) SISL adjoint sensitivity $\frac{\partial \mathcal{L}}{\partial \theta_{v,0}}$", fontsize=14)
    ax1.set_xlabel("Horizontal distance $x$ (km)", fontsize=11)
    ax1.set_ylabel("Altitude $z$ (km)", fontsize=11)

    # Panel (c): Horizontal profile cut at z = 1.5 km (center of bubble)
    z_idx = np.argmin(np.abs(grid.z_m - 1500.0))
    ax2 = fig.add_subplot(gs[2])
    ax2.plot(
        x_km,
        g_se_2d[:, z_idx],
        "o-",
        color="#e377c2",
        label=r"Split-Explicit ($\Delta t=1$ s)",
        linewidth=2,
        markersize=4,
        alpha=0.9,
    )
    ax2.plot(
        x_km, g_sisl_2d[:, z_idx], "s-", color="#1f77b4", label=r"SISL ($\Delta t=10$ s)", linewidth=2.5, markersize=4
    )
    ax2.set_title(f"(c) Sensitivity profile at $z = {grid.z_m[z_idx] / 1000.0:.1f}$ km", fontsize=14)
    ax2.set_xlabel("Horizontal distance $x$ (km)", fontsize=11)
    ax2.set_ylabel(r"$\frac{\partial \mathcal{L}}{\partial \theta_{v,0}}$ sensitivity value", fontsize=11)
    ax2.grid(True, ls="--", alpha=0.5)
    ax2.legend(fontsize=10.5)

    cbar = fig.colorbar(cf1, ax=[ax0, ax1], orientation="horizontal", fraction=0.06, pad=0.18)
    cbar.set_label(r"Adjoint gradient amplitude $\nabla_{\theta_0} \mathcal{L}$", fontsize=11)

    out_path = figure_dir / "adjoint_gradient_field_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SUCCESS] Saved gradient field comparison to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_args(parser)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    data_dir, figure_dir = setup_experiment_directories(args, kind="experiments", case="core_adjoint_analysis")
    run_gradient_field_comparison(data_dir, figure_dir, render=not args.no_render)
