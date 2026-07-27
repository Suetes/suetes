# Case runner.
import argparse
from pathlib import Path
import time
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.steppers import build_dynamical_core
from suetes.shared.driver import Simulation
from suetes.shared.artifacts import ArtifactLayout, save_plot_dataset


def compute_metrics(state, grid):
    # 1. Total Kinetic Energy: 0.5 * sum(rho * (u^2 + w^2)) * dV
    # Average u and w to cell centers
    u_c = 0.5 * (state["u"][:-1, :, :] + state["u"][1:, :, :])
    w_c = 0.5 * (state["w"][:, :, :-1] + state["w"][:, :, 1:])
    tke = float(0.5 * jnp.sum(state["rho"] * (u_c**2 + w_c**2)) * grid.dx * grid.dy * grid.dz)

    # 2. Total Enstrophy: sum(omega_y^2) * dV
    # Compute central differences on interior cell centers (nx-2, ny, nz-2)
    du_dz = (u_c[1:-1, :, 2:] - u_c[1:-1, :, :-2]) / (2.0 * grid.dz)
    dw_dx = (w_c[2:, :, 1:-1] - w_c[:-2, :, 1:-1]) / (2.0 * grid.dx)
    omega_y = du_dz - dw_dx
    enstrophy = float(jnp.sum(omega_y**2) * grid.dx * grid.dy * grid.dz)

    # 3. Max Updraft & Max Thermal Anomaly
    max_w = float(jnp.max(state["w"]))
    max_th = float(jnp.max(state["th_v"]))

    return tke, enstrophy, max_w, max_th


def compute_power_spectrum(field_2d, dx):
    """Computes 1D horizontal power spectral density of a 2D field (nx, nz)."""
    nx, nz = field_2d.shape
    # Take FFT along x for each z level
    fft_vals = np.fft.rfft(field_2d, axis=0)
    power = np.mean(np.abs(fft_vals) ** 2, axis=1)  # Average power across altitudes
    k = np.fft.rfftfreq(nx, d=dx) * 2.0 * np.pi
    return k[1:], power[1:]  # Exclude wavenumber 0 (mean)


def run_quantitative_study(core_type, dt, t_end=600.0, dx=50.0):
    print(f"\n=======================================================")
    print(f"[{core_type.upper()}] Running Quantitative Diagnostics (dt={dt}s)...")
    print(f"=======================================================")

    nx = int(10000.0 / dx)
    ny = 3
    nz = int(10000.0 / dx)

    grid = RegionalGrid3D(nx, ny, nz, dx, dx, dx, lat_center=0.0, lon_center=0.0)
    op = CGridOperator3D(grid)
    constants = {"g": 9.81, "cp": 1004.0, "Rd": 287.0, "cvd": 717.0, "p0": 100000.0}

    if core_type == "sisl":
        core_kwargs = {
            "dt": dt,
            "nu_div_factor": 0.0,
            "nu_h_factor": 0.0,
            "damp_height": 7500.0,
            "max_damp": 0.05,
            "N_bv": 0.0,
            "solver_tol": 1e-6,
            "solver_maxiter": 20,
            "solver_restart": 20,
            "alpha": 0.55,
        }
    else:
        ns = max(12, int(dt / 0.1))
        core_kwargs = {
            "dt": dt,
            "ns": ns,
            "nu_div_factor": 0.0,
            "nu_h_factor": 0.0,
            "damp_height": 7500.0,
            "max_damp": 0.05,
            "N_bv": 0.0,
            "alpha": 0.55,
        }

    tmp_phys = Euler3D(grid, op, constants, dt=dt, N_bv=0.0)
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
    r = jnp.sqrt(X**2 + (Z - 2000.0) ** 2)
    bubble = jnp.where(r <= 1500.0, 2.0 * jnp.cos(0.5 * jnp.pi * r / 1500.0) ** 2, 0.0)

    state["th_v"] = bg_ref["th_v"] + bubble
    state["rho"] = (
        constants["p0"] / (constants["Rd"] * state["th_v"]) * (bg_ref["pi"] ** (constants["cvd"] / constants["Rd"]))
    )

    stepper, actual_dt = build_dynamical_core(
        core_type=core_type, grid=grid, operators=op, constants=constants, initial_state=state, **core_kwargs
    )

    # Time integration with diagnostic snapshots every 100 seconds (for fast turnaround)
    dt_snapshot = 100.0
    chunk_steps = int(dt_snapshot / actual_dt)
    num_chunks = int(t_end / dt_snapshot)

    def step_fn(s, i):
        return stepper.step(s, i * actual_dt, None, lambda x, f: x), 0.0

    sim = Simulation(step_fn=step_fn, dt=actual_dt)

    times = [0.0]
    tke, enst, max_w, max_th = compute_metrics(state, grid)
    tkes = [tke]
    ensts = [enst]
    max_ws = [max_w]

    curr_state = state
    print(f"[{core_type.upper()}] Integrating to T={t_end}s in {num_chunks} diagnostic chunks...")
    for c in range(num_chunks):
        t_start = c * dt_snapshot
        t_end_chunk = (c + 1) * dt_snapshot
        curr_state = sim.run(curr_state, t_start=t_start, t_end=t_end_chunk, chunk_steps=chunk_steps)

        tke, enst, max_w, max_th = compute_metrics(curr_state, grid)
        times.append(t_end_chunk)
        tkes.append(tke)
        ensts.append(enst)
        max_ws.append(max_w)
        print(f"  T={t_end_chunk:3.0f}s | TKE={tke:.2e} | Enstrophy={enst:.2e} | Max W={max_w:.2f} m/s", flush=True)

    # Compute final horizontal power spectrum of vertical velocity w at y=1
    k_w, power_w = compute_power_spectrum(np.array(curr_state["w"][:-1, 1, :-1]), dx)

    return times, tkes, ensts, max_ws, k_w, power_w


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare dual-core quantitative diagnostics")
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--name", default="default")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        layout = ArtifactLayout(
            kind="experiments", case="core_quantitative_comparison", execution=args.name, output_root=args.output_root
        ).create()
        data_dir, figure_dir = layout.data, layout.figures
    else:
        data_dir = figure_dir = args.output_dir
        data_dir.mkdir(parents=True, exist_ok=True)

    t_end = 600.0
    dx = 50.0

    t_se, tke_se, enst_se, w_se, k_se, p_se = run_quantitative_study("split-explicit", dt=1.0, t_end=t_end, dx=dx)
    t_sisl, tke_sisl, enst_sisl, w_sisl, k_sisl, p_sisl = run_quantitative_study("sisl", dt=4.0, t_end=t_end, dx=dx)
    dataset = xr.Dataset(
        data_vars={
            "kinetic_energy": (("core", "time"), np.asarray([tke_se, tke_sisl])),
            "enstrophy": (("core", "time"), np.asarray([enst_se, enst_sisl])),
            "maximum_vertical_velocity": (("core", "time"), np.asarray([w_se, w_sisl])),
            "vertical_velocity_spectrum": (("core", "wavenumber"), np.asarray([p_se, p_sisl])),
        },
        coords={"core": ["split-explicit", "sisl"], "time": np.asarray(t_se), "wavenumber": np.asarray(k_se)},
        attrs={"dx_m": dx, "t_end_s": t_end},
    )
    artifact = save_plot_dataset(dataset, data_dir / "artifact.nc", experiment="core_quantitative_comparison")
    print(f"[OUTPUT] Saved {artifact}")
    if args.no_render:
        raise SystemExit(0)

    # --- PLOTTING QUANTITATIVE COMPARISON ---
    print("\n[PLOTTING] Generating quantitative diagnostic charts...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel (a): Enstrophy Evolution (Measure of rotational shear and turbulence)
    axes[0].plot(
        t_se, enst_se, "o-", color="#1f77b4", label=r"Split-Explicit ($\Delta t = 1.0$s)", linewidth=2, markersize=4
    )
    axes[0].plot(t_sisl, enst_sisl, "s-", color="#ff7f0e", label=r"SISL ($\Delta t = 4.0$s)", linewidth=2, markersize=4)
    axes[0].set_title("(a) Total Enstrophy Evolution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Time (s)", fontsize=12)
    axes[0].set_ylabel(r"Enstrophy $\int \omega_y^2 \, dV$ ($\text{m}^3/\text{s}^2$)", fontsize=12)
    axes[0].grid(True, which="both", ls="--", alpha=0.5)
    axes[0].legend(fontsize=10)

    # Panel (b): Max Vertical Velocity Updraft
    axes[1].plot(t_se, w_se, "o-", color="#1f77b4", label="Split-Explicit", linewidth=2, markersize=4)
    axes[1].plot(t_sisl, w_sisl, "s-", color="#ff7f0e", label="SISL", linewidth=2, markersize=4)
    axes[1].set_title(r"(b) Peak Updraft Velocity ($w_{\max}$)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Time (s)", fontsize=12)
    axes[1].set_ylabel("Vertical Velocity (m/s)", fontsize=12)
    axes[1].grid(True, which="both", ls="--", alpha=0.5)
    axes[1].legend(fontsize=10)

    # Panel (c): Horizontal Power Spectrum at T=600s
    axes[2].loglog(k_se, p_se, "-", color="#1f77b4", label="Split-Explicit", linewidth=2)
    axes[2].loglog(k_sisl, p_sisl, "-", color="#ff7f0e", label="SISL", linewidth=2)

    # Add reference k^-5/3 inertial subrange line
    k_ref = k_se[len(k_se) // 4 : len(k_se) // 2]
    p_ref = p_se[len(k_se) // 4] * (k_ref / k_ref[0]) ** (-5 / 3)
    axes[2].loglog(k_ref, p_ref, "k--", label="$k^{-5/3}$ Reference", alpha=0.7)

    axes[2].set_title("(c) Vertical Velocity Power Spectrum $E(k)$", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Horizontal Wavenumber $k$ (rad/m)", fontsize=12)
    axes[2].set_ylabel("Spectral Density ($w^2$ power)", fontsize=12)
    axes[2].grid(True, which="both", ls="--", alpha=0.5)
    axes[2].legend(fontsize=10)

    out_path = figure_dir / "dual_core_quantitative_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved quantitative diagnostic plot to: {out_path}\n")
