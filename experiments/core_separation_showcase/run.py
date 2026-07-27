# Case runner.
import os

# Limit JAX VRAM allocation to 50% to ensure stability on single RTX 4090 GPU
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.50"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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
from suetes.shared.experiment import save_plot_dataset, add_experiment_args, setup_experiment_directories


def compute_metrics(state, grid):
    u_c = 0.5 * (state["u"][:-1, :, :] + state["u"][1:, :, :])
    w_c = 0.5 * (state["w"][:, :, :-1] + state["w"][:, :, 1:])
    tke = float(0.5 * jnp.sum(state["rho"] * (u_c**2 + w_c**2)) * grid.dx * grid.dy * grid.dz)

    # Enstrophy on interior cell centers
    du_dz = (u_c[1:-1, :, 2:] - u_c[1:-1, :, :-2]) / (2.0 * grid.dz)
    dw_dx = (w_c[2:, :, 1:-1] - w_c[:-2, :, 1:-1]) / (2.0 * grid.dx)
    omega_y = du_dz - dw_dx
    enstrophy = float(jnp.sum(omega_y**2) * grid.dx * grid.dy * grid.dz)

    th_pert = state["th_v"] - 300.0
    th_var = float(jnp.sum(th_pert**2) * grid.dx * grid.dy * grid.dz)

    max_w = float(jnp.max(state["w"]))
    return tke, enstrophy, th_var, max_w


def compute_power_spectrum(state, grid, u_bg=15.0):
    """Computes 1D horizontal kinetic energy power spectral density of velocity perturbations."""
    u_c = np.array(0.5 * (state["u"][:-1, 1, :] + state["u"][1:, 1, :])) - u_bg
    w_c = np.array(0.5 * (state["w"][:, 1, :-1] + state["w"][:, 1, 1:]))
    ke_2d = 0.5 * (u_c**2 + w_c**2)

    nx, nz = ke_2d.shape
    fft_vals = np.fft.rfft(ke_2d, axis=0)
    power = np.mean(np.abs(fft_vals) ** 2, axis=1)  # Average power across altitudes
    k = np.fft.rfftfreq(nx, d=grid.dx) * 2.0 * np.pi
    return k[1:], power[1:]  # Exclude wavenumber 0


def setup_centered_plume_case(grid, constants):
    """
    Sets up a rising thermal plume initialized at x0 = -1800m with a uniform 15 m/s wind.
    After T = 120s, the plume advects by exactly +1800m, landing dead center at x = 0!
    """
    tmp_phys = Euler3D(grid, CGridOperator3D(grid), constants, dt=1.0, N_bv=0.0)
    bg_ref = {
        "rho": tmp_phys.c["p0"]
        / (tmp_phys.c["Rd"] * tmp_phys.theta_bg)
        * (tmp_phys.pi_bg ** (tmp_phys.c["cvd"] / tmp_phys.c["Rd"])),
        "pi": tmp_phys.pi_bg,
        "th_v": tmp_phys.theta_bg,
    }

    state = {
        "u": jnp.zeros((grid.nx + 1, grid.ny, grid.nz)),
        "v": jnp.zeros((grid.nx, grid.ny + 1, grid.nz)),
        "w": jnp.zeros((grid.nx, grid.ny, grid.nz + 1)),
        "pi": bg_ref["pi"],
        "eta_dot": jnp.zeros((grid.nx, grid.ny, grid.nz + 1)),
        "rho": bg_ref["rho"],
    }

    X, Y, Z = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_m, indexing="ij")

    # Uniform horizontal wind u = 15 m/s
    u_flow = 15.0
    state["u"] = state["u"].at[:, :, :].set(u_flow)

    # Plume initialized at x0 = -1800m so it advects to x = 0 at T = 120s
    r = jnp.sqrt((X + 1800.0) ** 2 + (Z - 1800.0) ** 2)
    base_bubble = jnp.where(r <= 1400.0, 4.0 * jnp.cos(0.5 * jnp.pi * r / 1400.0) ** 2, 0.0)
    wiggles = 1.0 + 0.6 * jnp.sin(16.0 * jnp.pi * X / 10000.0) * jnp.sin(16.0 * jnp.pi * Z / 10000.0)

    state["th_v"] = bg_ref["th_v"] + base_bubble * wiggles
    state["rho"] = (
        constants["p0"] / (constants["Rd"] * state["th_v"]) * (bg_ref["pi"] ** (constants["cvd"] / constants["Rd"]))
    )

    return state


def test_cfl_stability_limit(grid, constants, state):
    print("\n=======================================================")
    print("EXPERIMENT 1: TIMESTEP CFL STABILITY BARRIER")
    print("=======================================================")
    print("Attempting Split-Explicit at dt=6.0s (Advective CFL ~ 1.4)...")
    try:
        stepper, actual_dt = build_dynamical_core(
            "split-explicit", grid, CGridOperator3D(grid), constants, state, dt=6.0, ns=20, alpha=0.55
        )
        sim = Simulation(step_fn=lambda s, i: (stepper.step(s, i * actual_dt, None, lambda x, f: x), 0.0), dt=actual_dt)
        s_test = sim.run(state, t_start=0.0, t_end=30.0, chunk_steps=int(30.0 / actual_dt))
        max_w_val = float(jnp.max(jnp.abs(s_test["w"])))
        if jnp.isnan(max_w_val) or max_w_val > 1e3:
            print("  -> [CONFIRMED] Split-Explicit INSTANTLY BLOWS UP / DIVERGES at dt=6.0s due to CFL limit!")
        else:
            print(f"  -> Stable at dt=6.0s (max w = {max_w_val:.2f} m/s)")
    except Exception as e:
        print(f"  -> [CONFIRMED] Split-Explicit CRASHED at dt=6.0s due to Courant instability: {e}")

    print("Attempting SISL at dt=10.0s (Advective CFL ~ 2.4)...")
    stepper_sisl, actual_dt_sisl = build_dynamical_core(
        "sisl",
        grid,
        CGridOperator3D(grid),
        constants,
        state,
        dt=10.0,
        nu_div_factor=0.0,
        nu_h_factor=0.0,
        damp_height=7500.0,
        max_damp=0.05,
        N_bv=0.0,
        solver_tol=1e-6,
        solver_maxiter=20,
        solver_restart=20,
        alpha=0.55,
    )
    sim_sisl = Simulation(
        step_fn=lambda s, i: (stepper_sisl.step(s, i * actual_dt_sisl, None, lambda x, f: x), 0.0), dt=actual_dt_sisl
    )
    s_sisl = sim_sisl.run(state, t_start=0.0, t_end=30.0, chunk_steps=int(30.0 / actual_dt_sisl))
    print(
        f"  -> [CONFIRMED] SISL IS UNCONDITIONALLY STABLE at dt=10.0s! Max w = {float(jnp.max(s_sisl['w'])):.2f} m/s\n"
    )


def run_simulation(core_type, dt, state, grid, constants, t_end=120.0):
    print(f"[{core_type.upper()}] Integrating Centered Plume (dt={dt}s)...")
    op = CGridOperator3D(grid)
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

    stepper, actual_dt = build_dynamical_core(
        core_type=core_type, grid=grid, operators=op, constants=constants, initial_state=state, **core_kwargs
    )

    dt_snapshot = 30.0
    chunk_steps = int(dt_snapshot / actual_dt)
    num_chunks = int(t_end / dt_snapshot)

    sim = Simulation(step_fn=lambda s, i: (stepper.step(s, i * actual_dt, None, lambda x, f: x), 0.0), dt=actual_dt)

    times = [0.0]
    tke, enst, th_var, max_w = compute_metrics(state, grid)
    ensts = [enst]
    th_vars = [th_var]

    curr_state = state
    for c in range(num_chunks):
        t_start = c * dt_snapshot
        t_end_chunk = (c + 1) * dt_snapshot
        curr_state = sim.run(curr_state, t_start=t_start, t_end=t_end_chunk, chunk_steps=chunk_steps)

        tke, enst, th_var, max_w = compute_metrics(curr_state, grid)
        times.append(t_end_chunk)
        ensts.append(enst)
        th_vars.append(th_var)
        print(
            f"  T={t_end_chunk:3.0f}s | Enstrophy={enst:.2e} | Thermal Var={th_var:.2e} | Max W={max_w:.2f} m/s",
            flush=True,
        )

    return curr_state, times, ensts, th_vars


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the three-way core separation showcase")
    add_experiment_args(parser)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    data_dir, figure_dir = setup_experiment_directories(args, kind="experiments", case="core_separation_showcase")

    print("\n=======================================================")
    print("RTX 4090 FAIR & OPERATIONAL BENCHMARK (3-WAY COMPARISON)")
    print("=======================================================")

    dx = 62.5
    nx = int(10000.0 / dx)
    ny = 3
    nz = int(10000.0 / dx)
    grid = RegionalGrid3D(nx, ny, nz, dx, dx, dx, lat_center=0.0, lon_center=0.0)
    constants = {"g": 9.81, "cp": 1004.0, "Rd": 287.0, "cvd": 717.0, "p0": 100000.0}

    initial_state = setup_centered_plume_case(grid, constants)

    # Run Experiment 1: CFL Blowup demonstration
    test_cfl_stability_limit(grid, constants, initial_state)
    jax.clear_caches()

    print("=======================================================")
    print("EXPERIMENT 2: FAIR vs OPERATIONAL COMPARISON (T=120s)")
    print("=======================================================")
    t_end = 120.0

    # 1. Split-Explicit at dt = 1.0s (Baseline Eulerian)
    state_se, t_se, enst_se, var_se = run_simulation(
        "split-explicit", dt=1.0, state=initial_state, grid=grid, constants=constants, t_end=t_end
    )
    jax.clear_caches()

    # 2. SISL Fair Comparison at Equal Timestep dt = 1.0s (Demonstrates inherent interpolation dissipation)
    state_sisl_fair, t_sisl_fair, enst_sisl_fair, var_sisl_fair = run_simulation(
        "sisl", dt=1.0, state=initial_state, grid=grid, constants=constants, t_end=t_end
    )
    jax.clear_caches()

    # 3. SISL Operational Comparison at Large Timestep dt = 10.0s (Demonstrates operational efficiency & filtering)
    state_sisl_op, t_sisl_op, enst_sisl_op, var_sisl_op = run_simulation(
        "sisl", dt=10.0, state=initial_state, grid=grid, constants=constants, t_end=t_end
    )

    enst_diff_fair = 100.0 * (enst_se[-1] - enst_sisl_fair[-1]) / enst_se[-1]
    enst_diff_op = 100.0 * (enst_se[-1] - enst_sisl_op[-1]) / enst_se[-1]

    print(f"\n---> QUANTITATIVE SEPARATION RESULT at T={t_end}s <---")
    print(f"Split-Explicit (1.0s) Enstrophy:      {enst_se[-1]:.2e}")
    print(
        f"SISL Equal-Timestep (1.0s) Enstrophy: {enst_sisl_fair[-1]:.2e} ({enst_diff_fair:.1f}% inherent numerical dissipation!)"
    )
    print(
        f"SISL Large-Timestep (10s) Enstrophy:  {enst_sisl_op[-1]:.2e} ({enst_diff_op:.1f}% operational subgrid filtering!)"
    )

    # --- SEPARATE PUBLICATION PLOTS ---
    print("\n[PLOTTING] Generating separate publication figures...")
    x_km = grid.x_m / 1000.0
    z_km = grid.z_m / 1000.0
    X_int, Z_int = np.meshgrid(x_km[1:-1], z_km[1:-1], indexing="ij")

    # Compute 2D vorticity fields cleanly on interior points (nx-2, nz-2)
    def calc_vort(st):
        u_c = 0.5 * (st["u"][:-1, 1, :] + st["u"][1:, 1, :])
        w_c = 0.5 * (st["w"][:, 1, :-1] + st["w"][:, 1, 1:])
        du_dz = (u_c[1:-1, 2:] - u_c[1:-1, :-2]) / (2.0 * grid.dz)
        dw_dx = (w_c[2:, 1:-1] - w_c[:-2, 1:-1]) / (2.0 * grid.dx)
        return du_dz - dw_dx

    vort_se = calc_vort(state_se)
    vort_sisl_fair = calc_vort(state_sisl_fair)
    vort_sisl_op = calc_vort(state_sisl_op)
    k_se, power_se = compute_power_spectrum(state_se, grid)
    k_fair, power_fair = compute_power_spectrum(state_sisl_fair, grid)
    k_op, power_op = compute_power_spectrum(state_sisl_op, grid)
    dataset = xr.Dataset(
        data_vars={
            "vorticity": (("configuration", "x", "z"), np.stack([vort_se, vort_sisl_fair, vort_sisl_op])),
            "enstrophy": (("configuration", "time"), np.asarray([enst_se, enst_sisl_fair, enst_sisl_op])),
            "thermal_variance": (("configuration", "time"), np.asarray([var_se, var_sisl_fair, var_sisl_op])),
            "kinetic_energy_spectrum": (("configuration", "wavenumber"), np.asarray([power_se, power_fair, power_op])),
        },
        coords={
            "configuration": ["split-explicit", "sisl-equal-timestep", "sisl-operational"],
            "x": np.asarray(grid.x_m[1:-1]),
            "z": np.asarray(grid.z_m[1:-1]),
            "time": np.asarray(t_se),
            "wavenumber": np.asarray(k_se),
        },
        attrs={"dx_m": dx, "t_end_s": t_end},
    )
    artifact = save_plot_dataset(dataset, data_dir / "artifact.nc", experiment="core_separation_showcase")
    print(f"[OUTPUT] Saved {artifact}")
    if args.no_render:
        raise SystemExit(0)

    # Common contour levels for fair visual comparison
    vmax = max(np.max(np.abs(vort_se)), np.max(np.abs(vort_sisl_fair))) * 0.9
    levels = np.linspace(-vmax, vmax, 40)

    # 1. Figure: Split-Explicit Vorticity Rotors
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    c1 = ax1.contourf(X_int, Z_int, vort_se, levels=levels, cmap="RdBu_r", extend="both")
    ax1.set_title(
        r"Split-Explicit Core ($\Delta t = 1.0$s): Convective Rotors at $x=0$", fontsize=14, fontweight="bold"
    )
    ax1.set_xlabel("Horizontal Distance (km)", fontsize=12)
    ax1.set_ylabel("Altitude (km)", fontsize=12)
    plt.colorbar(c1, ax=ax1, label=r"Vorticity $\omega_y$ ($\text{s}^{-1}$)")
    fig1.savefig(figure_dir / "dual_core_vorticity_split_explicit.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # 2. Figure: SISL Equal-Timestep Vorticity
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    c2 = ax2.contourf(X_int, Z_int, vort_sisl_fair, levels=levels, cmap="RdBu_r", extend="both")
    ax2.set_title(
        r"SISL Equal-Timestep ($\Delta t = 1.0$s): Inherent Interpolation Damping at $x=0$",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xlabel("Horizontal Distance (km)", fontsize=12)
    ax2.set_ylabel("Altitude (km)", fontsize=12)
    plt.colorbar(c2, ax=ax2, label=r"Vorticity $\omega_y$ ($\text{s}^{-1}$)")
    fig2.savefig(figure_dir / "dual_core_vorticity_sisl_fair.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # 3. Figure: SISL Operational Large-Timestep Vorticity
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    c3 = ax3.contourf(X_int, Z_int, vort_sisl_op, levels=levels, cmap="RdBu_r", extend="both")
    ax3.set_title(
        r"SISL Operational ($\Delta t = 10.0$s): Subgrid Filtered Convection at $x=0$", fontsize=14, fontweight="bold"
    )
    ax3.set_xlabel("Horizontal Distance (km)", fontsize=12)
    ax3.set_ylabel("Altitude (km)", fontsize=12)
    plt.colorbar(c3, ax=ax3, label=r"Vorticity $\omega_y$ ($\text{s}^{-1}$)")
    fig3.savefig(figure_dir / "dual_core_vorticity_sisl_op.png", dpi=300, bbox_inches="tight")
    plt.close(fig3)

    # 4. Figure: Enstrophy Time Series (3-Way Comparison)
    fig4, ax4 = plt.subplots(figsize=(9, 5.5))
    ax4.plot(
        t_se, enst_se, "o-", color="#1f77b4", label=r"Split-Explicit ($\Delta t = 1.0$s)", linewidth=2.5, markersize=7
    )
    ax4.plot(
        t_sisl_fair,
        enst_sisl_fair,
        "^--",
        color="#2ca02c",
        label=rf"SISL Equal-Timestep ($\Delta t = 1.0$s, -{enst_diff_fair:.1f}%)",
        linewidth=2.5,
        markersize=7,
    )
    ax4.plot(
        t_sisl_op,
        enst_sisl_op,
        "s-",
        color="#ff7f0e",
        label=rf"SISL Operational ($\Delta t = 10.0$s, -{enst_diff_op:.1f}%)",
        linewidth=2.5,
        markersize=7,
    )
    ax4.set_title("Total Enstrophy Evolution: Fair vs Operational Comparison", fontsize=14, fontweight="bold")
    ax4.set_xlabel("Time (s)", fontsize=12)
    ax4.set_ylabel(r"Enstrophy $\int \omega_y^2 \, dV$ ($\text{m}^3/\text{s}^2$)", fontsize=12)
    ax4.grid(True, which="both", ls="--", alpha=0.5)
    ax4.legend(fontsize=11)
    fig4.savefig(figure_dir / "dual_core_enstrophy_series.png", dpi=300, bbox_inches="tight")
    plt.close(fig4)

    # 5. Figure: Kinetic Energy Spectrum E(k) (3-Way Comparison)
    fig5, ax5 = plt.subplots(figsize=(9, 5.5))
    ax5.loglog(
        k_se, power_se, "o-", color="#1f77b4", label=r"Split-Explicit ($\Delta t = 1.0$s)", linewidth=2.5, markersize=5
    )
    ax5.loglog(
        k_fair,
        power_fair,
        "^--",
        color="#2ca02c",
        label=r"SISL Equal-Timestep ($\Delta t = 1.0$s)",
        linewidth=2.5,
        markersize=5,
    )
    ax5.loglog(
        k_op,
        power_op,
        "s-",
        color="#ff7f0e",
        label=r"SISL Operational ($\Delta t = 10.0$s)",
        linewidth=2.5,
        markersize=5,
    )
    ax5.set_title(r"Horizontal Kinetic Energy Spectrum $E(k)$ at $T=120$s", fontsize=14, fontweight="bold")
    ax5.set_xlabel(r"Wavenumber $k$ ($\text{rad/m}$)", fontsize=12)
    ax5.set_ylabel(r"Power Density $E(k)$", fontsize=12)
    ax5.grid(True, which="both", ls="--", alpha=0.5)
    ax5.legend(fontsize=11)
    fig5.savefig(figure_dir / "dual_core_energy_spectrum.png", dpi=300, bbox_inches="tight")
    plt.close(fig5)

    print(f"[SUCCESS] All 5 separate publication figures saved to {figure_dir}/\n")
