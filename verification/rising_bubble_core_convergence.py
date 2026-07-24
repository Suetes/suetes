import os
import gc
import argparse
import json
from pathlib import Path
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import jax
import jax.numpy as jnp

# Enable X64 for precise convergence tests
jax.config.update("jax_enable_x64", True)

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.steppers import build_dynamical_core
from suetes.shared.artifacts import ArtifactLayout
from suetes.shared.driver import Simulation

RESOLUTIONS = [250.0, 125.0, 62.5, 31.25]
DT_AT_250M = 1.0
SPLIT_EXPLICIT_NS = 12
NUM_PROGRESS_UPDATES = 6


def release_jax_memory():
    jax.clear_caches()
    gc.collect()


def block_average_2d(field, factor=2):
    nx, nz = field.shape
    return field.reshape(
        nx // factor, factor, nz // factor, factor
    ).mean(axis=(1, 3))


def restrict_to_coarse(field, key, factor=2):
    """Restrict a fine C-grid slice without destroying its native staggering."""
    if key in ('pi', 'th_v'):
        return block_average_2d(field, factor)
    if key == 'u':
        # Coarse x-faces coincide with every factor-th fine x-face; average only
        # across the transverse (z) fine cells.
        coincident_faces = field[::factor, :]
        nz = coincident_faces.shape[1]
        return coincident_faces.reshape(
            coincident_faces.shape[0], nz // factor, factor
        ).mean(axis=2)
    if key == 'w':
        # Coarse z-faces coincide vertically; average only across fine x cells.
        coincident_faces = field[:, ::factor]
        nx = coincident_faces.shape[0]
        return coincident_faces.reshape(
            nx // factor, factor, coincident_faces.shape[1]
        ).mean(axis=1)
    raise ValueError(f"Unsupported field for restriction: {key}")

def get_native_slices(res_3d):
    """Copy a pseudo-2D x-z slice to host while retaining C-grid staggering."""
    return {
        'u': np.array(res_3d['u'][:, 1, :]),
        'w': np.array(res_3d['w'][:, 1, :]),
        'pi': np.array(res_3d['pi'][:, 1, :]),
        'th_v': np.array(res_3d['th_v'][:, 1, :]),
    }

def run_bubble_at_resolution(
    core_type, dx, *, t_end, alpha=0.55, dt_mode="constant"
):
    nx = int(10000 / dx)
    nz = int(10000 / dx)
    ny = 3
    
    if dt_mode == "scaling":
        dt = DT_AT_250M * dx / 250.0
        if core_type == "sisl":
            core_kwargs = {"dt": dt, "nu_div_factor": 0.0, "nu_h_factor": 0.0, "damp_height": 7500.0, "max_damp": 0.0, "alpha": alpha, "solver_tol": 1e-12, "solver_maxiter": 100, "solver_restart": 100}
        elif core_type == "split-explicit":
            core_kwargs = {"dt": dt, "ns": SPLIT_EXPLICIT_NS, "nu_div_factor": 0.0, "nu_h_factor": 0.0, "damp_height": 7500.0, "max_damp": 0.0, "alpha": alpha}
        else:
            raise ValueError(f"Unknown core: {core_type}")
    elif dt_mode == "constant":
        # Constant reference timestep to isolate spatial second-order accuracy (Option A)
        dt = 0.5
        if core_type == "sisl":
            core_kwargs = {"dt": dt, "nu_div_factor": 0.0, "nu_h_factor": 0.0, "damp_height": 7500.0, "max_damp": 0.0, "alpha": alpha, "solver_tol": 1e-12, "solver_maxiter": 100, "solver_restart": 100}
        elif core_type == "split-explicit":
            core_kwargs = {"dt": dt, "ns": SPLIT_EXPLICIT_NS, "nu_div_factor": 0.0, "nu_h_factor": 0.0, "damp_height": 7500.0, "max_damp": 0.0, "alpha": alpha}
        else:
            raise ValueError(f"Unknown core: {core_type}")
    else:
        raise ValueError(f"Unknown dt_mode: {dt_mode}")

    acoustic_info = (
        f", ns={SPLIT_EXPLICIT_NS}, dt_acoustic={dt / SPLIT_EXPLICIT_NS:.5f}s"
        if core_type == "split-explicit" else ""
    )
    print(f"  dx={dx:g}m, dt={dt:g}s{acoustic_info}")

    grid = RegionalGrid3D(nx, ny, nz, dx, dx, dx, lat_center=0.0, lon_center=0.0)
    # Force pure Cartesian geometry to eliminate pseudo-2D advection boundary artifacts
    for k in grid.m_factors: 
        grid.m_factors[k] = jnp.ones_like(grid.m_factors[k])
    grid.dm_dx_m = jnp.zeros_like(grid.dm_dx_m)
    grid.dm_dy_m = jnp.zeros_like(grid.dm_dy_m)
    grid.f_u = jnp.zeros_like(grid.f_u)
    grid.f_v = jnp.zeros_like(grid.f_v)


    op = CGridOperator3D(grid)
    constants = {'g': 9.81, 'cp': 1004.0, 'Rd': 287.0, 'cvd': 717.0, 'p0': 100000.0}

    tmp_phys = Euler3D(grid, op, constants, dt=dt)
    bg_ref = {
        'rho': tmp_phys.c['p0'] / (tmp_phys.c['Rd'] * tmp_phys.theta_bg) * \
               (tmp_phys.pi_bg ** (tmp_phys.c['cvd'] / tmp_phys.c['Rd'])),
        'pi': tmp_phys.pi_bg, 'th_v': tmp_phys.theta_bg
    }

    state = {
        'u': jnp.zeros((nx+1, ny, nz)), 'v': jnp.zeros((nx, ny+1, nz)), 'w': jnp.zeros((nx, ny, nz+1)),
        'pi': bg_ref['pi'], 'eta_dot': jnp.zeros((nx, ny, nz+1)), 'rho': bg_ref['rho'],
    }

    X, Y, Z = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_m, indexing='ij')
    bubble = jnp.where((X**2 + (Z - 2000.0)**2) <= 1500.0**2, 
                       2.0 * jnp.cos(0.5 * jnp.pi * jnp.sqrt(X**2 + (Z - 2000.0)**2) / 1500.0)**2, 0.0)
    
    state['th_v'] = bg_ref['th_v'] + bubble
    state['rho'] = constants['p0'] / (constants['Rd'] * state['th_v']) * \
                   (bg_ref['pi'] ** (constants['cvd'] / constants['Rd']))

    # Initialize history fields to ensure JAX carries them inside the scan loop
    state['u_prev'] = state['u']
    state['v_prev'] = state['v']
    state['w_prev'] = state['w']
    state['eta_dot_prev'] = state['eta_dot']
    state['tend_th_v_prev'] = jnp.zeros_like(state['th_v'])
    state['is_first_step'] = 1.0

    stepper, dt = build_dynamical_core(
        core_type=core_type, grid=grid, operators=op, constants=constants,
        initial_state=state, **core_kwargs
    )

    sim = Simulation(step_fn=lambda s, i: (stepper.step(s, i*dt, None, lambda x, f: x), jnp.max(jnp.abs(s['w']))), dt=dt)
    total_steps = int(round(t_end / dt))
    if not np.isclose(total_steps * dt, t_end):
        raise ValueError(f"dt={dt} does not land exactly on t_end={t_end}")
    chunk_steps = max(1, total_steps // NUM_PROGRESS_UPDATES)
    raw_res = sim.run(
        state, t_start=0.0, t_end=t_end, chunk_steps=chunk_steps
    )
    slices = get_native_slices(raw_res)
    del raw_res, state, tmp_phys, stepper, sim
    del bg_ref, bubble, X, Y, Z, op, grid
    release_jax_memory()
    return slices

def run_study(core_type, *, t_end, alpha=0.55, dt_mode="constant"):
    print(f"\n========================================")
    print(f"Running self-convergence study for {core_type.upper()} (alpha={alpha}, dt_mode={dt_mode})...")
    print(f"========================================")
    dxs = RESOLUTIONS
    slices = []
    for dx in dxs:
        release_jax_memory()
        result = run_bubble_at_resolution(
            core_type, dx, t_end=t_end, alpha=alpha, dt_mode=dt_mode
        )
        slices.append(result)
        del result
        release_jax_memory()
        print(f"Completed simulation at dx = {dx} m | VRAM caches cleared.")

    keys = ['u', 'w', 'pi', 'th_v']
    errors = {k: [] for k in keys}
    
    for i in range(len(dxs) - 1):
        coarse = slices[i]
        fine = slices[i+1]
        
        # 1500m boundary sponge/clipping zone crop
        crop_x = int(1500.0 / dxs[i])
        crop_z = int(1500.0 / dxs[i])
        
        for k in keys:
            fine_restricted = restrict_to_coarse(fine[k], k)
            
            # Crop to interior to avoid boundary clipping/sponge artifacts
            coarse_cropped = coarse[k][crop_x:-crop_x, crop_z:-crop_z]
            fine_cropped = fine_restricted[crop_x:-crop_x, crop_z:-crop_z]
            
            err = float(np.sqrt(np.mean((fine_cropped - coarse_cropped)**2)))
            errors[k].append(err)
            
        print(f"Res {dxs[i]:6.1f} -> {dxs[i+1]:6.1f}m | "
              f"err_u: {errors['u'][-1]:.2e} | "
              f"err_w: {errors['w'][-1]:.2e} | err_pi: {errors['pi'][-1]:.2e} | "
              f"err_th: {errors['th_v'][-1]:.2e}")

    print("\n--- Self Convergence Rates ---")
    rates = {k: [] for k in keys}
    for i in range(len(dxs) - 2):
        print(f"Res {dxs[i]:6.1f} -> {dxs[i+1]:6.1f}m:")
        for k in keys:
            if errors[k][i+1] == 0.0 or errors[k][i] == 0.0:
                rate = np.nan
            else:
                rate = np.log2(errors[k][i] / errors[k][i+1])
            rates[k].append(rate)
            print(f"  Rate {k:2s}: {rate:.2f}")

    return errors, rates, slices


def report_cross_core_convergence(sisl_slices, split_slices):
    """Check that both discretizations approach the same continuum solution."""
    keys = ['u', 'w', 'pi', 'th_v']
    differences = {key: [] for key in keys}
    rates = {key: [] for key in keys}

    print("\n--- Cross-Core Differences and Convergence Rates ---")
    for dx, sisl, split in zip(RESOLUTIONS, sisl_slices, split_slices):
        crop = int(1500.0 / dx)
        for key in keys:
            sisl_interior = sisl[key][crop:-crop, crop:-crop]
            split_interior = split[key][crop:-crop, crop:-crop]
            differences[key].append(
                float(np.sqrt(np.mean((sisl_interior - split_interior) ** 2)))
            )
        print(
            f"dx={dx:6.2f}m | "
            + " | ".join(
                f"{key}={differences[key][-1]:.2e}" for key in keys
            )
        )

    for i in range(len(RESOLUTIONS) - 1):
        print(f"Cross-core rate {RESOLUTIONS[i]:g} -> {RESOLUTIONS[i+1]:g}m:")
        for key in keys:
            coarse = differences[key][i]
            fine = differences[key][i + 1]
            rate = np.log2(coarse / fine) if coarse > 0.0 and fine > 0.0 else np.nan
            rates[key].append(rate)
            print(f"  Rate {key:4s}: {rate:.2f}")

    return differences, rates

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the rising-bubble self/cross convergence audit."
    )
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--name")
    parser.add_argument("--t-end", type=float, default=24.0)
    return parser.parse_args()


def main():
    args = parse_args()
    execution = args.name or f"t{args.t_end:g}"
    errors_sisl, rates_sisl, slices_sisl = run_study(
        "sisl", t_end=args.t_end, alpha=0.5, dt_mode="scaling"
    )
    errors_se, rates_se, slices_se = run_study(
        "split-explicit", t_end=args.t_end, alpha=0.5, dt_mode="scaling"
    )
    cross_differences, cross_rates = report_cross_core_convergence(
        slices_sisl, slices_se
    )
    summary = {
        "test": "rising_bubble_core_convergence",
        "t_end_s": args.t_end,
        "resolutions_m": RESOLUTIONS,
        "dt_at_250m_s": DT_AT_250M,
        "dt_mode": "scaling",
        "split_explicit_ns": SPLIT_EXPLICIT_NS,
        "cores": {
            "sisl": {
                "self_errors": errors_sisl, "self_orders": rates_sisl
            },
            "split-explicit": {
                "self_errors": errors_se, "self_orders": rates_se
            },
        },
        "cross_core": {
            "differences": cross_differences, "orders": cross_rates
        },
    }
    layout = ArtifactLayout(
        kind="verification", case="rising_bubble_core_convergence",
        execution=execution, output_root=args.output_root,
    ).create()
    path = layout.data / "summary.json"
    with path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(f"Saved {args.t_end:g} s bubble convergence artifact to {path}")


if __name__ == "__main__":
    main()
