#!/usr/bin/env python3
"""Differentiate mature-supercell precipitation with respect to its state."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
import numpy as np
import xarray as xr

from run import (
    CONSTANTS,
    DOMAIN_X,
    DOMAIN_Y,
    DOMAIN_Z,
    ConstantLaplacianDiffusion,
    generate_wk_sounding,
)
from suetes.physics.base import PhysicsSuite
from suetes.physics.microphysics import KesslerWarmRain
from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.steppers import build_dynamical_core
from suetes.shared.artifacts import save_plot_dataset


def build_case(args):
    nx = round(DOMAIN_X / args.dx)
    ny = round(DOMAIN_Y / args.dx)
    nz = round(DOMAIN_Z / args.dz)
    if not (
        np.isclose(nx * args.dx, DOMAIN_X)
        and np.isclose(ny * args.dx, DOMAIN_Y)
        and np.isclose(nz * args.dz, DOMAIN_Z)
    ):
        raise ValueError("dx and dz must divide the domain dimensions")

    grid = RegionalGrid3D(
        nx, ny, nz, args.dx, args.dx, args.dz,
        lat_center=0.0, lon_center=0.0,
    )
    operators = CGridOperator3D(grid, periodic_axes=(0,))
    dry_theta_1d, theta_v_1d, exner_1d, vapor_1d = generate_wk_sounding(
        np.asarray(grid.z_m)
    )
    theta_v_bg = jnp.broadcast_to(theta_v_1d, (nx, ny, nz))
    exner_bg = jnp.broadcast_to(exner_1d, (nx, ny, nz))
    vapor_bg = jnp.broadcast_to(vapor_1d, (nx, ny, nz))
    rho_bg = (
        CONSTANTS["p0"] / (CONSTANTS["Rd"] * theta_v_bg)
        * exner_bg ** (CONSTANTS["cvd"] / CONSTANTS["Rd"])
    )
    u_profile = -12.0 + 4.8e-3 * jnp.minimum(grid.z_m, 2500.0)
    u_bg = jnp.broadcast_to(u_profile, (nx + 1, ny, nz))

    x, y, z = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_m, indexing="ij")
    radius = jnp.sqrt(
        (x / 10_000.0) ** 2
        + (y / 10_000.0) ** 2
        + ((z - 2000.0) / 2000.0) ** 2
    )
    bubble = jnp.where(
        radius <= 1.0,
        3.0 * jnp.cos(0.5 * jnp.pi * radius) ** 2,
        0.0,
    )
    dry_theta = jnp.broadcast_to(dry_theta_1d, (nx, ny, nz)) + bubble
    theta_v_initial = dry_theta * (1.0 + 0.61 * vapor_bg)
    rho_initial = (
        CONSTANTS["p0"] / (CONSTANTS["Rd"] * theta_v_initial)
        * exner_bg ** (CONSTANTS["cvd"] / CONSTANTS["Rd"])
    )
    state = {
        "u": u_bg,
        "v": jnp.zeros((nx, ny + 1, nz), dtype=jnp.float32),
        "w": jnp.zeros((nx, ny, nz + 1), dtype=jnp.float32),
        "eta_dot": jnp.zeros((nx, ny, nz + 1), dtype=jnp.float32),
        "pi": exner_bg,
        "rho": rho_initial,
        "th_v": theta_v_initial,
        "q": vapor_bg,
        "q_c": jnp.zeros((nx, ny, nz), dtype=jnp.float32),
        "q_r": jnp.zeros((nx, ny, nz), dtype=jnp.float32),
        "precip_step": jnp.zeros((nx, ny), dtype=jnp.float32),
    }
    background_state = {
        "th_v": theta_v_bg,
        "pi": exner_bg,
        "rho": rho_bg,
    }

    index = jnp.arange(ny)
    distance = jnp.minimum(index, ny - 1 - index)
    mask_y = jnp.where(
        distance < args.relaxation_cells,
        jnp.cos(
            0.5 * jnp.pi * distance / args.relaxation_cells
        ) ** 2,
        0.0,
    )[None, :, None]

    def boundary_conditions(current, forcing):
        del forcing
        references = {
            "u": u_bg,
            "v": jnp.zeros_like(current["v"]),
            "th_v": theta_v_bg,
            "pi": exner_bg,
            "rho": rho_bg,
            "q": vapor_bg,
            "q_c": jnp.zeros_like(current["q_c"]),
            "q_r": jnp.zeros_like(current["q_r"]),
        }
        output = {}
        for key, value in current.items():
            if key not in references:
                output[key] = value
                continue
            mask = (
                jnp.pad(mask_y, ((0, 0), (0, 1), (0, 0)), mode="edge")
                if key == "v" else mask_y
            )
            output[key] = (1.0 - mask) * value + mask * references[key]
        output["u"] = output["u"].at[-1].set(output["u"][0])
        return output

    suite = PhysicsSuite()
    diffusion = ConstantLaplacianDiffusion(
        grid,
        args.diffusivity,
        args.dt,
        periodic_axes=(0,),
        tracer_diffusivity=args.tracer_diffusivity,
    )
    suite.add_tendency_scheme(diffusion)
    suite.add_update_scheme(diffusion)
    suite.add_update_scheme(KesslerWarmRain(CONSTANTS, dt=args.dt, grid=grid))
    for tracer in ("q", "q_c", "q_r"):
        suite.register_tracer(tracer)

    stepper, actual_dt = build_dynamical_core(
        core_type="split-explicit",
        grid=grid,
        operators=operators,
        constants=CONSTANTS,
        initial_state=state,
        physics_suite=suite,
        dt=args.dt,
        ns=args.acoustic_substeps,
        alpha=0.5,
        nu_h_factor=0.0,
        nu_div_factor=args.divergence_damping,
        damp_height=args.sponge_start,
        max_damp=args.sponge_strength,
        background_state=background_state,
    )
    return grid, state, stepper, boundary_conditions, actual_dt


def save_restart(path, state, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(
        temporary,
        **{key: np.asarray(value) for key, value in state.items()},
    )
    temporary.replace(path)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))


def load_restart(path):
    with np.load(path) as source:
        return {
            key: jnp.asarray(source[key])
            for key in source.files
        }


def parse_float_list(value):
    values = tuple(float(item) for item in value.split(",") if item.strip())
    if not values or any(item <= 0.0 for item in values):
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of positive values"
        )
    return values


def dilate_horizontal_mask(mask, iterations):
    """Dilate a mask periodically in x and without wrapping in y."""
    result = np.asarray(mask, dtype=bool)
    for _ in range(iterations):
        expanded = result | np.roll(result, 1, axis=0) | np.roll(
            result, -1, axis=0
        )
        expanded[:, 1:] |= result[:, :-1]
        expanded[:, :-1] |= result[:, 1:]
        result = expanded
    return result


def center_vertical_velocity(w):
    return 0.5 * (w[:, :, :-1] + w[:, :, 1:])


def forward_spinup(state, stepper, boundary_conditions, dt, t_end, chunk_steps):
    def chunk(current, start_step):
        def one_step(carry, offset):
            step_index = start_step + offset
            next_state = stepper.step(
                carry, step_index * dt, None, boundary_conditions
            )
            return next_state, jnp.max(jnp.abs(next_state["w"]))

        return jax.lax.scan(one_step, current, jnp.arange(chunk_steps))

    compiled = jax.jit(chunk).lower(state, 0).compile()
    current = state
    current_step = 0
    total_steps = round(t_end / dt)
    if total_steps % chunk_steps:
        raise ValueError("spinup steps must be divisible by spinup-chunk-steps")
    for _ in range(total_steps // chunk_steps):
        current, maximum_w = compiled(current, current_step)
        jax.tree_util.tree_leaves(current)[0].block_until_ready()
        current_step += chunk_steps
        print(
            f"  spin-up t={current_step * dt:6.0f} s | "
            f"max|w|={float(jnp.max(maximum_w)):7.3f} m/s"
        )
    return current


def run(args):
    if args.window_end <= args.spinup_end:
        raise ValueError("window-end must exceed spinup-end")
    if not 0.0 < args.target_rain_fraction <= 1.0:
        raise ValueError("target-rain-fraction must lie in (0, 1]")
    if args.target_halo_cells < 0:
        raise ValueError("target-halo-cells must be non-negative")
    if args.diffusivity < 0.0 or args.tracer_diffusivity < 0.0:
        raise ValueError("diffusivities must be non-negative")
    grid, initial_state, stepper, boundary_conditions, dt = build_case(args)
    restart = args.output_dir / "spinup_state.npz"
    if args.reuse_spinup:
        if not restart.is_file():
            raise FileNotFoundError(restart)
        metadata_path = restart.with_suffix(".json")
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        restart_metadata = json.loads(metadata_path.read_text())
        expected = {
            "time_s": args.spinup_end,
            "dx_m": args.dx,
            "dz_m": args.dz,
            "dt_s": args.dt,
            "diffusivity_m2_s": args.diffusivity,
            "tracer_diffusivity_m2_s": args.tracer_diffusivity,
        }
        mismatches = {
            key: (
                restart_metadata.get(
                    key,
                    33.33
                    if key in {
                        "diffusivity_m2_s",
                        "tracer_diffusivity_m2_s",
                    }
                    else np.nan,
                ),
                value,
            )
            for key, value in expected.items()
            if not np.isclose(
                restart_metadata.get(
                    key,
                    33.33
                    if key in {
                        "diffusivity_m2_s",
                        "tracer_diffusivity_m2_s",
                    }
                    else np.nan,
                ),
                value,
            )
        }
        if mismatches:
            raise ValueError(
                f"Spin-up checkpoint does not match requested configuration: "
                f"{mismatches}"
            )
        state_spinup = load_restart(restart)
        print(f"Loaded {restart}")
    else:
        print(f"Spinning up to t={args.spinup_end:g} s")
        state_spinup = forward_spinup(
            initial_state,
            stepper,
            boundary_conditions,
            dt,
            args.spinup_end,
            args.spinup_chunk_steps,
        )
        save_restart(
            restart,
            jax.device_get(state_spinup),
            {
                "time_s": args.spinup_end,
                "dx_m": args.dx,
                "dz_m": args.dz,
                "dt_s": args.dt,
                "diffusivity_m2_s": args.diffusivity,
                "tracer_diffusivity_m2_s": args.tracer_diffusivity,
                "periodic_x": True,
            },
        )
        print(f"Saved {restart}")

    nx, ny, nz = state_spinup["th_v"].shape
    y_index = jnp.arange(ny)
    interior_y = (
        (y_index >= args.relaxation_cells)
        & (y_index < ny - args.relaxation_cells)
    )
    interior_mask = jnp.broadcast_to(interior_y[None, :], (nx, ny))
    control_mask = interior_mask[:, :, None]
    window_duration = args.window_end - args.spinup_end
    rate_factor = 3600.0 / window_duration

    moisture_factor = (
        1.0
        + (1.0 / CONSTANTS["epsilon"] - 1.0) * state_spinup["q"]
        - state_spinup["q_c"]
        - state_spinup["q_r"]
    )
    dry_theta_spinup = state_spinup["th_v"] / moisture_factor

    def controlled_state(delta_theta, delta_q):
        state = dict(state_spinup)
        theta = dry_theta_spinup + delta_theta * control_mask
        q = state_spinup["q"] + delta_q * control_mask
        factor = (
            1.0
            + (1.0 / CONSTANTS["epsilon"] - 1.0) * q
            - state_spinup["q_c"]
            - state_spinup["q_r"]
        )
        state["q"] = q
        state["th_v"] = theta * factor
        state["rho"] = (
            CONSTANTS["p0"] / (CONSTANTS["Rd"] * state["th_v"])
            * state["pi"] ** (CONSTANTS["cvd"] / CONSTANTS["Rd"])
        )
        state["window_rain"] = jnp.zeros((nx, ny), state["th_v"].dtype)
        return state

    total_steps = round((args.window_end - args.spinup_end) / dt)
    if total_steps % args.adjoint_chunk_steps:
        raise ValueError(
            "adjoint window steps must be divisible by adjoint-chunk-steps"
        )
    number_chunks = total_steps // args.adjoint_chunk_steps
    start_step = round(args.spinup_end / dt)

    @jax.checkpoint
    def scan_chunk(current, chunk_index):
        def one_step(carry, offset):
            step_index = (
                start_step
                + chunk_index * args.adjoint_chunk_steps
                + offset
            )
            next_state = stepper.step(
                carry, step_index * dt, None, boundary_conditions
            )
            next_state["window_rain"] = (
                carry["window_rain"] + next_state["precip_step"]
            )
            return next_state, None

        return jax.lax.scan(
            one_step, current, jnp.arange(args.adjoint_chunk_steps)
        )[0], None

    def integrate_window(state):
        return jax.lax.scan(
            scan_chunk, state, jnp.arange(number_chunks)
        )[0]

    zeros = jnp.zeros_like(state_spinup["th_v"])
    integrate_compiled = jax.jit(
        lambda delta_theta, delta_q: integrate_window(
            controlled_state(delta_theta, delta_q)
        )
    )
    print("Locating the fixed precipitation target from the baseline trajectory")
    baseline_final = integrate_compiled(zeros, zeros)
    baseline_final["window_rain"].block_until_ready()
    baseline_rain_field = np.asarray(baseline_final["window_rain"])
    peak_rain = float(np.max(baseline_rain_field))
    if not np.isfinite(peak_rain) or peak_rain <= 0.0:
        raise FloatingPointError("Baseline trajectory produced no finite rain")
    target_mask_np = (
        baseline_rain_field >= args.target_rain_fraction * peak_rain
    ) & np.asarray(interior_mask)
    target_mask_np = dilate_horizontal_mask(
        target_mask_np, args.target_halo_cells
    )
    target_mask_np &= np.asarray(interior_mask)
    if not np.any(target_mask_np):
        raise RuntimeError("The fixed storm target mask is empty")
    target_mask = jnp.asarray(target_mask_np)
    rain_denominator = jnp.sum(target_mask)
    print(
        f"Fixed storm target: {int(np.sum(target_mask_np))} columns, "
        f"threshold={args.target_rain_fraction:.3f} of "
        f"{peak_rain:.3f} mm peak accumulation"
    )

    def objective(delta_theta, delta_q):
        final = integrate_window(controlled_state(delta_theta, delta_q))
        value = rate_factor * (
            jnp.sum(final["window_rain"] * target_mask)
            / rain_denominator
        )
        return value, final["window_rain"]

    value_and_gradient = jax.jit(
        jax.value_and_grad(objective, argnums=(0, 1), has_aux=True)
    )
    print(
        f"Differentiating precipitation over "
        f"{args.spinup_end:g}--{args.window_end:g} s"
    )
    start = time.perf_counter()
    (
        (baseline_rain, baseline_rain_field),
        (gradient_theta, gradient_q),
    ) = value_and_gradient(zeros, zeros)
    baseline_rain.block_until_ready()
    gradient_theta.block_until_ready()
    gradient_q.block_until_ready()
    baseline_rain_field.block_until_ready()
    print(
        f"Adjoint complete in {time.perf_counter() - start:.1f} s | "
        f"target-region mean rain rate={float(baseline_rain):.6f} mm h^-1"
    )
    for name, gradient in (
        ("theta", gradient_theta),
        ("water vapor", gradient_q),
    ):
        if not bool(jnp.all(jnp.isfinite(gradient))):
            raise FloatingPointError(f"Non-finite {name} sensitivity")

    evaluate_objective = jax.jit(objective)
    theta_unit_direction = gradient_theta / jnp.maximum(
        jnp.max(jnp.abs(gradient_theta)), 1.0e-30
    )
    q_unit_direction = gradient_q / jnp.maximum(
        jnp.max(jnp.abs(gradient_q)), 1.0e-30
    )

    def taylor_sweep(name, amplitudes, unit_direction, gradient, theta_control):
        actual_values = []
        linear_values = []
        relative_errors = []
        print(f"{name} centered Taylor sweep:")
        for amplitude in amplitudes:
            perturbation = amplitude * unit_direction
            if theta_control:
                value_plus, _ = evaluate_objective(perturbation, zeros)
                value_minus, _ = evaluate_objective(-perturbation, zeros)
            else:
                value_plus, _ = evaluate_objective(zeros, perturbation)
                value_minus, _ = evaluate_objective(zeros, -perturbation)
            value_plus.block_until_ready()
            value_minus.block_until_ready()
            linear = jnp.sum(gradient * perturbation)
            # Half of the centered objective difference has the same units and
            # leading-order value as the one-sided linear response g dot dx.
            actual = 0.5 * (value_plus - value_minus)
            relative = jnp.abs(actual - linear) / jnp.maximum(
                jnp.abs(linear), 1.0e-20
            )
            actual_values.append(float(actual))
            linear_values.append(float(linear))
            relative_errors.append(float(relative))
            print(
                f"  amplitude={amplitude:.3e}: "
                f"actual={float(actual):.6e}, linear={float(linear):.6e}, "
                f"relative error={float(relative):.3e}"
            )
        return (
            np.asarray(actual_values),
            np.asarray(linear_values),
            np.asarray(relative_errors),
        )

    theta_actual, theta_linear, theta_error = taylor_sweep(
        "Dry-potential-temperature",
        args.taylor_theta_amplitudes,
        theta_unit_direction,
        gradient_theta,
        True,
    )
    q_actual, q_linear, q_error = taylor_sweep(
        "Water-vapor",
        args.taylor_q_amplitudes,
        q_unit_direction,
        gradient_q,
        False,
    )

    theta_column = jnp.sqrt(jnp.sum(gradient_theta**2, axis=2))
    q_column = jnp.sqrt(jnp.sum(gradient_q**2, axis=2))
    # A section through the strongest precipitating column is more
    # representative than the geometric mid-plane for an asymmetric storm.
    peak_x, section_y = np.unravel_index(
        np.argmax(baseline_rain_field), baseline_rain_field.shape
    )
    del peak_x
    spinup_w = center_vertical_velocity(state_spinup["w"])
    final_w = center_vertical_velocity(baseline_final["w"])
    dataset = xr.Dataset(
        data_vars={
            "window_accumulated_rain": (
                ("x", "y"), np.asarray(baseline_rain_field)
            ),
            "theta_sensitivity": (
                ("x", "y", "z"), np.asarray(gradient_theta)
            ),
            "water_vapor_sensitivity": (
                ("x", "y", "z"), np.asarray(gradient_q)
            ),
            "theta_column_sensitivity": (
                ("x", "y"), np.asarray(theta_column)
            ),
            "water_vapor_column_sensitivity": (
                ("x", "y"), np.asarray(q_column)
            ),
            "theta_sensitivity_xz": (
                ("x", "z"), np.asarray(gradient_theta[:, section_y, :])
            ),
            "water_vapor_sensitivity_xz": (
                ("x", "z"), np.asarray(gradient_q[:, section_y, :])
            ),
            "target_mask": (("x", "y"), target_mask_np.astype(np.int8)),
            "spinup_vertical_velocity_xz": (
                ("x", "z"), np.asarray(spinup_w[:, section_y, :])
            ),
            "final_vertical_velocity_xz": (
                ("x", "z"), np.asarray(final_w[:, section_y, :])
            ),
            "spinup_cloud_water_xz": (
                ("x", "z"), np.asarray(state_spinup["q_c"][:, section_y, :])
            ),
            "spinup_rain_water_xz": (
                ("x", "z"), np.asarray(state_spinup["q_r"][:, section_y, :])
            ),
            "final_cloud_water_xz": (
                ("x", "z"), np.asarray(baseline_final["q_c"][:, section_y, :])
            ),
            "final_rain_water_xz": (
                ("x", "z"), np.asarray(baseline_final["q_r"][:, section_y, :])
            ),
            "theta_taylor_actual_change": (
                ("theta_taylor_amplitude",), theta_actual
            ),
            "theta_taylor_linear_change": (
                ("theta_taylor_amplitude",), theta_linear
            ),
            "theta_taylor_relative_error": (
                ("theta_taylor_amplitude",), theta_error
            ),
            "water_vapor_taylor_actual_change": (
                ("q_taylor_amplitude",), q_actual
            ),
            "water_vapor_taylor_linear_change": (
                ("q_taylor_amplitude",), q_linear
            ),
            "water_vapor_taylor_relative_error": (
                ("q_taylor_amplitude",), q_error
            ),
        },
        coords={
            "x": np.asarray(grid.x_m),
            "y": np.asarray(grid.y_m),
            "z": np.asarray(grid.z_m),
            "theta_taylor_amplitude": np.asarray(
                args.taylor_theta_amplitudes
            ),
            "q_taylor_amplitude": np.asarray(args.taylor_q_amplitudes),
        },
        attrs={
            "benchmark": "ERF supercell precipitation sensitivity",
            "spinup_end_s": args.spinup_end,
            "window_end_s": args.window_end,
            "dx_m": args.dx,
            "dz_m": args.dz,
            "dt_s": args.dt,
            "diffusivity_m2_s": args.diffusivity,
            "tracer_diffusivity_m2_s": args.tracer_diffusivity,
            "periodic_x": 1,
            "objective": "fixed-storm-region mean rain rate in window",
            "baseline_objective_mm_h": float(baseline_rain),
            "target_rain_fraction": args.target_rain_fraction,
            "target_halo_cells": args.target_halo_cells,
            "target_columns": int(np.sum(target_mask_np)),
            "section_y_index": int(section_y),
            "section_y_m": float(np.asarray(grid.y_m)[section_y]),
            "taylor_test": "centered directional difference",
            "target_mask_note": (
                "Derived once from the unperturbed accumulated-rain field "
                "and held fixed during differentiation."
            ),
        },
    )
    artifact = save_plot_dataset(
        dataset,
        args.output_dir / "sensitivity.nc",
        experiment="squall_line_3d_precipitation_sensitivity",
    )
    print(f"Saved {artifact}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/squall_line_3d_sensitivity"),
    )
    parser.add_argument("--dx", type=float, default=500.0)
    parser.add_argument("--dz", type=float, default=500.0)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--spinup-end", type=float, default=6600.0)
    parser.add_argument("--window-end", type=float, default=6750.0)
    parser.add_argument("--spinup-chunk-steps", type=int, default=600)
    parser.add_argument("--adjoint-chunk-steps", type=int, default=20)
    parser.add_argument(
        "--taylor-theta-amplitudes",
        type=parse_float_list,
        default=(1.0e-3, 3.0e-3, 1.0e-2),
        help="comma-separated maximum-norm dry-theta perturbations (K)",
    )
    parser.add_argument(
        "--taylor-q-amplitudes",
        type=parse_float_list,
        default=(1.0e-7, 3.0e-7, 1.0e-6),
        help="comma-separated maximum-norm vapor perturbations (kg kg-1)",
    )
    parser.add_argument(
        "--target-rain-fraction",
        type=float,
        default=0.10,
        help="baseline-rain fraction defining the frozen storm footprint",
    )
    parser.add_argument(
        "--target-halo-cells",
        type=int,
        default=4,
        help="horizontal dilation applied to the frozen storm footprint",
    )
    parser.add_argument("--reuse-spinup", action="store_true")
    parser.add_argument("--diffusivity", type=float, default=33.33)
    parser.add_argument(
        "--tracer-diffusivity",
        type=float,
        default=33.33,
        help="Laplacian diffusivity for q_v, q_c, and q_r (m2 s-1)",
    )
    parser.add_argument("--acoustic-substeps", type=int, default=2)
    parser.add_argument("--divergence-damping", type=float, default=0.0)
    parser.add_argument("--relaxation-cells", type=int, default=10)
    parser.add_argument("--sponge-start", type=float, default=18_000.0)
    parser.add_argument("--sponge-strength", type=float, default=0.05)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
