#!/usr/bin/env python3
"""Smooth-mode temporal convergence tests for both regional 3-D cores.

Three deliberately smooth problems complement the nonlinear rising bubble:

* ``tracer``: an interior Gaussian translated by uniform flow;
* ``gravity``: a small-amplitude standing buoyancy mode;
* ``acoustic``: a small-amplitude pressure/velocity standing mode.

Each problem is integrated on one fixed grid with a timestep-halving family.
The finest run is not treated as truth: errors are differences between
successive refinements and the reported order is the three-level Richardson
self-convergence order.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from suetes.regional3d.euler import Euler3D
from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.steppers import build_dynamical_core
from suetes.shared.artifacts import ArtifactLayout


jax.config.update("jax_enable_x64", True)

CONSTANTS = {
    "g": 9.81, "cp": 1004.0, "Rd": 287.0,
    "cvd": 717.0, "p0": 100000.0,
}
LX = 20_000.0
LY = 1_875.0
LZ = 10_000.0


class PassiveTracerSuite:
    tracer_keys = ["q_test"]

    def __init__(self, shape):
        self.shape = shape

    def get_explicit_tendencies(
        self, state, bg, interior_mask=None, ml_params=None
    ):
        # Returning an array (rather than relying on the core's scalar zero
        # fallback) keeps the multistep SISL history shape stable in a scan.
        return {"q_test": jnp.zeros(self.shape, dtype=jnp.float64)}

    def apply_state_updates(self, state):
        return state


CASE_CONFIG = {
    "tracer": {"t_end": 20.0, "dt": [2.0, 1.0, 0.5, 0.25],
               "fields": ("q_test",)},
    "gravity": {"t_end": 20.0, "dt": [2.0, 1.0, 0.5, 0.25],
                "fields": ("th_v", "w")},
    "acoustic": {"t_end": 10.0, "dt": [1.0, 0.5, 0.25, 0.125],
                 "fields": ("pi", "u")},
}


def _cartesian_grid(nx: int, nz: int) -> RegionalGrid3D:
    grid = RegionalGrid3D(
        nx, 3, nz, LX / nx, LY / 3, LZ / nz,
        lat_center=0.0, lon_center=0.0,
    )
    for key in grid.m_factors:
        grid.m_factors[key] = jnp.ones_like(grid.m_factors[key])
    grid.dm_dx_m = jnp.zeros_like(grid.dm_dx_m)
    grid.dm_dy_m = jnp.zeros_like(grid.dm_dy_m)
    grid.f_u = jnp.zeros_like(grid.f_u)
    grid.f_v = jnp.zeros_like(grid.f_v)
    return grid


def _initial_state(
    case: str, grid: RegionalGrid3D, physics: Euler3D
) -> dict[str, jax.Array]:
    rho_bg = (
        CONSTANTS["p0"] / (CONSTANTS["Rd"] * physics.theta_bg)
        * physics.pi_bg ** (CONSTANTS["cvd"] / CONSTANTS["Rd"])
    )
    state = {
        "u": jnp.zeros((grid.nx + 1, grid.ny, grid.nz)),
        "v": jnp.zeros((grid.nx, grid.ny + 1, grid.nz)),
        "w": jnp.zeros((grid.nx, grid.ny, grid.nz + 1)),
        "eta_dot": jnp.zeros((grid.nx, grid.ny, grid.nz + 1)),
        "pi": physics.pi_bg,
        "th_v": physics.theta_bg,
        "rho": rho_bg,
    }
    x_m = grid.x_m[:, None, None]
    z_m = grid.z_m[None, None, :]
    if case == "tracer":
        state["u"] = jnp.full_like(state["u"], 100.0)
        state["q_test"] = jnp.exp(
            -((x_m + 3000.0) ** 2 + (z_m - 5000.0) ** 2)
            / (2.0 * 900.0**2)
        ) + jnp.zeros_like(state["th_v"])
    elif case == "gravity":
        # Small amplitude keeps the mode in the linear, smooth regime.
        mode = (
            jnp.cos(2.0 * jnp.pi * x_m / LX)
            * jnp.sin(jnp.pi * z_m / LZ)
        )
        state["th_v"] = physics.theta_bg + 1.0e-2 * mode
        state["rho"] = (
            CONSTANTS["p0"] / (CONSTANTS["Rd"] * state["th_v"])
            * state["pi"] ** (CONSTANTS["cvd"] / CONSTANTS["Rd"])
        )
    elif case == "acoustic":
        amplitude = 1.0e-6
        state["pi"] = physics.pi_bg + amplitude * jnp.cos(
            2.0 * jnp.pi * x_m / LX
        )
        state["rho"] = (
            CONSTANTS["p0"] / (CONSTANTS["Rd"] * state["th_v"])
            * state["pi"] ** (CONSTANTS["cvd"] / CONSTANTS["Rd"])
        )
    else:
        raise ValueError(case)
    return state


def _run(case: str, core: str, dt: float, nx: int, nz: int) -> dict[str, np.ndarray]:
    grid = _cartesian_grid(nx, nz)
    operators = CGridOperator3D(grid)
    suite = PassiveTracerSuite((nx, 3, nz)) if case == "tracer" else None
    physics = Euler3D(
        grid, operators, CONSTANTS, dt=dt,
        N_bv=0.01 if case == "gravity" else 0.0,
        damp_height=1.0e9, max_damp=0.0,
        nu_h_factor=0.0, nu_div_factor=0.0,
        physics_suite=suite,
    )
    state = _initial_state(case, grid, physics)
    kwargs = {
        "dt": dt, "alpha": 0.5, "N_bv": 0.01 if case == "gravity" else 0.0,
        "damp_height": 1.0e9, "max_damp": 0.0,
        "nu_h_factor": 0.0, "nu_div_factor": 0.0,
    }
    if core == "sisl":
        kwargs.update(
            solver_tol=1.0e-10, solver_maxiter=80, solver_restart=40
        )
    else:
        kwargs["ns"] = 12
    stepper, actual_dt = build_dynamical_core(
        core, grid, operators, CONSTANTS, state,
        physics_suite=suite, **kwargs,
    )
    steps = round(CASE_CONFIG[case]["t_end"] / actual_dt)

    def advance(current, index):
        return stepper.step(
            current, index * actual_dt, None, lambda value, forcing: value
        ), None

    final, _ = jax.lax.scan(advance, state, jnp.arange(steps))
    jax.tree_util.tree_leaves(final)[0].block_until_ready()
    result = {
        field: np.asarray(final[field]) for field in CASE_CONFIG[case]["fields"]
    }
    del final, state, stepper, physics, operators, grid
    gc.collect()
    jax.clear_caches()
    return result


def _cropped(field: np.ndarray) -> np.ndarray:
    # Remove the locked/open lateral boundary stencil while retaining most of
    # the smooth mode. Vertical rigid boundaries are removed for w.
    slices = [slice(4, -4), slice(None), slice(None)]
    if field.ndim == 3 and field.shape[2] > 8:
        slices[2] = slice(2, -2)
    return field[tuple(slices)]


def _relative_error(coarse: np.ndarray, fine: np.ndarray) -> float:
    difference = _cropped(coarse) - _cropped(fine)
    scale = np.linalg.norm(_cropped(fine))
    return float(np.linalg.norm(difference) / max(scale, np.finfo(float).tiny))


def run_study(
    case: str, core: str, nx: int, nz: int
) -> tuple[list[float], dict[str, list[float]], dict[str, list[float]]]:
    dt_values = CASE_CONFIG[case]["dt"]
    solutions = []
    for dt in dt_values:
        print(f"[{case}/{core}] dt={dt:g} s")
        solutions.append(_run(case, core, dt, nx, nz))
    errors = {field: [] for field in CASE_CONFIG[case]["fields"]}
    rates = {field: [] for field in CASE_CONFIG[case]["fields"]}
    for left, right in zip(solutions[:-1], solutions[1:]):
        for field in errors:
            errors[field].append(_relative_error(left[field], right[field]))
    for field, values in errors.items():
        rates[field] = [
            math.log(left / right, 2.0)
            for left, right in zip(values[:-1], values[1:])
        ]
    return dt_values, errors, rates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=tuple(CASE_CONFIG), action="append",
        help="Run selected case(s); default is all three",
    )
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--nz", type=int, default=16)
    parser.add_argument("--min-wave-order", type=float, default=1.7)
    parser.add_argument(
        "--min-tracer-order", type=float, default=0.9,
        help=(
            "Current conservative FFSL transport is first-order in this "
            "timestep-refinement test; keep this separate from the wave gate"
        ),
    )
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--name", default="default")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = args.case or list(CASE_CONFIG)
    summary = {
        "passed": True,
        "thresholds": {
            "wave_order": args.min_wave_order,
            "tracer_order": args.min_tracer_order,
        },
        "cases": {},
    }
    for case in cases:
        summary["cases"][case] = {}
        for core in ("sisl", "split-explicit"):
            dt, errors, rates = run_study(case, core, args.nx, args.nz)
            # The last rate is least contaminated by higher-order terms.
            threshold = (
                args.min_tracer_order if case == "tracer"
                else args.min_wave_order
            )
            passed = all(values[-1] >= threshold for values in rates.values())
            summary["passed"] &= passed
            summary["cases"][case][core] = {
                "passed": passed, "dt_s": dt,
                "self_errors": errors, "orders": rates,
            }
            print(
                f"[{case}/{core}] finest orders: "
                + ", ".join(f"{key}={value[-1]:.3f}" for key, value in rates.items())
            )
    layout = ArtifactLayout(
        kind="verification", case="smooth_mode_core_convergence",
        execution=args.name, output_root=args.output_root,
    ).create()
    path = layout.data / "summary.json"
    with path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Summary: {path}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
