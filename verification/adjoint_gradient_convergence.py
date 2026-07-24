"""Verify dual-core adjoints with Taylor tests and temporal refinement.

The Taylor test checks the derivative of each *discrete* model.  The temporal
study then checks whether the two discrete gradients converge as the large
time step is reduced on a fixed grid.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from suetes.regional3d.euler import Euler3D
from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.steppers import build_dynamical_core
from suetes.shared.artifacts import ArtifactLayout


def build_case(nx: int = 12, ny: int = 3, nz: int = 8, dx: float = 375.0):
    grid = RegionalGrid3D(nx, ny, nz, dx, dx, dx, lat_center=0.0, lon_center=0.0)
    for name in grid.m_factors:
        grid.m_factors[name] = jnp.ones_like(grid.m_factors[name])
    grid.dm_dx_m = jnp.zeros_like(grid.dm_dx_m)
    grid.dm_dy_m = jnp.zeros_like(grid.dm_dy_m)
    grid.f_u = jnp.zeros_like(grid.f_u)
    grid.f_v = jnp.zeros_like(grid.f_v)
    operators = CGridOperator3D(grid)
    constants = {
        "g": 9.81, "cp": 1004.0, "Rd": 287.0, "cvd": 717.0, "p0": 100000.0
    }
    physics = Euler3D(grid, operators, constants, dt=1.0, N_bv=0.0)
    pi = physics.pi_bg
    theta = physics.theta_bg
    rho = constants["p0"] / (constants["Rd"] * theta) * (
        pi ** (constants["cvd"] / constants["Rd"])
    )
    state = {
        "u": jnp.zeros((nx + 1, ny, nz)),
        "v": jnp.zeros((nx, ny + 1, nz)),
        "w": jnp.zeros((nx, ny, nz + 1)),
        "pi": pi,
        "eta_dot": jnp.zeros((nx, ny, nz + 1)),
        "rho": rho,
    }
    x, _, z = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_m, indexing="ij")
    radius = jnp.sqrt(x**2 + (z - 1500.0) ** 2)
    bubble = jnp.where(
        radius <= 1000.0,
        3.0 * jnp.cos(0.5 * jnp.pi * radius / 1000.0) ** 2,
        0.0,
    )
    state["th_v"] = theta + bubble
    state["rho"] = constants["p0"] / (constants["Rd"] * state["th_v"]) * (
        pi ** (constants["cvd"] / constants["Rd"])
    )
    state["u_prev"] = state["u"]
    state["v_prev"] = state["v"]
    state["w_prev"] = state["w"]
    state["eta_dot_prev"] = state["eta_dot"]
    state["tend_th_v_prev"] = jnp.zeros_like(state["th_v"])
    state["is_first_step"] = 1.0
    return grid, operators, constants, state


def objective_function(core, grid, operators, constants, state_template, dt, final_time):
    steps = int(round(final_time / dt))
    if not np.isclose(steps * dt, final_time):
        raise ValueError("final_time must be an integer multiple of dt")
    # No sponge is used: a per-step damping map would change the continuous
    # problem when the number of steps changes during temporal refinement.
    kwargs = dict(
        dt=dt, damp_height=1.0e9, max_damp=0.0, alpha=0.5,
        nu_div_factor=0.0, nu_h_factor=0.0,
    )
    if core == "split-explicit":
        kwargs["ns"] = 40
    else:
        kwargs.update(solver_tol=1e-10, solver_maxiter=60, solver_restart=60)

    def objective(theta):
        state = dict(state_template)
        state["th_v"] = theta
        stepper, actual_dt = build_dynamical_core(
            core_type=core,
            grid=grid,
            operators=operators,
            constants=constants,
            initial_state=state,
            **kwargs,
        )

        def advance(current, index):
            return (
                stepper.step(current, index * actual_dt, None, lambda value, _: value),
                None,
            )

        advance_fn = jax.checkpoint(advance) if core == "split-explicit" else advance
        final, _ = jax.lax.scan(advance_fn, state, jnp.arange(steps))
        w_mass = 0.5 * (final["w"][..., :-1] + final["w"][..., 1:])
        theta_prime = final["th_v"] - 300.0
        volume = grid.dx * grid.dy * grid.dz
        return 0.5 * jnp.sum(
            final["rho"] * w_mass**2 + 0.1 * theta_prime**2
        ) * volume

    return objective


def smooth_direction(grid):
    x, y, z = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_m, indexing="ij")
    lx = float(grid.nx * grid.dx)
    lz = float(grid.nz * grid.dz)
    direction = (
        jnp.cos(2.0 * jnp.pi * x / lx)
        * jnp.cos(jnp.pi * z / lz)
        * jnp.ones_like(y)
    )
    return direction / jnp.sqrt(jnp.mean(direction**2))


def taylor_test(objective, theta, direction, epsilons):
    value_and_grad = jax.jit(jax.value_and_grad(objective))
    value, gradient = value_and_grad(theta)
    directional = jnp.vdot(gradient, direction)
    evaluate = jax.jit(objective)
    remainders_zero = []
    remainders_first = []
    for epsilon in epsilons:
        perturbed = evaluate(theta + epsilon * direction)
        delta = perturbed - value
        remainders_zero.append(float(jnp.abs(delta)))
        remainders_first.append(float(jnp.abs(delta - epsilon * directional)))
    return np.asarray(gradient), remainders_zero, remainders_first


def observed_orders(errors):
    errors = np.asarray(errors)
    return np.log(errors[:-1] / errors[1:]) / np.log(2.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--name", default="default")
    parser.add_argument("--final-time", type=float, default=2.0)
    args = parser.parse_args()

    grid, operators, constants, state = build_case()
    direction = smooth_direction(grid)
    epsilons = np.asarray([2.0**-k for k in range(2, 13)])
    time_steps = [0.5, 0.25, 0.125, 0.0625]
    results = {}
    gradients_by_core = {}

    for core in ("split-explicit", "sisl"):
        print(f"\n[{core}] Taylor test")
        taylor_dt = 0.25 if core == "split-explicit" else 0.5
        objective = objective_function(
            core, grid, operators, constants, state, taylor_dt, args.final_time
        )
        _, remainder_zero, remainder_first = taylor_test(
            objective, state["th_v"], direction, epsilons
        )
        first_orders = observed_orders(remainder_zero)
        second_orders = observed_orders(remainder_first)
        for epsilon, r0, r1 in zip(epsilons, remainder_zero, remainder_first):
            print(f"  eps={epsilon:.3e}: R0={r0:.6e}, R1={r1:.6e}")

        print(f"[{core}] temporal gradient refinement")
        gradients = []
        for dt in time_steps:
            objective = objective_function(
                core, grid, operators, constants, state, dt, args.final_time
            )
            gradient = jax.jit(jax.grad(objective))(state["th_v"])
            gradient.block_until_ready()
            gradients.append(np.asarray(gradient))
            print(f"  dt={dt:g}: ||g||2={np.linalg.norm(gradients[-1]):.6e}")
        differences = [
            np.linalg.norm(gradients[index] - gradients[index + 1])
            for index in range(len(gradients) - 1)
        ]
        refinement_orders = observed_orders(differences)
        gradients_by_core[core] = gradients
        results[core] = {
            "taylor_dt": taylor_dt,
            "epsilons": epsilons.tolist(),
            "zero_order_remainder": remainder_zero,
            "first_order_remainder": remainder_first,
            "zero_order_observed_orders": first_orders.tolist(),
            "first_order_observed_orders": second_orders.tolist(),
            "time_steps": time_steps,
            "gradient_norms": [float(np.linalg.norm(g)) for g in gradients],
            "successive_gradient_differences": [float(v) for v in differences],
            "gradient_refinement_orders": refinement_orders.tolist(),
        }

    # Both gradients live on the same fixed grid, so direct cross-core norms
    # need no interpolation or gradient-density rescaling.
    cross_differences = []
    for index, _ in enumerate(time_steps):
        cross_differences.append(float(np.linalg.norm(
            gradients_by_core["split-explicit"][index]
            - gradients_by_core["sisl"][index]
        )))
    cross_orders = observed_orders(cross_differences)
    minimum_taylor_order = min(
        min(result["first_order_observed_orders"][:8])
        for result in results.values()
    )
    minimum_gradient_order = min(
        min(result["gradient_refinement_orders"])
        for result in results.values()
    )
    passed = minimum_taylor_order >= 1.9 and minimum_gradient_order >= 1.8

    summary = {
        "test": "adjoint_gradient_convergence",
        "grid": {"nx": grid.nx, "ny": grid.ny, "nz": grid.nz, "dx_m": grid.dx},
        "final_time_s": args.final_time,
        "gradient_definition": "derivative with respect to cell theta_v values",
        "results": results,
        "cross_core": {
            "time_steps": time_steps,
            "gradient_differences": cross_differences,
            "observed_orders": cross_orders.tolist(),
            "interpretation": (
                "Fixed-grid diagnostic only; temporal refinement approaches a "
                "nonzero spatial-discretization difference between the cores."
            ),
        },
        "acceptance": {
            "minimum_taylor_order": 1.9,
            "minimum_temporal_gradient_order": 1.8,
            "observed_minimum_taylor_order": float(minimum_taylor_order),
            "observed_minimum_temporal_gradient_order": float(
                minimum_gradient_order
            ),
            "passed": passed,
        },
    }
    layout = ArtifactLayout(
        kind="verification",
        case="adjoint_gradient_convergence",
        execution=args.name,
        output_root=args.output_root,
    ).create()
    path = layout.data / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved {path}")
    if not passed:
        raise SystemExit("Adjoint-gradient verification failed its acceptance criteria")


if __name__ == "__main__":
    main()
