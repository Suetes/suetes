"""Focused regression tests for the general 3-D dynamical-core equations.

These tests intentionally cover configurations that the rising-bubble and flat
MMS benchmarks do not: nonlinear Exner continuity, rotation, map-curvature
terms, and complete-step density/tracer budgets.  Each test is small and can be
run independently, for example::

    pytest -q -s tests/test_core_formula_regressions.py::test_split_continuity_includes_theta_prime_flux
"""

import jax
import jax.numpy as jnp
import pytest

from suetes.regional3d.euler import Euler3D
from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.steppers import SISLStepper3D, SplitExplicitStepper3D


jax.config.update("jax_enable_x64", True)


CONSTANTS = {
    "g": 9.81,
    "cp": 1004.0,
    "Rd": 287.0,
    "cvd": 717.0,
    "p0": 100000.0,
}


class PassiveTracerSuite:
    """Minimal physics-suite interface used to activate a passive tracer."""

    tracer_keys = ["q_test"]

    def get_explicit_tendencies(self, state, bg, interior_mask=None, ml_params=None):
        return {}

    def apply_state_updates(self, state):
        return state


def _environment(
    dt=0.1,
    *,
    nx=8,
    ny=8,
    nz=5,
    dx=2000.0,
    dy=2000.0,
    dz=500.0,
    latitude=45.0,
    terrain=None,
    physics_suite=None,
):
    if terrain is None:
        terrain = lambda x, y: jnp.zeros_like(x + y)
    grid = RegionalGrid3D(
        nx, ny, nz, dx, dy, dz, latitude, 0.0, h_func=terrain
    )
    op = CGridOperator3D(grid)
    physics = Euler3D(
        grid,
        op,
        CONSTANTS,
        dt=dt,
        N_bv=0.0,
        damp_height=1.0e9,
        max_damp=0.0,
        nu_div_factor=0.0,
        nu_h_factor=0.0,
        physics_suite=physics_suite,
    )
    return grid, op, physics


def _rho_from_eos(physics, pi, th_v):
    c = physics.c
    return c["p0"] / (c["Rd"] * th_v) * pi ** (c["cvd"] / c["Rd"])


def _base_state(physics, u=0.0, v=0.0):
    grid = physics.grid
    th_v = physics.theta_bg
    pi = physics.pi_bg
    return {
        "u": jnp.full((grid.nx + 1, grid.ny, grid.nz), u, dtype=jnp.float64),
        "v": jnp.full((grid.nx, grid.ny + 1, grid.nz), v, dtype=jnp.float64),
        "w": jnp.zeros((grid.nx, grid.ny, grid.nz + 1), dtype=jnp.float64),
        "eta_dot": jnp.zeros((grid.nx, grid.ny, grid.nz + 1), dtype=jnp.float64),
        "pi": pi,
        "th_v": th_v,
        "rho": _rho_from_eos(physics, pi, th_v),
    }


def _background(physics):
    rho = _rho_from_eos(physics, physics.pi_bg, physics.theta_bg)
    return physics.precompute_bg(
        {"rho": rho, "pi": physics.pi_bg, "th_v": physics.theta_bg}
    )


def _identity_bc(state, forcing):
    return state


def _nonlinear_pi_tendency(physics, state, theta_prime, bg):
    """Independent discretization of the theta-prime Exner flux."""
    op, grid = physics.op, physics.grid
    theta_u = op.avg(theta_prime, axis=0, from_loc="m", to_loc="u")
    theta_v = op.avg(theta_prime, axis=1, from_loc="m", to_loc="v")
    theta_w = op.avg(theta_prime, axis=2, from_loc="m", to_loc="w")
    m_u = grid.m_factors["u"][..., None]
    m_v = grid.m_factors["v"][..., None]
    m_m = grid.m_factors["m"][..., None]
    fx = state["u"] * bg["rho_u"] * theta_u * bg["dz_u"] / m_u
    fy = state["v"] * bg["rho_v"] * theta_v * bg["dz_v"] / m_v
    fz = state["eta_dot"] * bg["dz_w_full"] * bg["rho_w"] * theta_w
    fz = fz.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
    dx = op.diff(fx, axis=0, from_loc="u", to_loc="m") / bg["dz_m_full"]
    dy = op.diff(fy, axis=1, from_loc="v", to_loc="m") / bg["dz_m_full"]
    dz = op.diff(fz, axis=2, from_loc="w", to_loc="m") * grid.dz / bg["dz_m_full"]
    return -bg["C_pi"] * (m_m * (dx + dy) + dz)


def test_split_continuity_includes_theta_prime_flux():
    """Split slow forcing must include div(rho * theta' * velocity)."""
    grid, _, physics = _environment(dt=0.2)
    stepper = SplitExplicitStepper3D(physics, dt=0.2, ns=2, alpha=0.5)
    state = _base_state(physics)

    x = jnp.arange(grid.nx + 1, dtype=jnp.float64)[:, None, None]
    z = jnp.arange(grid.nz, dtype=jnp.float64)[None, None, :]
    state["u"] = 3.0 + 0.15 * x + jnp.zeros_like(state["u"])
    theta_prime = (
        4.0
        + 0.3 * jnp.arange(grid.nx, dtype=jnp.float64)[:, None, None]
        + 0.2 * z
    )
    theta_prime = theta_prime + jnp.zeros_like(state["th_v"])
    state["th_v"] = physics.theta_bg + theta_prime
    state["rho"] = _rho_from_eos(physics, state["pi"], state["th_v"])

    bg = _background(physics)
    actual = stepper._compute_slow_tendencies(state, bg, {}, {}, None)["pi"]

    prime = {
        "u": state["u"], "v": state["v"], "w": state["w"],
        "eta_dot": state["eta_dot"], "pi": state["pi"] - physics.pi_bg,
        "th_v": state["th_v"],
        "th_v_prime_u": physics.op.avg(theta_prime, 0, "m", "u"),
        "th_v_prime_v": physics.op.avg(theta_prime, 1, "m", "v"),
        "th_v_prime_w": physics.op.avg(theta_prime, 2, "m", "w"),
    }
    linear = physics.get_tendencies(prime, bg, is_explicit=True)["pi"]
    nonlinear = _nonlinear_pi_tendency(physics, state, theta_prime, bg)
    expected = linear + nonlinear
    error = float(jnp.max(jnp.abs(actual - expected)))
    signal = float(jnp.max(jnp.abs(nonlinear)))
    print(f"nonlinear continuity signal={signal:.3e}, missing-term error={error:.3e}")
    assert signal > 1.0e-12
    assert error < 1.0e-11 * signal, "split-explicit Exner forcing omits theta-prime flux"


def test_sisl_nonlinear_pgf_does_not_rescale_coriolis():
    """A uniform theta perturbation cannot alter horizontal Coriolis rotation."""
    dt = 0.05
    _, _, physics = _environment(dt=dt, latitude=55.0)

    def advance(theta_offset):
        state = _base_state(physics, u=8.0, v=12.0)
        state["th_v"] = state["th_v"] + theta_offset
        state["rho"] = _rho_from_eos(physics, state["pi"], state["th_v"])
        stepper = SISLStepper3D(
            physics, dt, alpha=0.5, solver_tol=1.0e-11,
            solver_maxiter=80, solver_restart=40,
        )
        return stepper.step(state, 0.0, None, _identity_bc)

    reference = advance(0.0)
    warm = advance(30.0)
    # Exclude solver-locked lateral faces.
    du = warm["u"][1:-1] - reference["u"][1:-1]
    dv = warm["v"][:, 1:-1] - reference["v"][:, 1:-1]
    error = float(jnp.maximum(jnp.max(jnp.abs(du)), jnp.max(jnp.abs(dv))))
    coriolis_increment = float(jnp.max(jnp.abs(reference["u"] - 8.0)))
    print(f"Coriolis increment={coriolis_increment:.3e}, theta-dependent change={error:.3e}")
    assert coriolis_increment > 1.0e-8
    assert error < 1.0e-5 * coriolis_increment, (
        "SISL nonlinear PGF correction is contaminating the Coriolis increment"
    )


def test_sisl_applies_full_metric_curvature_tendency():
    """The leading-order SISL increment must contain the full metric tendency."""
    dt = 1.0e-3
    grid, _, physics = _environment(dt=dt, latitude=0.0)
    # Isolate the curvature formula from projection and Coriolis effects.
    for loc in ("m", "u", "v", "w"):
        grid.m_factors[loc] = jnp.ones_like(grid.m_factors[loc])
    grid.f_u = jnp.zeros_like(grid.f_u)
    grid.f_v = jnp.zeros_like(grid.f_v)
    grid.dm_dx_m = jnp.full_like(grid.dm_dx_m, 2.0e-4)
    grid.dm_dy_m = jnp.full_like(grid.dm_dy_m, -1.0e-4)

    state = _base_state(physics, u=10.0, v=6.0)
    theta_prime = state["th_v"] - physics.theta_bg
    bg = _background(physics)
    prime = {
        "u": state["u"], "v": state["v"], "w": state["w"],
        "eta_dot": state["eta_dot"], "pi": state["pi"] - physics.pi_bg,
        "th_v": state["th_v"],
        "th_v_prime_u": physics.op.avg(theta_prime, 0, "m", "u"),
        "th_v_prime_v": physics.op.avg(theta_prime, 1, "m", "v"),
        "th_v_prime_w": physics.op.avg(theta_prime, 2, "m", "w"),
    }
    expected = physics.get_tendencies(prime, bg, is_explicit=True)
    stepper = SISLStepper3D(
        physics, dt, alpha=0.5, solver_tol=1.0e-12,
        solver_maxiter=80, solver_restart=40,
    )
    result = stepper.step(state, 0.0, None, _identity_bc)
    numerical_u = (result["u"] - state["u"]) / dt
    numerical_v = (result["v"] - state["v"]) / dt
    # The imposed fields are uniform, so advection vanishes. Ignore locked faces.
    err_u = jnp.max(jnp.abs(numerical_u[1:-1] - expected["u"][1:-1]))
    err_v = jnp.max(jnp.abs(numerical_v[:, 1:-1] - expected["v"][:, 1:-1]))
    error = float(jnp.maximum(err_u, err_v))
    signal = float(jnp.maximum(jnp.max(jnp.abs(expected["u"])), jnp.max(jnp.abs(expected["v"]))))
    print(f"metric tendency={signal:.3e}, leading-order error={error:.3e}")
    assert signal > 1.0e-4
    assert error < 2.0e-3 * signal, "SISL underweights explicit metric curvature"


@pytest.mark.parametrize("core", ["sisl", "split_explicit"])
def test_complete_step_conserves_density_and_passive_tracer_mass(core):
    """Closed-box budgets must survive final EOS/tracer reconciliation."""
    dt = 0.02
    suite = PassiveTracerSuite()
    grid, _, physics = _environment(dt=dt, physics_suite=suite)
    state = _base_state(physics)

    i_u = jnp.arange(grid.nx + 1, dtype=jnp.float64)[:, None, None]
    j_v = jnp.arange(grid.ny + 1, dtype=jnp.float64)[None, :, None]
    state["u"] = 2.0 * jnp.sin(jnp.pi * i_u / grid.nx) + jnp.zeros_like(state["u"])
    state["v"] = -1.5 * jnp.sin(jnp.pi * j_v / grid.ny) + jnp.zeros_like(state["v"])
    x = jnp.arange(grid.nx, dtype=jnp.float64)[:, None, None]
    y = jnp.arange(grid.ny, dtype=jnp.float64)[None, :, None]
    state["q_test"] = 1.0 + 0.2 * jnp.sin(2.0 * jnp.pi * x / grid.nx) * jnp.cos(2.0 * jnp.pi * y / grid.ny)
    state["q_test"] = state["q_test"] + jnp.zeros_like(state["rho"])

    volumes = grid.dx * grid.dy * grid.dz_m_full / grid.m_factors["m"][..., None] ** 2
    mass0 = jnp.sum(state["rho"] * volumes, dtype=jnp.float64)
    tracer0 = jnp.sum(state["rho"] * state["q_test"] * volumes, dtype=jnp.float64)

    # Both steppers transport this tracer mass with their FFSL operator.  The
    # final rho*q field must reproduce that transported field even if rho is
    # subsequently diagnosed from the equation of state.
    bg = _background(physics)

    if core == "sisl":
        stepper = SISLStepper3D(
            physics, dt, alpha=0.5, solver_tol=1.0e-11,
            solver_maxiter=80, solver_restart=40,
        )
    else:
        stepper = SplitExplicitStepper3D(physics, dt, ns=2, alpha=0.5)
    transported_tracer_mass = stepper.ffsl_advector.advect_3d_split(
        state["rho"] * state["q_test"], state, bg
    )
    result = stepper.step(state, 0.0, None, _identity_bc)

    mass1 = jnp.sum(result["rho"] * volumes, dtype=jnp.float64)
    tracer1 = jnp.sum(result["rho"] * result["q_test"] * volumes, dtype=jnp.float64)
    rho_drift = float(jnp.abs(mass1 - mass0) / mass0)
    tracer_drift = float(jnp.abs(tracer1 - tracer0) / tracer0)
    local_tracer_error = float(
        jnp.linalg.norm(result["rho"] * result["q_test"] - transported_tracer_mass)
        / jnp.linalg.norm(transported_tracer_mass)
    )
    print(
        f"{core}: rho drift={rho_drift:.3e}, tracer drift={tracer_drift:.3e}, "
        f"local rho*q error={local_tracer_error:.3e}"
    )
    assert rho_drift < 1.0e-10, "complete core step does not conserve closed-box density"
    assert tracer_drift < 1.0e-10, "complete core step does not conserve passive tracer mass"
    assert local_tracer_error < 1.0e-12, (
        "final EOS density is inconsistent with the conservatively transported tracer mass"
    )
