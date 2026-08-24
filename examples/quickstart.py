"""Small CPU smoke simulation for a new Suêtes installation."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from suetes.shared.driver import Simulation
from suetes.slice2d.euler import VerticalSlice
from suetes.slice2d.geometry import StaggeredGrid
from suetes.slice2d.steppers import SISLStepper


CONSTANTS = {
    "g": 9.81,
    "cp": 1004.0,
    "cvd": 717.0,
    "Rd": 287.0,
    "p0": 100000.0,
}


def main():
    """Integrate a small rising thermal for two steps and check finiteness."""
    grid = StaggeredGrid(24, 12, 12_000.0, 6_000.0, h_func=lambda x: 0.0)
    grid.periodic_x = True
    core = VerticalSlice(grid, CONSTANTS, damp_height=5_000.0, N_bv=0.0)

    theta = 300.0
    pi = 1.0 - CONSTANTS["g"] * grid.Z_m / (CONSTANTS["cp"] * theta)
    rho = (
        CONSTANTS["p0"]
        / (CONSTANTS["Rd"] * theta)
        * pi ** (CONSTANTS["cvd"] / CONSTANTS["Rd"])
    )
    radius = jnp.sqrt((grid.X_m - 6_000.0) ** 2 + (grid.Z_m - 1_500.0) ** 2)
    perturbation = jnp.where(radius < 1_000.0, 0.5 * jnp.cos(0.5 * jnp.pi * radius / 1_000.0) ** 2, 0.0)
    state = {
        "u": jnp.zeros_like(grid.X_u),
        "w": jnp.zeros_like(grid.X_w),
        "rho": rho,
        "pi": pi,
        "th_v": theta + perturbation,
        "eta_dot": jnp.zeros_like(grid.X_w),
    }

    dt = 0.5
    stepper = SISLStepper(core, dt, solver_tol=1.0e-6)

    def boundary_conditions(next_state, forcing):
        del forcing
        next_state["w"] = next_state["w"].at[:, 0].set(0.0).at[:, -1].set(0.0)
        next_state["eta_dot"] = next_state["eta_dot"].at[:, 0].set(0.0).at[:, -1].set(0.0)
        return next_state

    def step(current_state, step_index):
        next_state = stepper.step(
            current_state,
            step_index * dt,
            forcing=None,
            bc_fn=boundary_conditions,
        )
        return next_state, jnp.max(jnp.abs(next_state["w"]))

    final_state = Simulation(step, dt).run(state, 0.0, 1.0, chunk_steps=2)
    finite = all(bool(jnp.all(jnp.isfinite(value))) for value in final_state.values())
    if not finite:
        raise RuntimeError("Quick-start simulation produced non-finite values")
    print(f"Suêtes quick start passed; max |w| = {float(jnp.max(jnp.abs(final_state['w']))):.6f} m s-1")


if __name__ == "__main__":
    main()
