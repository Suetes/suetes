#!/usr/bin/env python3
"""Run the ERF/WRF idealized two-dimensional moist squall-line benchmark."""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

# A million-cell state with three moist tracers and RK/acoustic work arrays does
# not coexist reliably with JAX's default large up-front GPU reservation.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

# ERF's published comparison is not bitwise, and Suetes' production GPU mode is
# single precision.  Float64 approximately doubles the live moist-state memory.
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
import numpy as np
import xarray as xr

from suetes.physics.base import PhysicsSuite
from suetes.physics.microphysics import KesslerWarmRain
from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.steppers import build_dynamical_core
from suetes.shared.artifacts import save_plot_dataset


CONSTANTS = {"g": 9.81, "cp": 1004.0, "Rd": 287.0, "cvd": 717.0, "p0": 100_000.0, "epsilon": 0.622}
DOMAIN_X = 150_000.0
DOMAIN_Z = 24_000.0
SNAPSHOT_TIMES = (3000.0, 6000.0, 9000.0)


def saturation_mixing_ratio(theta: float, exner: float) -> float:
    temperature = theta * exner
    pressure = CONSTANTS["p0"] * exner ** (CONSTANTS["cp"] / CONSTANTS["Rd"])
    saturation_pressure = 611.2 * np.exp(17.67 * (temperature - 273.15) / (temperature - 29.65))
    return CONSTANTS["epsilon"] * saturation_pressure / (pressure - (1.0 - CONSTANTS["epsilon"]) * saturation_pressure)


def generate_wk_sounding(z_mass: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return dry theta, virtual theta, Exner pressure, and vapor on mass levels."""
    z_tr = 12_000.0
    theta_0 = 300.0
    theta_tr = 343.0
    temperature_tr = 213.0

    def dry_theta(height):
        return np.where(
            height <= z_tr,
            theta_0 + (theta_tr - theta_0) * (height / z_tr) ** 1.25,
            theta_tr * np.exp(CONSTANTS["g"] * (height - z_tr) / (CONSTANTS["cp"] * temperature_tr)),
        )

    def relative_humidity(height):
        return np.where(height <= z_tr, 1.0 - 0.75 * (height / z_tr) ** 1.25, 0.25)

    # Include z=0 so Exner pressure is anchored at the physical surface,
    # rather than incorrectly assigning pi=1 at the first mass level.
    levels = np.concatenate([[0.0], np.asarray(z_mass, dtype=np.float32)])
    theta_all = dry_theta(levels)
    rh_all = relative_humidity(levels)
    exner_all = np.empty_like(levels)
    vapor_all = np.empty_like(levels)
    theta_v_all = np.empty_like(levels)
    exner_all[0] = 1.0

    for k in range(levels.size):
        if k:
            dz = levels[k] - levels[k - 1]
            exner_all[k] = exner_all[k - 1] - CONSTANTS["g"] * dz / (CONSTANTS["cp"] * theta_v_all[k - 1])
        for _ in range(8):
            q_sat = saturation_mixing_ratio(theta_all[k], exner_all[k])
            vapor_all[k] = min(rh_all[k] * q_sat, 0.014)
            theta_v_all[k] = theta_all[k] * (1.0 + 0.61 * vapor_all[k])
            if k:
                theta_v_mean = 0.5 * (theta_v_all[k - 1] + theta_v_all[k])
                exner_all[k] = exner_all[k - 1] - CONSTANTS["g"] * dz / (CONSTANTS["cp"] * theta_v_mean)

    return tuple(values[1:].astype(np.float32) for values in (theta_all, theta_v_all, exner_all, vapor_all))


class ConstantLaplacianDiffusion:
    """ERF benchmark's constant second-order diffusion for resolved fields."""

    def __init__(self, grid: RegionalGrid3D, diffusivity: float, dt: float):
        self.grid = grid
        self.diffusivity = float(diffusivity)
        self.dt = float(dt)

    @staticmethod
    def _second_difference(field, axis, spacing):
        padding = [(0, 0), (0, 0), (0, 0)]
        padding[axis] = (1, 1)
        extended = jnp.pad(field, padding, mode="edge")
        if axis == 0:
            numerator = extended[2:] - 2.0 * field + extended[:-2]
        elif axis == 1:
            numerator = extended[:, 2:] - 2.0 * field + extended[:, :-2]
        else:
            numerator = extended[:, :, 2:] - 2.0 * field + extended[:, :, :-2]
        return numerator / spacing**2

    def get_tendencies(self, state, bg):
        tendencies = {}
        for key in ("u", "v", "w", "th_v"):
            if key not in state:
                continue
            field = state[key]
            # The benchmark diffuses the thermodynamic perturbation, leaving
            # the prescribed hydrostatic sounding stationary.
            if key == "th_v":
                field = field - bg["th_v"]
            laplacian = (
                self._second_difference(field, 0, self.grid.dx)
                + self._second_difference(field, 1, self.grid.dy)
                + self._second_difference(field, 2, self.grid.dz)
            )
            tendencies[key] = self.diffusivity * laplacian
        return tendencies

    def apply_update(self, state):
        """Apply the same constant diffusion to moisture after FFSL transport."""
        updates = {}
        for key in ("q", "q_c", "q_r"):
            field = state[key]
            laplacian = (
                self._second_difference(field, 0, self.grid.dx)
                + self._second_difference(field, 1, self.grid.dy)
                + self._second_difference(field, 2, self.grid.dz)
            )
            updates[key] = jnp.maximum(field + self.dt * self.diffusivity * laplacian, 0.0)
        return updates


def validate_discretization(args):
    nx = round(DOMAIN_X / args.dx)
    nz = round(DOMAIN_Z / args.dz)
    if not np.isclose(nx * args.dx, DOMAIN_X):
        raise ValueError("dx must divide the 150-km domain exactly")
    if not np.isclose(nz * args.dz, DOMAIN_Z):
        raise ValueError("dz must divide the 24-km domain exactly")
    for value in (*SNAPSHOT_TIMES, args.chunk_seconds):
        if not np.isclose(round(value / args.dt) * args.dt, value):
            raise ValueError(f"{value:g} s must be an integer multiple of dt")
    return nx, nz


def run(args):
    nx, nz = validate_discretization(args)
    ny = args.ny
    grid = RegionalGrid3D(nx, ny, nz, args.dx, args.dx, args.dz, lat_center=0.0, lon_center=0.0)
    operators = CGridOperator3D(grid)
    theta, theta_v, exner, vapor = generate_wk_sounding(np.asarray(grid.z_m))

    theta_bg = jnp.broadcast_to(theta_v, (nx, ny, nz))
    exner_bg = jnp.broadcast_to(exner, (nx, ny, nz))
    vapor_bg = jnp.broadcast_to(vapor, (nx, ny, nz))
    rho_bg = CONSTANTS["p0"] / (CONSTANTS["Rd"] * theta_bg) * exner_bg ** (CONSTANTS["cvd"] / CONSTANTS["Rd"])
    u_profile = -12.0 + 4.8e-3 * jnp.minimum(grid.z_m, 2500.0)
    u_bg = jnp.broadcast_to(u_profile, (nx + 1, ny, nz))

    x, _, z = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_m, indexing="ij")
    radius = jnp.sqrt((x / 10_000.0) ** 2 + ((z - 2000.0) / 1500.0) ** 2)
    bubble = jnp.where(radius < 1.0, 3.0 * jnp.cos(0.5 * jnp.pi * radius) ** 2, 0.0)
    dry_theta = jnp.broadcast_to(theta, (nx, ny, nz)) + bubble
    initial_theta_v = dry_theta * (1.0 + 0.61 * vapor_bg)
    initial_rho = (
        CONSTANTS["p0"] / (CONSTANTS["Rd"] * initial_theta_v) * exner_bg ** (CONSTANTS["cvd"] / CONSTANTS["Rd"])
    )
    state = {
        "u": u_bg,
        "v": jnp.zeros((nx, ny + 1, nz)),
        "w": jnp.zeros((nx, ny, nz + 1)),
        "eta_dot": jnp.zeros((nx, ny, nz + 1)),
        "pi": exner_bg,
        "rho": initial_rho,
        "th_v": initial_theta_v,
        "q": vapor_bg,
        "q_c": jnp.zeros((nx, ny, nz)),
        "q_r": jnp.zeros((nx, ny, nz)),
        "precip_step": jnp.zeros((nx, ny)),
    }
    background_state = {"th_v": theta_bg, "pi": exner_bg, "rho": rho_bg}

    relaxation_cells = args.relaxation_cells
    if relaxation_cells:
        index = jnp.arange(nx)
        distance = jnp.minimum(index, nx - 1 - index)
        mask_x = jnp.where(distance < relaxation_cells, jnp.cos(0.5 * jnp.pi * distance / relaxation_cells) ** 2, 0.0)[
            :, None, None
        ]
    else:
        mask_x = jnp.zeros((nx, 1, 1))

    def boundary_conditions(current, forcing):
        del forcing
        output = {}
        references = {
            "u": u_bg,
            "v": jnp.zeros_like(current["v"]),
            "th_v": theta_bg,
            "pi": exner_bg,
            "rho": rho_bg,
            "q": vapor_bg,
            "q_c": jnp.zeros_like(current["q_c"]),
            "q_r": jnp.zeros_like(current["q_r"]),
        }
        for key, value in current.items():
            if key not in references:
                output[key] = value
                continue
            mask = jnp.pad(mask_x, ((0, 1), (0, 0), (0, 0)), mode="edge") if key == "u" else mask_x
            output[key] = (1.0 - mask) * value + mask * references[key]
        return output

    physics = PhysicsSuite()
    diffusion = ConstantLaplacianDiffusion(grid, args.diffusivity, args.dt)
    physics.add_tendency_scheme(diffusion)
    physics.add_update_scheme(diffusion)
    physics.add_update_scheme(KesslerWarmRain(CONSTANTS, dt=args.dt, grid=grid))
    for tracer in ("q", "q_c", "q_r"):
        physics.register_tracer(tracer)

    stepper, actual_dt = build_dynamical_core(
        core_type="split-explicit",
        grid=grid,
        operators=operators,
        constants=CONSTANTS,
        initial_state=state,
        physics_suite=physics,
        dt=args.dt,
        ns=args.acoustic_substeps,
        alpha=0.5,
        nu_h_factor=0.0,
        nu_div_factor=args.divergence_damping,
        damp_height=args.sponge_start,
        max_damp=args.sponge_strength,
        background_state=background_state,
    )
    if not np.isclose(actual_dt, args.dt):
        raise RuntimeError(f"Requested dt={args.dt}, core returned {actual_dt}")

    state["accumulated_rain"] = jnp.zeros((nx, ny), dtype=state["th_v"].dtype)
    chunk_steps = round(args.chunk_seconds / actual_dt)

    def advance_chunk(current_state, start_step):
        def scan_step(carry, offset):
            next_state = stepper.step(carry, (start_step + offset) * actual_dt, forcing=None, bc_fn=boundary_conditions)
            next_state["accumulated_rain"] = carry["accumulated_rain"] + next_state["precip_step"]
            return next_state, jnp.max(jnp.abs(next_state["w"]))

        return jax.lax.scan(scan_step, current_state, jnp.arange(chunk_steps))

    advance_chunk = jax.jit(advance_chunk)
    print(f"ERF squall line: {nx}x{ny}x{nz}, dx={args.dx:g} m, dz={args.dz:g} m, dt={actual_dt:g} s")
    print(f"Compiling {args.chunk_seconds:g}-s chunk ({chunk_steps} timesteps)...")
    compiled = advance_chunk.lower(state, 0).compile()

    y_mid = ny // 2

    def dry_potential_temperature(current_state):
        moisture_factor = (
            1.0 + (1.0 / CONSTANTS["epsilon"] - 1.0) * current_state["q"] - current_state["q_c"] - current_state["q_r"]
        )
        return current_state["th_v"] / moisture_factor

    times = [0.0]
    theta_snapshots = [np.asarray(dry_potential_temperature(state)[:, y_mid, :])]
    cloud_snapshots = [np.asarray(state["q_c"][:, y_mid, :])]
    rain_snapshots = [np.asarray(state["q_r"][:, y_mid, :])]
    accumulation_snapshots = [np.asarray(jnp.mean(state["accumulated_rain"], axis=1))]

    current = state
    current_step = 0
    start_wall = time.perf_counter()
    next_snapshot = iter(SNAPSHOT_TIMES)
    target = next(next_snapshot)
    max_w_over_run = 0.0
    while current_step * actual_dt < SNAPSHOT_TIMES[-1]:
        current, chunk_max_w = compiled(current, current_step)
        jax.tree_util.tree_leaves(current)[0].block_until_ready()
        current_step += chunk_steps
        simulation_time = current_step * actual_dt
        chunk_max_w_value = float(jnp.max(chunk_max_w))
        field_names = tuple(current)
        finite_flags = np.asarray(jnp.stack([jnp.all(jnp.isfinite(current[key])) for key in field_names]))
        bad_fields = [key for key, is_finite in zip(field_names, finite_flags) if not is_finite]
        if not np.isfinite(chunk_max_w_value) or bad_fields:
            fields = ", ".join(bad_fields) if bad_fields else "w"
            raise FloatingPointError(
                f"Non-finite state first detected by t={simulation_time:g} s "
                f"in: {fields}. Increase --acoustic-substeps or reduce --dt."
            )
        max_w_over_run = max(max_w_over_run, chunk_max_w_value)
        if simulation_time + 1.0e-9 >= target:
            times.append(target)
            theta_snapshots.append(np.asarray(dry_potential_temperature(current)[:, y_mid, :]))
            cloud_snapshots.append(np.asarray(current["q_c"][:, y_mid, :]))
            rain_snapshots.append(np.asarray(current["q_r"][:, y_mid, :]))
            accumulation_snapshots.append(np.asarray(jnp.mean(current["accumulated_rain"], axis=1)))
            current_max_w = float(jnp.max(jnp.abs(current["w"])))
            print(
                f"  t={target:6.0f} s | max|w|={current_max_w:7.3f} m/s "
                f"| peak rain={float(jnp.max(current['accumulated_rain'])):7.2f} mm"
            )
            try:
                target = next(next_snapshot)
            except StopIteration:
                target = math.inf

    finite = all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree_util.tree_leaves(current))
    if not finite:
        raise FloatingPointError("ERF squall-line benchmark produced non-finite values")

    dataset = xr.Dataset(
        data_vars={
            "dry_potential_temperature": (("time", "x", "z"), np.stack(theta_snapshots)),
            "environment_dry_potential_temperature": (("x", "z"), np.broadcast_to(theta, (nx, nz))),
            "cloud_water": (("time", "x", "z"), np.stack(cloud_snapshots)),
            "rain_water": (("time", "x", "z"), np.stack(rain_snapshots)),
            "accumulated_rain": (("time", "x"), np.stack(accumulation_snapshots)),
        },
        coords={"time": np.asarray(times), "x": np.asarray(grid.x_m), "z": np.asarray(grid.z_m)},
        attrs={
            "benchmark": "ERF idealized two-dimensional squall line",
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "dx_m": args.dx,
            "dz_m": args.dz,
            "dt_s": args.dt,
            "precision": "float32",
            "diffusivity_m2_s": args.diffusivity,
            "acoustic_substeps": args.acoustic_substeps,
            "relaxation_cells": args.relaxation_cells,
            "maximum_vertical_velocity_m_s": max_w_over_run,
            "wall_time_s": time.perf_counter() - start_wall,
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = save_plot_dataset(dataset, args.output_dir / "artifact.nc", experiment="erf_squall_line_2d")
    print(f"Saved {artifact}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/erf_squall_line_2d"))
    parser.add_argument("--dx", type=float, default=100.0)
    parser.add_argument("--dz", type=float, default=100.0)
    parser.add_argument("--ny", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--diffusivity", type=float, default=200.0)
    # ERF uses dt=0.25 s for its RK integrator. Suetes retains that outer step,
    # but its fractional-step acoustic integration needs subcycling at 100 m.
    parser.add_argument("--acoustic-substeps", type=int, default=2)
    parser.add_argument("--divergence-damping", type=float, default=0.03)
    parser.add_argument("--relaxation-cells", type=int, default=20)
    parser.add_argument("--sponge-start", type=float, default=18_000.0)
    parser.add_argument("--sponge-strength", type=float, default=0.05)
    parser.add_argument("--chunk-seconds", type=float, default=100.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
