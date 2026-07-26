#!/usr/bin/env python3
"""Run a coarse-resolution version of the ERF three-dimensional supercell."""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

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


CONSTANTS = {
    "g": 9.81,
    "cp": 1004.0,
    "Rd": 287.0,
    "cvd": 717.0,
    "p0": 100_000.0,
    "epsilon": 0.622,
}
DOMAIN_X = 150_000.0
DOMAIN_Y = 100_000.0
DOMAIN_Z = 24_000.0
DEFAULT_SNAPSHOTS = (1800.0, 3600.0, 5400.0, 7200.0)


def saturation_mixing_ratio(theta: float, exner: float) -> float:
    temperature = theta * exner
    pressure = CONSTANTS["p0"] * exner ** (CONSTANTS["cp"] / CONSTANTS["Rd"])
    saturation_pressure = 611.2 * np.exp(
        17.67 * (temperature - 273.15) / (temperature - 29.65)
    )
    return (
        CONSTANTS["epsilon"] * saturation_pressure
        / (pressure - (1.0 - CONSTANTS["epsilon"]) * saturation_pressure)
    )


def generate_wk_sounding(z_mass: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return dry theta, virtual theta, Exner pressure, and vapor."""
    z_tr = 12_000.0
    theta_0 = 300.0
    theta_tr = 343.0
    temperature_tr = 213.0

    def dry_theta(height):
        return np.where(
            height <= z_tr,
            theta_0 + (theta_tr - theta_0) * (height / z_tr) ** 1.25,
            theta_tr
            * np.exp(
                CONSTANTS["g"] * (height - z_tr)
                / (CONSTANTS["cp"] * temperature_tr)
            ),
        )

    def relative_humidity(height):
        return np.where(
            height <= z_tr,
            1.0 - 0.75 * (height / z_tr) ** 1.25,
            0.25,
        )

    levels = np.concatenate([[0.0], np.asarray(z_mass, dtype=np.float32)])
    theta = dry_theta(levels)
    rh = relative_humidity(levels)
    exner = np.empty_like(levels)
    vapor = np.empty_like(levels)
    theta_v = np.empty_like(levels)
    exner[0] = 1.0

    for k in range(levels.size):
        if k:
            dz = levels[k] - levels[k - 1]
            exner[k] = (
                exner[k - 1]
                - CONSTANTS["g"] * dz
                / (CONSTANTS["cp"] * theta_v[k - 1])
            )
        for _ in range(8):
            vapor[k] = min(
                rh[k] * saturation_mixing_ratio(theta[k], exner[k]), 0.014
            )
            theta_v[k] = theta[k] * (1.0 + 0.61 * vapor[k])
            if k:
                theta_v_mean = 0.5 * (theta_v[k - 1] + theta_v[k])
                exner[k] = (
                    exner[k - 1]
                    - CONSTANTS["g"] * dz
                    / (CONSTANTS["cp"] * theta_v_mean)
                )

    return tuple(
        values[1:].astype(np.float32)
        for values in (theta, theta_v, exner, vapor)
    )


class ConstantLaplacianDiffusion:
    """Constant second-order diffusion used in the published ERF setup."""

    def __init__(
        self,
        grid: RegionalGrid3D,
        diffusivity: float,
        dt: float,
        periodic_axes=(),
        tracer_diffusivity: float | None = None,
    ):
        self.grid = grid
        self.diffusivity = float(diffusivity)
        self.tracer_diffusivity = float(
            diffusivity if tracer_diffusivity is None else tracer_diffusivity
        )
        self.dt = float(dt)
        self.periodic_axes = frozenset(periodic_axes)

    def _second_difference(self, field, axis, spacing):
        padding = [(0, 0), (0, 0), (0, 0)]
        padding[axis] = (1, 1)
        mode = "wrap" if axis in self.periodic_axes else "edge"
        extended = jnp.pad(field, padding, mode=mode)
        if axis == 0:
            numerator = extended[2:] - 2.0 * field + extended[:-2]
        elif axis == 1:
            numerator = extended[:, 2:] - 2.0 * field + extended[:, :-2]
        else:
            numerator = extended[:, :, 2:] - 2.0 * field + extended[:, :, :-2]
        return numerator / spacing**2

    def _laplacian(self, field):
        return (
            self._second_difference(field, 0, self.grid.dx)
            + self._second_difference(field, 1, self.grid.dy)
            + self._second_difference(field, 2, self.grid.dz)
        )

    def get_tendencies(self, state, background):
        tendencies = {}
        for key in ("u", "v", "w", "th_v"):
            if key in state:
                field = (
                    state[key] - background["th_v"]
                    if key == "th_v" else state[key]
                )
                tendencies[key] = self.diffusivity * self._laplacian(field)
        return tendencies

    def apply_update(self, state):
        return {
            key: jnp.maximum(
                state[key]
                + self.dt
                * self.tracer_diffusivity
                * self._laplacian(state[key]),
                0.0,
            )
            for key in ("q", "q_c", "q_r")
        }


def parse_snapshots(value: str) -> tuple[float, ...]:
    snapshots = tuple(float(item) for item in value.split(",") if item.strip())
    if not snapshots or any(b <= a for a, b in zip(snapshots, snapshots[1:])):
        raise argparse.ArgumentTypeError("snapshot times must be increasing")
    return snapshots


def validate(args):
    if args.diffusivity < 0.0 or args.tracer_diffusivity < 0.0:
        raise ValueError("diffusivities must be non-negative")
    dimensions = (
        round(DOMAIN_X / args.dx),
        round(DOMAIN_Y / args.dx),
        round(DOMAIN_Z / args.dz),
    )
    for cells, spacing, length, name in zip(
        dimensions,
        (args.dx, args.dx, args.dz),
        (DOMAIN_X, DOMAIN_Y, DOMAIN_Z),
        ("x", "y", "z"),
    ):
        if not np.isclose(cells * spacing, length):
            raise ValueError(f"spacing must divide the {name}-domain exactly")
    if args.snapshots[-1] > args.t_end:
        raise ValueError("snapshot time exceeds t_end")
    if any(
        not any(np.isclose(value, snapshot) for snapshot in args.snapshots)
        for value in args.volume_times
    ):
        raise ValueError("volume-times must also appear in snapshots")
    for value in (*args.snapshots, args.chunk_seconds, args.t_end):
        if not np.isclose(round(value / args.dt) * args.dt, value):
            raise ValueError(f"{value:g} s must be an integer multiple of dt")
    if any(
        not np.isclose(round(value / args.chunk_seconds) * args.chunk_seconds, value)
        for value in args.snapshots
    ):
        raise ValueError("snapshot times must be multiples of chunk-seconds")
    return dimensions


def run(args):
    nx, ny, nz = validate(args)
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

    # ERF is periodic in x and open in y. Suetes currently has neither exact
    # boundary type, so only the remote y edges are weakly relaxed; the x
    # boundaries are left untouched. The storm remains far from all boundaries.
    if args.relaxation_cells:
        index = jnp.arange(ny)
        distance = jnp.minimum(index, ny - 1 - index)
        mask_y = jnp.where(
            distance < args.relaxation_cells,
            jnp.cos(
                0.5 * jnp.pi * distance / args.relaxation_cells
            ) ** 2,
            0.0,
        )[None, :, None]
    else:
        mask_y = jnp.zeros((1, ny, 1), dtype=jnp.float32)

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
        # The two x-face endpoints represent the same periodic u face.
        output["u"] = output["u"].at[-1, :, :].set(output["u"][0, :, :])
        return output

    physics = PhysicsSuite()
    diffusion = ConstantLaplacianDiffusion(
        grid,
        args.diffusivity,
        args.dt,
        periodic_axes=(0,),
        tracer_diffusivity=args.tracer_diffusivity,
    )
    physics.add_tendency_scheme(diffusion)
    physics.add_update_scheme(diffusion)
    physics.add_update_scheme(
        KesslerWarmRain(CONSTANTS, dt=args.dt, grid=grid)
    )
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

    state["accumulated_rain"] = jnp.zeros((nx, ny), dtype=jnp.float32)
    chunk_steps = round(args.chunk_seconds / actual_dt)

    def advance_chunk(current_state, start_step):
        def scan_step(carry, offset):
            next_state = stepper.step(
                carry,
                (start_step + offset) * actual_dt,
                forcing=None,
                bc_fn=boundary_conditions,
            )
            next_state["accumulated_rain"] = (
                carry["accumulated_rain"] + next_state["precip_step"]
            )
            return next_state, jnp.max(jnp.abs(next_state["w"]))

        return jax.lax.scan(scan_step, current_state, jnp.arange(chunk_steps))

    print(
        f"ERF coarse 3-D supercell: {nx}x{ny}x{nz} "
        f"({nx * ny * nz:,} cells), dx={args.dx:g} m, "
        f"dz={args.dz:g} m, dt={actual_dt:g} s"
    )
    print(
        f"Compiling {args.chunk_seconds:g}-s chunk "
        f"({chunk_steps} timesteps)..."
    )
    compiled = jax.jit(advance_chunk).lower(state, 0).compile()

    def physical_dry_theta(current):
        moisture_factor = (
            1.0
            + (1.0 / CONSTANTS["epsilon"] - 1.0) * current["q"]
            - current["q_c"]
            - current["q_r"]
        )
        return current["th_v"] / moisture_factor

    y_mid = ny // 2
    times = [0.0]
    theta_xz = [np.asarray(physical_dry_theta(state)[:, y_mid, :])]
    cloud_xz = [np.asarray(state["q_c"][:, y_mid, :])]
    rain_xz = [np.asarray(state["q_r"][:, y_mid, :])]
    surface_theta = [np.asarray(physical_dry_theta(state)[:, :, 0])]
    accumulated_rain = [np.asarray(state["accumulated_rain"])]
    maximum_cloud = [np.asarray(jnp.max(state["q_c"], axis=2))]
    maximum_rain = [np.asarray(jnp.max(state["q_r"], axis=2))]
    volume_times = []
    volume_cloud = []
    volume_rain = []

    current = state
    current_step = 0
    next_snapshot = iter(args.snapshots)
    target = next(next_snapshot)
    maximum_w = 0.0
    start_wall = time.perf_counter()
    while current_step * actual_dt < args.t_end:
        current, chunk_w = compiled(current, current_step)
        jax.tree_util.tree_leaves(current)[0].block_until_ready()
        current_step += chunk_steps
        simulation_time = current_step * actual_dt
        maximum_w = max(maximum_w, float(jnp.max(chunk_w)))
        bad_fields = [
            key for key in current
            if not bool(jnp.all(jnp.isfinite(current[key])))
        ]
        if bad_fields:
            raise FloatingPointError(
                f"Non-finite state by t={simulation_time:g} s in "
                + ", ".join(bad_fields)
            )
        if simulation_time + 1.0e-9 >= target:
            dry = physical_dry_theta(current)
            times.append(target)
            theta_xz.append(np.asarray(dry[:, y_mid, :]))
            cloud_xz.append(np.asarray(current["q_c"][:, y_mid, :]))
            rain_xz.append(np.asarray(current["q_r"][:, y_mid, :]))
            surface_theta.append(np.asarray(dry[:, :, 0]))
            accumulated_rain.append(np.asarray(current["accumulated_rain"]))
            maximum_cloud.append(np.asarray(jnp.max(current["q_c"], axis=2)))
            maximum_rain.append(np.asarray(jnp.max(current["q_r"], axis=2)))
            if any(np.isclose(target, value) for value in args.volume_times):
                volume_times.append(target)
                volume_cloud.append(np.asarray(current["q_c"]))
                volume_rain.append(np.asarray(current["q_r"]))
            print(
                f"  t={target:6.0f} s | "
                f"max|w|={float(jnp.max(jnp.abs(current['w']))):7.3f} m/s | "
                f"peak rain={float(jnp.max(current['accumulated_rain'])):7.2f} mm"
            )
            try:
                target = next(next_snapshot)
            except StopIteration:
                target = math.inf

    dataset = xr.Dataset(
        data_vars={
            "dry_potential_temperature_xz": (
                ("time", "x", "z"), np.stack(theta_xz)
            ),
            "environment_dry_potential_temperature": (
                ("x", "z"), np.broadcast_to(dry_theta_1d, (nx, nz))
            ),
            "cloud_water_xz": (
                ("time", "x", "z"), np.stack(cloud_xz)
            ),
            "rain_water_xz": (
                ("time", "x", "z"), np.stack(rain_xz)
            ),
            "surface_dry_potential_temperature": (
                ("time", "x", "y"), np.stack(surface_theta)
            ),
            "accumulated_rain": (
                ("time", "x", "y"), np.stack(accumulated_rain)
            ),
            "column_maximum_cloud_water": (
                ("time", "x", "y"), np.stack(maximum_cloud)
            ),
            "column_maximum_rain_water": (
                ("time", "x", "y"), np.stack(maximum_rain)
            ),
            "cloud_water_3d": (
                ("volume_time", "x", "y", "z"), np.stack(volume_cloud)
            ),
            "rain_water_3d": (
                ("volume_time", "x", "y", "z"), np.stack(volume_rain)
            ),
        },
        coords={
            "time": np.asarray(times, dtype=np.float32),
            "x": np.asarray(grid.x_m),
            "y": np.asarray(grid.y_m),
            "z": np.asarray(grid.z_m),
            "volume_time": np.asarray(volume_times, dtype=np.float32),
        },
        attrs={
            "benchmark": "coarse ERF three-dimensional moist supercell",
            "reference_resolution_m": 250.0,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "dx_m": args.dx,
            "dz_m": args.dz,
            "dt_s": args.dt,
            "precision": "float32",
            "diffusivity_m2_s": args.diffusivity,
            "tracer_diffusivity_m2_s": args.tracer_diffusivity,
            "acoustic_substeps": args.acoustic_substeps,
            "periodic_x": 1,
            "relaxation_cells_y": args.relaxation_cells,
            "boundary_note": (
                "Periodic x matches ERF; Suetes approximates ERF's open y "
                "boundaries with weak relaxation and its high-order top "
                "outflow with a rigid lid plus upper sponge."
            ),
            "maximum_vertical_velocity_m_s": maximum_w,
            "wall_time_s": time.perf_counter() - start_wall,
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = save_plot_dataset(
        dataset,
        args.output_dir / "artifact.nc",
        experiment="erf_squall_line_3d_coarse",
    )
    print(f"Saved {artifact}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/squall_line_3d")
    )
    parser.add_argument("--dx", type=float, default=500.0)
    parser.add_argument("--dz", type=float, default=500.0)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--t-end", type=float, default=7200.0)
    parser.add_argument(
        "--snapshots",
        type=parse_snapshots,
        default=DEFAULT_SNAPSHOTS,
        help="comma-separated snapshot times in seconds",
    )
    parser.add_argument(
        "--volume-times",
        type=parse_snapshots,
        default=(1800.0, 7200.0),
        help="snapshot times for which full 3-D cloud/rain fields are retained",
    )
    parser.add_argument("--diffusivity", type=float, default=33.33)
    parser.add_argument(
        "--tracer-diffusivity",
        type=float,
        default=33.33,
        help=(
            "Laplacian diffusivity for q_v, q_c, and q_r (m2 s-1); "
            "set to zero for the controlled no-tracer-diffusion experiment"
        ),
    )
    parser.add_argument("--acoustic-substeps", type=int, default=2)
    parser.add_argument("--divergence-damping", type=float, default=0.03)
    parser.add_argument("--relaxation-cells", type=int, default=10)
    parser.add_argument("--sponge-start", type=float, default=18_000.0)
    parser.add_argument("--sponge-strength", type=float, default=0.05)
    parser.add_argument("--chunk-seconds", type=float, default=50.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
