"""Shared data/model utilities for the manifest-driven PGF experiments."""

import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from suetes.physics.base import PhysicsSuite
from suetes.regional3d.boundaries import BenchmarkSponge
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.steppers import build_dynamical_core
from suetes.shared.transforms import NEUVECoordinate

NX, NY, NZ = 32, 12, 16
DX, DY, DZ = 500.0, 500.0, 1000.0
DT = 0.8
DEFAULT_STEPS = 500
CONSTANTS = {
    "g": 9.81, "cp": 1004.0, "cv": 717.0, "Rd": 287.0,
    "p0": 100000.0, "kappa": 287.0 / 1004.0, "cvd": 717.0,
}
GENERATOR_VERSION = 1


def parse_seeds(text):
    return [int(value) for value in text.split(",") if value.strip()]


def make_dataset(family, seeds, role, path=None):
    dataset = {
        "schema": "suetes-neuve-pgf-dataset-v1",
        "role": role,
        "terrain_family": family,
        "generator_version": GENERATOR_VERSION,
        "seeds": [int(seed) for seed in seeds],
        "grid": {"nx": NX, "ny": NY, "nz": NZ, "dx": DX, "dy": DY, "dz": DZ},
        "integration": {"dt": DT, "steps": DEFAULT_STEPS},
    }
    if not dataset["seeds"]:
        raise ValueError("A PGF dataset must contain at least one terrain seed")
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as stream:
            json.dump(dataset, stream, indent=2)
    return dataset


def load_dataset(path):
    with open(path) as stream:
        dataset = json.load(stream)
    if dataset.get("schema") != "suetes-neuve-pgf-dataset-v1":
        raise ValueError(f"Unsupported PGF dataset manifest: {path}")
    if dataset.get("generator_version") != GENERATOR_VERSION:
        raise ValueError("Terrain-generator version mismatch")
    return dataset


def ridge_terrain(seed):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 3)
    amplitudes = (
        1800.0 + jax.random.uniform(keys[0], minval=-150.0, maxval=150.0),
        2400.0 + jax.random.uniform(keys[1], minval=-150.0, maxval=150.0),
        1600.0 + jax.random.uniform(keys[2], minval=-150.0, maxval=150.0),
    )

    def terrain(x, y):
        del y
        return (
            amplitudes[0] * jnp.exp(-0.5 * ((x + 2500.0) / 750.0) ** 2)
            + amplitudes[1] * jnp.exp(-0.5 * (x / 850.0) ** 2)
            + amplitudes[2] * jnp.exp(-0.5 * ((x - 2500.0) / 800.0) ** 2)
        )
    return terrain


def random_3d_terrain(seed):
    keys = jax.random.split(jax.random.PRNGKey(seed), 6)
    count = 4
    amplitude = jax.random.uniform(keys[0], (count,), minval=900.0, maxval=2300.0)
    x0 = jax.random.uniform(keys[1], (count,), minval=-6000.0, maxval=6000.0)
    y0 = jax.random.uniform(keys[2], (count,), minval=-2200.0, maxval=2200.0)
    sx = jax.random.uniform(keys[3], (count,), minval=650.0, maxval=1800.0)
    sy = jax.random.uniform(keys[4], (count,), minval=650.0, maxval=1800.0)
    angle = jax.random.uniform(keys[5], (count,), minval=-jnp.pi, maxval=jnp.pi)

    def terrain(x, y):
        height = jnp.zeros_like(x)
        for peak in range(count):
            xr, yr = x - x0[peak], y - y0[peak]
            c, s = jnp.cos(angle[peak]), jnp.sin(angle[peak])
            xp, yp = c * xr + s * yr, -s * xr + c * yr
            height += amplitude[peak] * jnp.exp(
                -0.5 * ((xp / sx[peak]) ** 2 + (yp / sy[peak]) ** 2)
            )
        return 2800.0 * jnp.tanh(height / 2800.0)
    return terrain

def terrains_from_dataset(dataset):
    factory = {
        "ridge": ridge_terrain,
        "random3d": random_3d_terrain,
    }.get(dataset["terrain_family"])
    if factory is None:
        raise ValueError(f"Unknown terrain family: {dataset['terrain_family']}")
    return [factory(seed) for seed in dataset["seeds"]]


def physical_tke(state, grid, interior_width=4):
    u = 0.5 * (state["u"][:-1] + state["u"][1:])
    v = 0.5 * (state["v"][:, :-1] + state["v"][:, 1:])
    w = 0.5 * (state["w"][..., :-1] + state["w"][..., 1:])
    energy = 0.5 * (u**2 + v**2 + w**2)
    volume = DX * DY * grid.dz_m_full / grid.m_factors["m"][..., None] ** 2
    mask = jnp.zeros((NX, NY), dtype=volume.dtype)
    mask = mask.at[interior_width:NX-interior_width,
                   interior_width:NY-interior_width].set(1.0)
    weight = volume * mask[..., None]
    return jnp.sum(weight * energy) / jnp.sum(weight)


def grid_diagnostics(grid):
    zx = jnp.gradient(grid.Z_m, axis=0) / DX
    zy = jnp.gradient(grid.Z_m, axis=1) / DY
    first = int(0.4 * NZ)
    slope2 = jnp.mean(zx[..., first:]**2 + zy[..., first:]**2)
    zxx = jnp.gradient(zx, axis=0) / DX
    zyy = jnp.gradient(zy, axis=1) / DY
    curvature = jnp.mean((4000.0 * (zxx + zyy))**2)
    return jnp.min(grid.dz_m_full), slope2, curvature


def build_case(transform, terrain):
    grid = RegionalGrid3D(
        NX, NY, NZ, DX, DY, DZ, lat_center=45.0, lon_center=0.0,
        h_func=terrain, transform=transform,
    )
    operators = CGridOperator3D(grid)
    suite = PhysicsSuite()
    physics = Euler3D(
        grid, operators, CONSTANTS, dt=DT, N_bv=0.02,
        damp_height=12000.0, max_damp=0.3,
        nu_div_factor=0.0, nu_h_factor=0.0, physics_suite=suite,
    )
    rho = (
        physics.c["p0"] / (physics.c["Rd"] * physics.theta_bg)
        * physics.pi_bg ** (physics.c["cvd"] / physics.c["Rd"])
    )
    state = {
        "u": jnp.zeros_like(grid.Z_u), "v": jnp.zeros_like(grid.Z_v),
        "w": jnp.zeros((NX, NY, NZ + 1)),
        "eta_dot": jnp.zeros((NX, NY, NZ + 1)),
        "pi": physics.pi_bg, "rho": rho, "th_v": physics.theta_bg,
    }
    sponge = BenchmarkSponge(NX, NY, sponge_depth=4, axes=("x", "y"))

    def boundary(current, forcing=None):
        del forcing
        return sponge.blend(current, {
            "u": jnp.zeros_like(grid.Z_u), "v": jnp.zeros_like(grid.Z_v),
            "pi": physics.pi_bg, "rho": rho, "th_v": physics.theta_bg,
        })

    stepper, _ = build_dynamical_core(
        core_type="split-explicit", grid=grid, operators=operators,
        constants=CONSTANTS, initial_state=state, physics_suite=suite,
        dt=DT, ns=20, nu_div_factor=0.0, nu_h_factor=0.0,
        damp_height=12000.0, max_damp=0.3, N_bv=0.02,
    )
    return grid, state, stepper, boundary


def run_case(transform, terrain, steps=DEFAULT_STEPS):
    grid, state, stepper, boundary = build_case(transform, terrain)

    def scan(current, _):
        next_state = stepper.step(current, 0.0, None, boundary)
        return next_state, (
            physical_tke(next_state, grid),
            jnp.max(jnp.abs(next_state["u"])),
            jnp.max(jnp.abs(next_state["w"])),
        )
    final, series = jax.lax.scan(scan, state, jnp.arange(steps))
    minimum, slope2, curvature = grid_diagnostics(grid)
    return final, series, grid, (minimum, slope2, curvature)


def neuve_template(params=None):
    return NEUVECoordinate(
        params=params, hidden_dim=64, key_seed=42,
        condition_on_terrain=True, legacy_behavior=True,
    )


def save_neuve(path, params):
    flat, _ = jax.tree_util.tree_flatten(params)
    np.savez(path, *[np.asarray(value) for value in flat])


def load_neuve(path):
    template = neuve_template()
    data = np.load(path)
    arrays = [data[f"arr_{index}"] for index in range(len(data.files))]
    _, tree = jax.tree_util.tree_flatten(template.params)
    return template.with_params(jax.tree_util.tree_unflatten(
        tree, [jnp.asarray(value) for value in arrays]
    ))
