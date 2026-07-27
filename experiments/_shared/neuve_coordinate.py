"""Shared utilities for manifest-driven neural-coordinate experiments."""

# This module is shared by the NEUVE case and its regression tests.

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
from suetes.regional3d.steppers import FluxFormAdvector
from suetes.shared.transforms import NEUVECoordinate

NX, NY, NZ = 32, 12, 16
DX, DY, DZ = 500.0, 500.0, 1000.0
DT = 0.8
DEFAULT_STEPS = 500
MOUNTAIN_DT = 2.0
MOUNTAIN_STEPS = 200
TARGET_DEFAULTS = {
    "pgf_rest": {"dt": DT, "steps": DEFAULT_STEPS},
    "tracer_reversibility": {"dt": 2.0, "steps": 100},
    "mountain_flux": {"dt": MOUNTAIN_DT, "steps": MOUNTAIN_STEPS},
}
CONSTANTS = {"g": 9.81, "cp": 1004.0, "cv": 717.0, "Rd": 287.0, "p0": 100000.0, "kappa": 287.0 / 1004.0, "cvd": 717.0}
GENERATOR_VERSION = 1


def parse_seeds(text):
    return [int(value) for value in text.split(",") if value.strip()]


def make_dataset(family, seeds, role, path=None, target="pgf_rest", steps=None):
    if steps is None:
        steps = TARGET_DEFAULTS[target]["steps"]
    dataset = {
        "schema": "suetes-neuve-pgf-dataset-v1",
        "role": role,
        "target": target,
        "terrain_family": family,
        "generator_version": GENERATOR_VERSION,
        "seeds": [int(seed) for seed in seeds],
        "grid": {"nx": NX, "ny": NY, "nz": NZ, "dx": DX, "dy": DY, "dz": DZ},
        "integration": {"dt": TARGET_DEFAULTS[target]["dt"], "steps": int(steps)},
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


def dataset_target(dataset):
    """Return the target, retaining compatibility with existing PGF manifests."""
    return dataset.get("target", "pgf_rest")


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
            height += amplitude[peak] * jnp.exp(-0.5 * ((xp / sx[peak]) ** 2 + (yp / sy[peak]) ** 2))
        return 2800.0 * jnp.tanh(height / 2800.0)

    return terrain


def multiscale_3d_terrain(seed):
    """Terrain mixing broad massifs with independently located narrow peaks."""
    keys = jax.random.split(jax.random.PRNGKey(seed), 12)
    broad_count, narrow_count = 2, 4
    broad_amplitude = jax.random.uniform(keys[0], (broad_count,), minval=900.0, maxval=1800.0)
    broad_x = jax.random.uniform(keys[1], (broad_count,), minval=-4500.0, maxval=4500.0)
    broad_y = jax.random.uniform(keys[2], (broad_count,), minval=-1400.0, maxval=1400.0)
    broad_sx = jax.random.uniform(keys[3], (broad_count,), minval=2500.0, maxval=4200.0)
    broad_sy = jax.random.uniform(keys[4], (broad_count,), minval=1700.0, maxval=3000.0)
    broad_angle = jax.random.uniform(keys[5], (broad_count,), minval=-jnp.pi, maxval=jnp.pi)
    narrow_amplitude = jax.random.uniform(keys[6], (narrow_count,), minval=500.0, maxval=1300.0)
    narrow_x = jax.random.uniform(keys[7], (narrow_count,), minval=-5500.0, maxval=5500.0)
    narrow_y = jax.random.uniform(keys[8], (narrow_count,), minval=-1900.0, maxval=1900.0)
    # Keep every feature resolved by at least roughly four horizontal cells.
    narrow_sx = jax.random.uniform(keys[9], (narrow_count,), minval=900.0, maxval=1500.0)
    narrow_sy = jax.random.uniform(keys[10], (narrow_count,), minval=900.0, maxval=1500.0)
    narrow_angle = jax.random.uniform(keys[11], (narrow_count,), minval=-jnp.pi, maxval=jnp.pi)

    def add_features(height, x, y, amplitude, x0, y0, sx, sy, angle):
        for feature in range(amplitude.shape[0]):
            xr, yr = x - x0[feature], y - y0[feature]
            c, s = jnp.cos(angle[feature]), jnp.sin(angle[feature])
            xp, yp = c * xr + s * yr, -s * xr + c * yr
            height += amplitude[feature] * jnp.exp(-0.5 * ((xp / sx[feature]) ** 2 + (yp / sy[feature]) ** 2))
        return height

    def terrain(x, y):
        height = add_features(
            jnp.zeros_like(x), x, y, broad_amplitude, broad_x, broad_y, broad_sx, broad_sy, broad_angle
        )
        height = add_features(height, x, y, narrow_amplitude, narrow_x, narrow_y, narrow_sx, narrow_sy, narrow_angle)
        return 2800.0 * jnp.tanh(height / 2800.0)

    return terrain


def terrains_from_dataset(dataset):
    factory = {"ridge": ridge_terrain, "random3d": random_3d_terrain, "multiscale3d": multiscale_3d_terrain}.get(
        dataset["terrain_family"]
    )
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
    mask = mask.at[interior_width : NX - interior_width, interior_width : NY - interior_width].set(1.0)
    weight = volume * mask[..., None]
    return jnp.sum(weight * energy) / jnp.sum(weight)


def grid_diagnostics(grid):
    zx = jnp.gradient(grid.Z_m, axis=0) / DX
    zy = jnp.gradient(grid.Z_m, axis=1) / DY
    first = int(0.4 * NZ)
    slope2 = jnp.mean(zx[..., first:] ** 2 + zy[..., first:] ** 2)
    zxx = jnp.gradient(zx, axis=0) / DX
    zyy = jnp.gradient(zy, axis=1) / DY
    curvature = jnp.mean((4000.0 * (zxx + zyy)) ** 2)
    return jnp.min(grid.dz_m_full), slope2, curvature


def build_case(transform, terrain):
    grid = RegionalGrid3D(NX, NY, NZ, DX, DY, DZ, lat_center=45.0, lon_center=0.0, h_func=terrain, transform=transform)
    operators = CGridOperator3D(grid)
    suite = PhysicsSuite()
    physics = Euler3D(
        grid,
        operators,
        CONSTANTS,
        dt=DT,
        N_bv=0.02,
        damp_height=12000.0,
        max_damp=0.3,
        nu_div_factor=0.0,
        nu_h_factor=0.0,
        physics_suite=suite,
    )
    rho = physics.c["p0"] / (physics.c["Rd"] * physics.theta_bg) * physics.pi_bg ** (physics.c["cvd"] / physics.c["Rd"])
    state = {
        "u": jnp.zeros_like(grid.Z_u),
        "v": jnp.zeros_like(grid.Z_v),
        "w": jnp.zeros((NX, NY, NZ + 1)),
        "eta_dot": jnp.zeros((NX, NY, NZ + 1)),
        "pi": physics.pi_bg,
        "rho": rho,
        "th_v": physics.theta_bg,
    }
    sponge = BenchmarkSponge(NX, NY, sponge_depth=4, axes=("x", "y"))

    def boundary(current, forcing=None):
        del forcing
        return sponge.blend(
            current,
            {
                "u": jnp.zeros_like(grid.Z_u),
                "v": jnp.zeros_like(grid.Z_v),
                "pi": physics.pi_bg,
                "rho": rho,
                "th_v": physics.theta_bg,
            },
        )

    stepper, _ = build_dynamical_core(
        core_type="split-explicit",
        grid=grid,
        operators=operators,
        constants=CONSTANTS,
        initial_state=state,
        physics_suite=suite,
        dt=DT,
        ns=20,
        nu_div_factor=0.0,
        nu_h_factor=0.0,
        damp_height=12000.0,
        max_damp=0.3,
        N_bv=0.02,
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


def build_mountain_wave_case(transform, terrain):
    """Construct a compact stratified mountain-wave experiment."""
    grid = RegionalGrid3D(NX, NY, NZ, DX, DY, DZ, lat_center=45.0, lon_center=0.0, h_func=terrain, transform=transform)
    operators = CGridOperator3D(grid)
    suite = PhysicsSuite()
    wind = 10.0
    n_bv = 0.01
    physics = Euler3D(
        grid,
        operators,
        CONSTANTS,
        dt=MOUNTAIN_DT,
        N_bv=n_bv,
        damp_height=12000.0,
        max_damp=0.3,
        nu_div_factor=0.05,
        nu_h_factor=0.05,
        physics_suite=suite,
    )
    rho = physics.c["p0"] / (physics.c["Rd"] * physics.theta_bg) * physics.pi_bg ** (physics.c["cvd"] / physics.c["Rd"])
    state = {
        "u": wind * jnp.ones_like(grid.Z_u),
        "v": jnp.zeros_like(grid.Z_v),
        "w": jnp.zeros_like(grid.Z_w),
        "eta_dot": jnp.zeros_like(grid.Z_w),
        "pi": physics.pi_bg,
        "rho": rho,
        "th_v": physics.theta_bg,
    }
    sponge = BenchmarkSponge(NX, NY, sponge_depth=4, axes=("x", "y"), blend_vars=["u", "v", "th_v", "pi", "rho"])
    exterior = {
        "u": wind * jnp.ones_like(grid.Z_u),
        "v": jnp.zeros_like(grid.Z_v),
        "pi": physics.pi_bg,
        "rho": rho,
        "th_v": physics.theta_bg,
    }

    def boundary(current, forcing=None):
        del forcing
        return sponge.blend(current, exterior)

    stepper, _ = build_dynamical_core(
        core_type="split-explicit",
        grid=grid,
        operators=operators,
        constants=CONSTANTS,
        initial_state=state,
        physics_suite=suite,
        dt=MOUNTAIN_DT,
        ns=8,
        nu_div_factor=0.05,
        nu_h_factor=0.05,
        damp_height=12000.0,
        max_damp=0.3,
        N_bv=n_bv,
    )
    return grid, state, stepper, boundary, wind


def mountain_flux_consistency(transform, terrain, steps=MOUNTAIN_STEPS):
    """Measure vertical non-uniformity of resolved mountain-wave momentum flux."""
    grid, state, stepper, boundary, wind = build_mountain_wave_case(transform, terrain)
    horizontal_mask = jnp.zeros((NX, NY), dtype=grid.Z_m.dtype)
    horizontal_mask = horizontal_mask.at[4:-4, 4:-4].set(1.0)
    horizontal_area = DX * DY / grid.m_factors["m"] ** 2 * horizontal_mask
    # Exclude terrain-adjacent levels and the upper sponge.  Index bounds are
    # fixed so the objective cannot improve merely by moving its sampling mask.
    level_slice = slice(4, 12)

    def momentum_flux_profile(current):
        u = 0.5 * (current["u"][:-1] + current["u"][1:])
        w = 0.5 * (current["w"][..., :-1] + current["w"][..., 1:])
        return jnp.sum(horizontal_area[..., None] * current["rho"] * (u - wind) * w, axis=(0, 1))[level_slice]

    def flux_transmission_metric(flux):
        # Levels 4--5 provide the incident lower-tropospheric flux; levels
        # 8--11 measure its transmission into the upper troposphere.
        lower_reference = jnp.mean(flux[:2])
        upper = flux[4:]
        mismatch = jnp.mean((upper - lower_reference) ** 2)
        # Retain a small profile-energy contribution so a transient crossing
        # through zero cannot produce an ill-conditioned normalization.
        scale = lower_reference**2 + 0.05 * jnp.mean(flux**2) + 1.0e-12
        return mismatch / scale

    def scan_step(current, step):
        next_state = stepper.step(current, step * MOUNTAIN_DT, None, boundary)
        return next_state, momentum_flux_profile(next_state)

    # Reverse-mode differentiation through an uncheckpointed scan retains
    # hundreds of split-explicit step residuals.  Recompute each outer step in
    # the backward pass instead; this trades runtime for bounded GPU memory.
    checkpointed_step = jax.checkpoint(scan_step)
    final, flux_history = jax.lax.scan(checkpointed_step, state, jnp.arange(steps))
    del final
    spinup = steps // 2
    post_spinup_flux = flux_history[spinup:]
    count = jnp.arange(1, post_spinup_flux.shape[0] + 1, dtype=post_spinup_flux.dtype)[:, None]
    cumulative_mean_flux = jnp.cumsum(post_spinup_flux, axis=0) / count
    post_spinup_history = jax.vmap(flux_transmission_metric)(cumulative_mean_flux)
    history = jnp.full((steps,), jnp.nan, dtype=post_spinup_history.dtype)
    history = history.at[spinup:].set(post_spinup_history)
    time_mean_flux = cumulative_mean_flux[-1]
    objective = post_spinup_history[-1]
    flux_rms = jnp.sqrt(jnp.mean(time_mean_flux**2))
    minimum, slope2, curvature = grid_diagnostics(grid)
    return (objective, history, grid, (minimum, slope2, curvature, flux_rms))


def build_transport_case(transform, terrain, dt=2.0):
    """Construct a reversible, terrain-tangent prescribed transport problem."""
    grid = RegionalGrid3D(NX, NY, NZ, DX, DY, DZ, lat_center=45.0, lon_center=0.0, h_func=terrain, transform=transform)
    advector = FluxFormAdvector(grid, dt)
    speed = 12.0

    def envelope(z):
        return jnp.cos(0.5 * jnp.pi * z / grid.Lz) ** 2

    def envelope_derivative(z):
        return -0.5 * jnp.pi / grid.Lz * jnp.sin(jnp.pi * z / grid.Lz)

    # psi=U A(x)(z-h)g(z) gives a divergence-free x-z flow.  The
    # terrain, lid, and both lateral boundaries are all streamlines.
    half_width = 0.5 * NX * DX

    def lateral_envelope(x):
        return jnp.cos(0.5 * jnp.pi * x / half_width) ** 2

    def lateral_envelope_derivative(x):
        return -0.5 * jnp.pi / half_width * jnp.sin(jnp.pi * x / half_width)

    h_u = terrain(grid.x_c[:, None], grid.y_m[None, :])[..., None]
    a_u = lateral_envelope(grid.x_c)[:, None, None]
    u = speed * a_u * (envelope(grid.Z_u) + (grid.Z_u - h_u) * envelope_derivative(grid.Z_u))
    h_m = terrain(grid.x_m[:, None], grid.y_m[None, :])
    h_plus = terrain((grid.x_m + DX)[:, None], grid.y_m[None, :])
    h_minus = terrain((grid.x_m - DX)[:, None], grid.y_m[None, :])
    dh_dx = (h_plus - h_minus) / (2.0 * DX)
    a_m = lateral_envelope(grid.x_m)[:, None, None]
    da_dx_m = lateral_envelope_derivative(grid.x_m)[:, None, None]
    w = speed * (a_m * dh_dx[..., None] - da_dx_m * (grid.Z_w - h_m[..., None])) * envelope(grid.Z_w)
    u_w = speed * a_m * (envelope(grid.Z_w) + (grid.Z_w - h_m[..., None]) * envelope_derivative(grid.Z_w))
    eta_dot = (w - u_w * grid.z_xi_w) / grid.dz_w_full
    flow = {"u": u, "v": jnp.zeros_like(grid.Z_v), "eta_dot": eta_dot}
    reverse_flow = jax.tree.map(lambda value: -value, flow)

    x = grid.x_m[:, None, None]
    y = grid.y_m[None, :, None]
    z = grid.Z_m
    tracer = jnp.exp(
        -0.5 * ((x + 1800.0) / 1100.0) ** 2 - 0.5 * ((y - 500.0) / 1200.0) ** 2 - 0.5 * ((z - 4500.0) / 1300.0) ** 2
    ) + 0.65 * jnp.exp(
        -0.5 * ((x - 2200.0) / 850.0) ** 2 - 0.5 * ((y + 700.0) / 1000.0) ** 2 - 0.5 * ((z - 8500.0) / 1500.0) ** 2
    )
    rho = jnp.ones_like(tracer)
    background = {"dz_m_full": grid.dz_m_full}
    return grid, advector, flow, reverse_flow, background, rho, tracer


def transport_reversibility(transform, terrain, steps=100, dt=2.0):
    """Return round-trip tracer error and reverse-leg error history."""
    (grid, advector, flow, reverse_flow, background, rho0, tracer0) = build_transport_case(transform, terrain, dt)

    def advance(carry, velocity):
        rho, tracer_mass = carry
        return (
            advector.advect_3d_split(rho, velocity, background),
            advector.advect_3d_split(tracer_mass, velocity, background),
        )

    def forward_step(carry, _):
        return advance(carry, flow), None

    initial = (rho0, rho0 * tracer0)
    transported, _ = jax.lax.scan(forward_step, initial, jnp.arange(steps))
    volumes = DX * DY * grid.dz_m_full / grid.m_factors["m"][..., None] ** 2
    denominator = jnp.sum(volumes * rho0 * tracer0**2)

    def reverse_step(carry, _):
        next_carry = advance(carry, reverse_flow)
        rho, tracer_mass = next_carry
        tracer = tracer_mass / (rho + 1.0e-15)
        error = jnp.sqrt(jnp.sum(volumes * rho0 * (tracer - tracer0) ** 2) / denominator)
        return next_carry, error

    final, history = jax.lax.scan(reverse_step, transported, jnp.arange(steps))
    rho_final, tracer_mass_final = final
    tracer_final = tracer_mass_final / (rho_final + 1.0e-15)
    mass0 = jnp.sum(volumes * rho0 * tracer0)
    mass1 = jnp.sum(volumes * tracer_mass_final)
    mass_drift = jnp.abs(mass1 - mass0) / jnp.abs(mass0)
    minimum, slope2, curvature = grid_diagnostics(grid)
    return (history[-1], history, grid, (minimum, slope2, curvature, mass_drift))


def run_target_case(target, transform, terrain, steps):
    if target == "pgf_rest":
        final, series, grid, diagnostics = run_case(transform, terrain, steps)
        return jnp.mean(series[0]), series[0], grid, diagnostics
    if target == "tracer_reversibility":
        return transport_reversibility(transform, terrain, steps)
    if target == "mountain_flux":
        return mountain_flux_consistency(transform, terrain, steps)
    raise ValueError(f"Unknown coordinate-training target: {target}")


def neuve_template(params=None):
    return NEUVECoordinate(params=params, hidden_dim=64, key_seed=42, condition_on_terrain=True, legacy_behavior=True)


def save_neuve(path, params):
    flat, _ = jax.tree_util.tree_flatten(params)
    np.savez(path, *[np.asarray(value) for value in flat])


def load_neuve(path):
    template = neuve_template()
    data = np.load(path)
    arrays = [data[f"arr_{index}"] for index in range(len(data.files))]
    _, tree = jax.tree_util.tree_flatten(template.params)
    return template.with_params(jax.tree_util.tree_unflatten(tree, [jnp.asarray(value) for value in arrays]))
