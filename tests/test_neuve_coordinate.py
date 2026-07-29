"""Regression tests for the learned NEUVE vertical coordinate."""

import jax
import jax.numpy as jnp

from experiments._shared.neuve_coordinate import (
    RESTING_N_BV,
    build_case,
    build_transport_case,
    neuve_template,
    random_3d_terrain,
    transport_reversibility,
)
from experiments._shared.neuve_learned_coordinate import (
    DirectDensityCoordinate,
    LearnedDensityCoordinate,
    bspline_basis,
    direct_density_params,
    load_direct_density_coordinate,
    load_learned_coordinate,
    save_direct_density_coordinate,
    save_learned_coordinate,
    scalar_density_params,
)
from suetes.shared.transforms import (
    GalChenSigma,
    IntegralNeuralTransform,
    NEUVECoordinate,
)


def test_integral_neural_transform_has_exact_endpoints_and_positive_layers():
    """A smooth positive learned density must not create crossed layers."""

    def varying_density_model(_params, y, h, slope, laplacian):
        return 4.0 * jnp.sin(2.0 * jnp.pi * y) + 0.2 * h + 0.1 * slope

    transform = IntegralNeuralTransform(varying_density_model, params={})
    nx, ny, nz = 7, 5, 48
    x = jnp.linspace(-3000.0, 3000.0, nx)
    eta = jnp.linspace(-2000.0, 2000.0, ny)
    zeta = jnp.linspace(0.0, 16000.0, nz + 1)
    xi, _, zeta_3d = jnp.meshgrid(x, eta, zeta, indexing="ij")
    terrain_2d = (
        2790.0
        * jnp.exp(-0.5 * (x[:, None] / 900.0) ** 2)
        * jnp.exp(-0.5 * (eta[None, :] / 1200.0) ** 2)
    )
    terrain = jnp.broadcast_to(terrain_2d[..., None], zeta_3d.shape)

    physical_z = transform(xi, zeta_3d, terrain, 16000.0)

    assert jnp.allclose(physical_z[..., 0], terrain_2d, atol=1.0e-10)
    assert jnp.allclose(physical_z[..., -1], 16000.0, atol=1.0e-10)
    assert float(jnp.min(jnp.diff(physical_z, axis=-1))) > 0.0


def test_default_neuve_reduces_to_nearly_uniform_monotone_coordinate():
    coordinate = NEUVECoordinate(hidden_dim=64, key_seed=42)
    zeta = jnp.linspace(0.0, 16000.0, 33)
    physical_z = coordinate(
        jnp.linspace(0.0, 16000.0, 33), zeta, jnp.asarray(1800.0), 16000.0
    )

    assert jnp.isclose(physical_z[0], 1800.0)
    assert jnp.isclose(physical_z[-1], 16000.0)
    assert float(jnp.min(jnp.diff(physical_z))) > 0.0


def test_global_decay_profile_is_shared_between_terrain_columns():
    coordinate = NEUVECoordinate(
        hidden_dim=64,
        key_seed=42,
        condition_on_terrain=False,
    )
    x = jnp.linspace(0.0, 1500.0, 4)
    y = jnp.linspace(0.0, 1000.0, 3)
    zeta = jnp.linspace(0.0, 16000.0, 17)
    xi, _, zeta_3d = jnp.meshgrid(x, y, zeta, indexing="ij")
    terrain_2d = 500.0 + 300.0 * jnp.arange(4)[:, None] + jnp.zeros((4, 3))
    terrain = jnp.broadcast_to(terrain_2d[..., None], zeta_3d.shape)

    physical_z = coordinate(xi, zeta_3d, terrain, 16000.0)
    normalized_imprint = (physical_z - zeta_3d) / terrain

    assert jnp.allclose(
        normalized_imprint,
        normalized_imprint[0:1, 0:1, :],
        atol=2.0e-6,
    )
    assert float(jnp.min(jnp.diff(physical_z, axis=-1))) > 0.0


def test_reversible_transport_target_is_finite_and_mass_conservative():
    error, history, _, diagnostics = transport_reversibility(
        GalChenSigma(), random_3d_terrain(17), steps=2
    )

    assert history.shape == (2,)
    assert jnp.isfinite(error)
    assert float(error) > 0.0
    # The unit test may run with JAX's default float32 configuration.
    assert float(diagnostics[3]) < 1.0e-6


def test_reversible_transport_has_closed_lateral_boundaries():
    terrain = random_3d_terrain(17)
    _, _, flow, _, _, _, _ = build_transport_case(GalChenSigma(), terrain)

    assert float(jnp.max(jnp.abs(flow["u"][0]))) < 1.0e-6
    assert float(jnp.max(jnp.abs(flow["u"][-1]))) < 1.0e-6


def test_matched_hydrostatic_reference_remains_at_rest():
    _, state, stepper, boundary = build_case(
        neuve_template(),
        random_3d_terrain(17),
        reference_n_bv=RESTING_N_BV,
    )

    next_state = stepper.step(state, 0.0, None, boundary)

    assert jnp.max(jnp.abs(next_state["u"])) < 1.0e-6
    assert jnp.max(jnp.abs(next_state["v"])) < 1.0e-6
    assert jnp.max(jnp.abs(next_state["w"])) < 1.0e-6


def test_direct_density_coordinate_has_exact_endpoints_and_positive_layers():
    coordinate = LearnedDensityCoordinate({"global": jnp.linspace(2.0, -2.0, 12)})
    x = jnp.linspace(-2000.0, 2000.0, 5)
    y = jnp.linspace(-1000.0, 1000.0, 3)
    zeta = jnp.linspace(0.0, 16000.0, 33)
    xi, _, zeta_3d = jnp.meshgrid(x, y, zeta, indexing="ij")
    terrain_2d = 1800.0 * jnp.exp(-0.5 * (x[:, None] / 900.0) ** 2) * jnp.ones((1, 3))
    terrain = jnp.broadcast_to(terrain_2d[..., None], zeta_3d.shape)

    physical_z = coordinate(xi, zeta_3d, terrain, 16000.0)

    assert jnp.allclose(physical_z[..., 0], terrain_2d)
    assert jnp.allclose(physical_z[..., -1], 16000.0)
    assert float(jnp.min(jnp.diff(physical_z, axis=-1))) > 0.0


def test_zero_aggressiveness_direct_coordinate_is_galchen():
    direct = LearnedDensityCoordinate({"global": jnp.zeros(12)})
    galchen = GalChenSigma()
    x = jnp.linspace(-2000.0, 2000.0, 5)
    y = jnp.linspace(-1000.0, 1000.0, 3)
    zeta = jnp.linspace(0.0, 16000.0, 17)
    xi, _, zeta_3d = jnp.meshgrid(x, y, zeta, indexing="ij")
    terrain = jnp.broadcast_to(
        (1500.0 * jnp.exp(-0.5 * (x[:, None] / 900.0) ** 2))[..., None],
        zeta_3d.shape,
    )

    assert jnp.allclose(
        direct(xi, zeta_3d, terrain, 16000.0),
        galchen(xi, zeta_3d, terrain, 16000.0),
    )


def test_learned_coordinate_checkpoint_roundtrip(tmp_path):
    params = {"global": jnp.linspace(2.0, -2.0, 12)}
    path = tmp_path / "coordinate.npz"
    save_learned_coordinate(path, params, residual_scale=1.25)
    loaded = load_learned_coordinate(path)

    assert loaded.residual_scale == 1.25
    assert jnp.allclose(loaded.params["global"], params["global"])


def test_cubic_bspline_basis_is_local_and_partitions_unity():
    y = jnp.linspace(0.0, 1.0, 101)
    basis = bspline_basis(y, count=16)

    assert jnp.all(basis >= 0.0)
    assert jnp.allclose(jnp.sum(basis, axis=-1), 1.0, atol=1.0e-6)
    assert int(jnp.max(jnp.sum(basis > 1.0e-12, axis=-1))) <= 4
    assert jnp.allclose(basis[0], jax.nn.one_hot(0, 16))
    assert jnp.allclose(basis[-1], jax.nn.one_hot(15, 16))


def test_cubic_bspline_scalar_initialization_reproduces_linear_log_density():
    y = jnp.linspace(0.0, 1.0, 101)
    params = scalar_density_params(2.0, 16, basis="bspline")
    represented = jnp.sum(bspline_basis(y, 16) * params["global"], axis=-1)

    assert jnp.allclose(represented, 2.0 * (1.0 - 2.0 * y), atol=1.0e-6)


def test_bspline_checkpoint_roundtrip(tmp_path):
    params = {"global": jnp.linspace(2.0, -2.0, 16)}
    path = tmp_path / "coordinate.npz"
    save_learned_coordinate(path, params, residual_scale=1.25, basis="bspline")
    loaded = load_learned_coordinate(path)

    assert loaded.basis == "bspline"
    assert loaded.residual_scale == 1.25
    assert jnp.allclose(loaded.params["global"], params["global"])


def test_direct_mlp_density_has_exact_endpoints_and_positive_layers():
    params = direct_density_params(amplitude=2.0, hidden=8, seed=3)
    coordinate = DirectDensityCoordinate(params)
    x = jnp.linspace(-2000.0, 2000.0, 5)
    y = jnp.linspace(-1000.0, 1000.0, 3)
    zeta = jnp.linspace(0.0, 16000.0, 33)
    xi, _, zeta_3d = jnp.meshgrid(x, y, zeta, indexing="ij")
    terrain_2d = 1800.0 * jnp.exp(-0.5 * (x[:, None] / 900.0) ** 2) * jnp.ones((1, 3))
    terrain = jnp.broadcast_to(terrain_2d[..., None], zeta_3d.shape)

    physical_z = coordinate(xi, zeta_3d, terrain, 16000.0)

    assert jnp.allclose(physical_z[..., 0], terrain_2d)
    assert jnp.allclose(physical_z[..., -1], 16000.0)
    assert float(jnp.min(jnp.diff(physical_z, axis=-1))) > 0.0


def test_direct_mlp_density_checkpoint_roundtrip(tmp_path):
    params = direct_density_params(amplitude=2.0, hidden=8, seed=3)
    path = tmp_path / "coordinate.npz"
    save_direct_density_coordinate(path, params, residual_scale=1.25)
    loaded = load_direct_density_coordinate(path)

    assert loaded.residual_scale == 1.25
    assert jnp.allclose(loaded.params["amplitude"], params["amplitude"])
    assert jnp.allclose(
        loaded.params["network"]["w2"],
        params["network"]["w2"],
    )
