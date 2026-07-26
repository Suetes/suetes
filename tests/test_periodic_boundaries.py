"""Focused tests for opt-in periodic horizontal boundary operators."""

import jax.numpy as jnp
import numpy as np

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.steppers import FluxFormAdvector


def make_grid(nx=8, ny=4, nz=2):
    return RegionalGrid3D(
        nx, ny, nz,
        dx=1.0, dy=1.0, dz=1.0,
        lat_center=0.0, lon_center=0.0,
    )


def test_periodic_mass_to_face_difference_and_average():
    grid = make_grid()
    operator = CGridOperator3D(grid, periodic_axes=(0,))
    field = jnp.arange(grid.nx, dtype=jnp.float32)[:, None, None]

    difference = operator.diff(field, 0, "m", "u")[:, 0, 0]
    average = operator.avg(field, 0, "m", "u")[:, 0, 0]

    expected_boundary_difference = field[0, 0, 0] - field[-1, 0, 0]
    expected_boundary_average = 0.5 * (
        field[0, 0, 0] + field[-1, 0, 0]
    )
    np.testing.assert_allclose(
        difference[jnp.asarray([0, -1])], expected_boundary_difference
    )
    np.testing.assert_allclose(
        average[jnp.asarray([0, -1])], expected_boundary_average
    )
    np.testing.assert_allclose(difference[1:-1], 1.0)


def test_default_operator_keeps_open_edge_closure():
    grid = make_grid()
    operator = CGridOperator3D(grid)
    field = jnp.arange(grid.nx, dtype=jnp.float32)[:, None, None]

    difference = operator.diff(field, 0, "m", "u")[:, 0, 0]

    np.testing.assert_allclose(difference[jnp.asarray([0, -1])], 1.0)
    assert operator.periodic_axes == frozenset()


def test_periodic_ffsl_wraps_and_conserves_mass():
    grid = make_grid()
    advector = FluxFormAdvector(grid, dt=1.0, periodic_axes=(0,))
    scalar = jnp.asarray([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])

    positive_cfl = jnp.ones(grid.nx + 1)
    negative_cfl = -positive_cfl
    shifted_right = advector.advect_1d(
        scalar, positive_cfl, periodic=True
    )
    shifted_left = advector.advect_1d(
        scalar, negative_cfl, periodic=True
    )

    np.testing.assert_allclose(shifted_right, jnp.roll(scalar, 1))
    np.testing.assert_allclose(shifted_left, jnp.roll(scalar, -1))
    np.testing.assert_allclose(jnp.sum(shifted_right), jnp.sum(scalar))
    np.testing.assert_allclose(jnp.sum(shifted_left), jnp.sum(scalar))


def test_periodic_ffsl_fractional_transport_conserves_mass():
    grid = make_grid()
    advector = FluxFormAdvector(grid, dt=1.0, periodic_axes=(0,))
    scalar = jnp.asarray([0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0])
    cfl = jnp.full(grid.nx + 1, 0.35)

    transported = advector.advect_1d(scalar, cfl, periodic=True)

    np.testing.assert_allclose(
        jnp.sum(transported), jnp.sum(scalar), rtol=1e-6, atol=1e-6
    )
    assert float(jnp.min(transported)) >= 0.0
