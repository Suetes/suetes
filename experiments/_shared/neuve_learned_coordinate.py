"""Reusable learned-density coordinate used by the NEUVE PGF experiment."""

import json
import math

import jax
import jax.numpy as jnp
import numpy as np

from suetes.shared.transforms import BaseTransform

GL_X = jnp.asarray(
    [
        -0.9815606342467192,
        -0.9041172563704749,
        -0.7699026743093193,
        -0.5873179542866175,
        -0.3678314989981802,
        -0.1252334085114689,
        0.1252334085114689,
        0.3678314989981802,
        0.5873179542866175,
        0.7699026743093193,
        0.9041172563704749,
        0.9815606342467192,
    ]
)
GL_W = jnp.asarray(
    [
        0.0471753363865118,
        0.1069393259953184,
        0.1600783285433462,
        0.2031674267230659,
        0.2334925365383548,
        0.2491470458134028,
        0.2491470458134028,
        0.2334925365383548,
        0.2031674267230659,
        0.1600783285433462,
        0.1069393259953184,
        0.0471753363865118,
    ]
)


def bernstein_basis(y, count):
    powers = jnp.arange(count)
    coefficients = jnp.asarray([math.comb(count - 1, index) for index in range(count)])
    return coefficients * y[..., None] ** powers * (1.0 - y[..., None]) ** (count - 1 - powers)


def _open_uniform_knots(count, degree=3):
    if count <= degree:
        raise ValueError(f"A degree-{degree} B-spline requires at least {degree + 1} coefficients")
    internal_count = count - degree - 1
    internal = jnp.linspace(0.0, 1.0, internal_count + 2)[1:-1]
    return jnp.concatenate((jnp.zeros(degree + 1), internal, jnp.ones(degree + 1)))


def bspline_basis(y, count, degree=3):
    """Evaluate a clamped, open-uniform B-spline basis on ``[0, 1]``."""

    y = jnp.clip(jnp.asarray(y), 0.0, 1.0)
    knots = _open_uniform_knots(count, degree)
    values = ((y[..., None] >= knots[:-1]) & (y[..., None] < knots[1:])).astype(y.dtype)
    for order in range(1, degree + 1):
        function_count = knots.size - order - 1
        left_denominator = knots[order : order + function_count] - knots[:function_count]
        right_denominator = knots[order + 1 : order + function_count + 1] - knots[1 : function_count + 1]
        left = jnp.where(
            left_denominator > 0.0,
            (y[..., None] - knots[:function_count])
            / jnp.where(left_denominator > 0.0, left_denominator, 1.0)
            * values[..., :function_count],
            0.0,
        )
        right = jnp.where(
            right_denominator > 0.0,
            (knots[order + 1 : order + function_count + 1] - y[..., None])
            / jnp.where(right_denominator > 0.0, right_denominator, 1.0)
            * values[..., 1 : function_count + 1],
            0.0,
        )
        values = left + right
    values = values[..., :count]
    endpoint = jax.nn.one_hot(count - 1, count, dtype=y.dtype)
    return jnp.where((y == 1.0)[..., None], endpoint, values)


def density_basis(y, count, basis):
    if basis == "bernstein":
        return bernstein_basis(y, count)
    if basis == "bspline":
        return bspline_basis(y, count)
    raise ValueError(f"Unknown learned-density basis: {basis}")


class LearnedDensityCoordinate(BaseTransform):
    """Positive integrated density with optional terrain-conditioned residual."""

    def __init__(self, params, residual_scale=1.5, basis="bernstein"):
        self.params = params
        self.residual_scale = residual_scale
        self.basis = basis

    def _column_coefficients(self, xi, h):
        coefficients = self.params["global"]
        if "conditioner" not in self.params or h.ndim != 3:
            return coefficients
        h2 = h[:, :, 0] / 4000.0
        spacing = jnp.maximum(jnp.mean(jnp.abs(jnp.diff(xi[:, 0, 0]))), 1.0)
        hx = jnp.gradient(h[:, :, 0], axis=0) / spacing
        hy = jnp.gradient(h[:, :, 0], axis=1) / spacing
        slope = jnp.sqrt(hx**2 + hy**2 + 1.0e-12)
        curvature = 4000.0 * (jnp.gradient(hx, axis=0) + jnp.gradient(hy, axis=1)) / spacing
        features = jnp.stack((h2, slope, curvature), axis=-1)
        network = self.params["conditioner"]
        hidden = jnp.tanh(features @ network["w1"] + network["b1"])
        residual = jnp.tanh(hidden @ network["w2"] + network["b2"])
        residual -= jnp.mean(residual, axis=-1, keepdims=True)
        return coefficients + self.residual_scale * residual

    def __call__(self, xi, zeta, h, Lz):
        eta = jnp.clip(zeta / Lz, 0.0, 1.0)
        coefficients = self._column_coefficients(xi, h)

        def density(y):
            basis = density_basis(y, coefficients.shape[-1], self.basis)
            if coefficients.ndim == 1:
                log_density = jnp.sum(basis * coefficients, axis=-1)
            elif y.ndim == 1:
                log_density = jnp.sum(basis[None, None, ...] * coefficients[..., None, :], axis=-1)
            else:
                log_density = jnp.sum(basis * coefficients[..., None, None, :], axis=-1)
            return jnp.exp(jnp.clip(log_density, -6.0, 6.0))

        total_nodes = 0.5 * (GL_X + 1.0)
        total = 0.5 * jnp.sum(GL_W * density(total_nodes), axis=-1)
        partial_nodes = 0.5 * eta[..., None] * (GL_X + 1.0)
        partial = 0.5 * eta * jnp.sum(GL_W * density(partial_nodes), axis=-1)
        if jnp.ndim(total):
            total = total[..., None]
        cumulative = partial / total
        return zeta + h * (1.0 - cumulative)


class DirectDensityCoordinate(BaseTransform):
    """Terrain-conditioned MLP that predicts log density directly."""

    def __init__(self, params, residual_scale=1.5):
        self.params = params
        self.residual_scale = residual_scale

    @staticmethod
    def _terrain_features(xi, h):
        h2 = h[:, :, 0] / 4000.0
        spacing = jnp.maximum(jnp.mean(jnp.abs(jnp.diff(xi[:, 0, 0]))), 1.0)
        hx = jnp.gradient(h[:, :, 0], axis=0) / spacing
        hy = jnp.gradient(h[:, :, 0], axis=1) / spacing
        slope = jnp.sqrt(hx**2 + hy**2 + 1.0e-12)
        curvature = 4000.0 * (jnp.gradient(hx, axis=0) + jnp.gradient(hy, axis=1)) / spacing
        return jnp.stack((h2, slope, curvature), axis=-1)

    def _log_density(self, y, features):
        y = jnp.asarray(y)
        y_values = y if y.ndim >= 2 else jnp.reshape(y, (1, 1, y.shape[0]))
        horizontal_shape = features.shape[:-1]
        feature_shape = horizontal_shape + (1,) * (y_values.ndim - 2) + (features.shape[-1],)
        target_shape = jnp.broadcast_shapes(feature_shape[:-1], y_values.shape)
        feature_values = jnp.broadcast_to(jnp.reshape(features, feature_shape), target_shape + (3,))
        height_values = jnp.broadcast_to(y_values, target_shape)
        inputs = jnp.concatenate((height_values[..., None], feature_values), axis=-1)
        network = self.params["network"]
        hidden = jnp.tanh(inputs @ network["w1"] + network["b1"])
        hidden = jnp.tanh(hidden @ network["w2"] + network["b2"])
        residual = (hidden @ network["w3"] + network["b3"])[..., 0]
        base = self.params["amplitude"] * (1.0 - 2.0 * height_values)
        return base + self.residual_scale * residual

    def __call__(self, xi, zeta, h, Lz):
        if h.ndim != 3:
            raise ValueError("DirectDensityCoordinate requires a 3-D terrain field")
        eta = jnp.clip(zeta / Lz, 0.0, 1.0)
        features = self._terrain_features(xi, h)

        def density(y):
            return jnp.exp(jnp.clip(self._log_density(y, features), -6.0, 6.0))

        total_nodes = 0.5 * (GL_X + 1.0)
        total = 0.5 * jnp.sum(GL_W * density(total_nodes), axis=-1)
        partial_nodes = 0.5 * eta[..., None] * (GL_X + 1.0)
        partial = 0.5 * eta * jnp.sum(GL_W * density(partial_nodes), axis=-1)
        cumulative = partial / total[..., None]
        return zeta + h * (1.0 - cumulative)


def scalar_density_params(amplitude, basis_count, basis="bernstein"):
    if basis == "bernstein":
        locations = jnp.linspace(0.0, 1.0, basis_count)
    elif basis == "bspline":
        knots = _open_uniform_knots(basis_count)
        locations = jnp.asarray([jnp.mean(knots[index + 1 : index + 4]) for index in range(basis_count)])
    else:
        raise ValueError(f"Unknown learned-density basis: {basis}")
    return {"global": amplitude * (1.0 - 2.0 * locations)}


def save_learned_coordinate(path, params, residual_scale=1.5, basis="bernstein", **metadata):
    payload = {"global": np.asarray(params["global"]), "residual_scale": np.asarray(residual_scale)}
    if "conditioner" in params:
        payload.update({f"conditioner_{name}": np.asarray(value) for name, value in params["conditioner"].items()})
    np.savez(path, **payload, metadata=json.dumps(metadata), coordinate_format="learned_density_v1", basis=basis)


def load_learned_params(path):
    with np.load(path) as source:
        params = {"global": jnp.asarray(source["global"])}
        conditioner = {
            name.removeprefix("conditioner_"): jnp.asarray(source[name]) for name in source.files if name.startswith("conditioner_")
        }
        residual_scale = float(source["residual_scale"]) if "residual_scale" in source else 1.5
    if conditioner:
        params["conditioner"] = conditioner
    return params, residual_scale


def load_learned_basis(path):
    with np.load(path) as source:
        return str(source["basis"]) if "basis" in source else "bernstein"


def load_learned_coordinate(path):
    params, residual_scale = load_learned_params(path)
    return LearnedDensityCoordinate(params, residual_scale, basis=load_learned_basis(path))


def direct_density_params(amplitude, hidden, seed):
    first, second = jax.random.split(jax.random.PRNGKey(seed))
    return {
        "amplitude": jnp.asarray(amplitude),
        "network": {
            "w1": 0.2 * jax.random.normal(first, (4, hidden)),
            "b1": jnp.zeros(hidden),
            "w2": 0.2 * jax.random.normal(second, (hidden, hidden)),
            "b2": jnp.zeros(hidden),
            "w3": jnp.zeros((hidden, 1)),
            "b3": jnp.zeros(1),
        },
    }


def save_direct_density_coordinate(path, params, residual_scale=1.5, **metadata):
    payload = {"amplitude": np.asarray(params["amplitude"]), "residual_scale": np.asarray(residual_scale)}
    payload.update({f"network_{name}": np.asarray(value) for name, value in params["network"].items()})
    np.savez(path, **payload, metadata=json.dumps(metadata), coordinate_format="direct_density_mlp_v1")


def load_direct_density_coordinate(path):
    with np.load(path) as source:
        params = {
            "amplitude": jnp.asarray(source["amplitude"]),
            "network": {name.removeprefix("network_"): jnp.asarray(source[name]) for name in source.files if name.startswith("network_")},
        }
        residual_scale = float(source["residual_scale"]) if "residual_scale" in source else 1.5
    return DirectDensityCoordinate(params, residual_scale)
