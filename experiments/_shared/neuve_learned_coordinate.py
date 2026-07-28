"""Reusable learned-density coordinate used by the NEUVE PGF experiment."""

import json
import math

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
    return (
        coefficients
        * y[..., None] ** powers
        * (1.0 - y[..., None]) ** (count - 1 - powers)
    )


class LearnedDensityCoordinate(BaseTransform):
    """Positive integrated density with optional terrain-conditioned residual."""

    def __init__(self, params, residual_scale=1.5):
        self.params = params
        self.residual_scale = residual_scale

    def _column_coefficients(self, xi, h):
        coefficients = self.params["global"]
        if "conditioner" not in self.params or h.ndim != 3:
            return coefficients
        h2 = h[:, :, 0] / 4000.0
        spacing = jnp.maximum(jnp.mean(jnp.abs(jnp.diff(xi[:, 0, 0]))), 1.0)
        hx = jnp.gradient(h[:, :, 0], axis=0) / spacing
        hy = jnp.gradient(h[:, :, 0], axis=1) / spacing
        slope = jnp.sqrt(hx**2 + hy**2 + 1.0e-12)
        curvature = (
            4000.0 * (jnp.gradient(hx, axis=0) + jnp.gradient(hy, axis=1)) / spacing
        )
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
            basis = bernstein_basis(y, coefficients.shape[-1])
            if coefficients.ndim == 1:
                log_density = jnp.sum(basis * coefficients, axis=-1)
            elif y.ndim == 1:
                log_density = jnp.sum(
                    basis[None, None, ...] * coefficients[..., None, :],
                    axis=-1,
                )
            else:
                log_density = jnp.sum(
                    basis * coefficients[..., None, None, :],
                    axis=-1,
                )
            return jnp.exp(jnp.clip(log_density, -6.0, 6.0))

        total_nodes = 0.5 * (GL_X + 1.0)
        total = 0.5 * jnp.sum(GL_W * density(total_nodes), axis=-1)
        partial_nodes = 0.5 * eta[..., None] * (GL_X + 1.0)
        partial = 0.5 * eta * jnp.sum(GL_W * density(partial_nodes), axis=-1)
        if jnp.ndim(total):
            total = total[..., None]
        cumulative = partial / total
        return zeta + h * (1.0 - cumulative)


def scalar_density_params(amplitude, basis_count):
    return {"global": jnp.linspace(amplitude, -amplitude, basis_count)}


def save_learned_coordinate(path, params, residual_scale=1.5, **metadata):
    payload = {
        "global": np.asarray(params["global"]),
        "residual_scale": np.asarray(residual_scale),
    }
    if "conditioner" in params:
        payload.update(
            {
                f"conditioner_{name}": np.asarray(value)
                for name, value in params["conditioner"].items()
            }
        )
    np.savez(
        path,
        **payload,
        metadata=json.dumps(metadata),
        coordinate_format="learned_density_v1",
    )


def load_learned_params(path):
    with np.load(path) as source:
        params = {"global": jnp.asarray(source["global"])}
        conditioner = {
            name.removeprefix("conditioner_"): jnp.asarray(source[name])
            for name in source.files
            if name.startswith("conditioner_")
        }
        residual_scale = (
            float(source["residual_scale"]) if "residual_scale" in source else 1.5
        )
    if conditioner:
        params["conditioner"] = conditioner
    return params, residual_scale


def load_learned_coordinate(path):
    params, residual_scale = load_learned_params(path)
    return LearnedDensityCoordinate(params, residual_scale)
