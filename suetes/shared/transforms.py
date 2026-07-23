import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import serialization


# ===========================================================
# TRANSFORM STRATEGIES
# ===========================================================

class BaseTransform:
    def __call__(self, xi, zeta, h, Lz):
        raise NotImplementedError


class GalChenSigma(BaseTransform):
    """
    The classic 'Sigma-Z' linear decay (Gal-Chen & Somerville, 1975).
    z = h + zeta * (Lz - h) / Lz
    """
    def __call__(self, xi, zeta, h, Lz):
        return h + zeta * (Lz - h) / Lz


class HybridSigma(BaseTransform):
    """
    Hybrid Sigma-Z: Terrain influence decays exponentially with height.
    z = zeta + h * (1 - zeta/Lz) * jnp.exp(-zeta / self.scale_height)
    """
    def __init__(self, scale_height=5000.0):
        self.scale_height = scale_height

    def __call__(self, xi, zeta, h, Lz):
        decay = (1.0 - zeta/Lz) * jnp.exp(-zeta / self.scale_height)
        return zeta + h * decay


class SleveSimple(BaseTransform):
    """
    Simplified SLEVE Coordinate with a single topography scale.
    """
    def __init__(self, scale_s=4000.0, scale_l=15000.0, n=1.35):
        self.ss = scale_s
        self.sl = scale_l
        self.n = n

    def __call__(self, xi, zeta, h, Lz):
        zeta_safe = jnp.clip(zeta, 0.0, Lz)
        b_s = jnp.sinh((Lz - zeta_safe)/self.ss) / jnp.sinh(Lz/self.ss)
        b_s_safe = jnp.clip(b_s, 1e-8, 1.0)
        return zeta_safe + h * (b_s_safe**self.n)


class StretchedSleveSimple(BaseTransform):
    """
    Simplified SLEVE Coordinate with exponential boundary-layer stretching.
    """
    def __init__(self, stretch_kappa=2.5, scale_s=4000.0, n=1.35):
        """
        Args:
            stretch_kappa (float): Stretching factor. Higher values compress layers closer to the surface.
            scale_s (float): Scale height for the terrain decay.
            n (float): Exponent for the simplified SLEVE decay profile.
        """
        self.kappa = stretch_kappa
        self.ss = scale_s
        self.n = n

    def __call__(self, xi, zeta, h, Lz):
        # Intermediate mapping (Computational Uniform -> Stretched Non-Uniform)
        eta = zeta / Lz
        
        # Safe evaluation: if kappa is very close to 0, default to uniform spacing
        zeta_stretched = jnp.where(
            jnp.abs(self.kappa) > 1e-5,
            Lz * (jnp.exp(self.kappa * eta) - 1.0) / (jnp.exp(self.kappa) - 1.0),
            zeta
        )

        zeta_stretched = jnp.clip(zeta_stretched, 0.0, Lz)

        # Simplified SLEVE decay using the stretched coordinate
        b_s = jnp.sinh((Lz - zeta_stretched) / self.ss) / jnp.sinh(Lz / self.ss)
        b_s_safe = jnp.clip(b_s, 1e-8, 1.0)
        
        return zeta_stretched + h * (b_s_safe ** self.n)


class Sleve(BaseTransform):
    """
    General SLEVE Coordinate (Schär et al., 2002).
    """
    def __init__(self, h1_func, s1=15000.0, s2=2500.0):
        self.h1_func = h1_func
        self.s1 = s1
        self.s2 = s2

    def _b_func(self, zeta, Lz, s):
        return jnp.sinh((Lz - zeta)/s) / jnp.sinh(Lz/s)

    def __call__(self, xi, zeta, h, Lz):
        h1 = self.h1_func(xi)
        h2 = h - h1
        b1 = self._b_func(zeta, Lz, self.s1)
        b2 = self._b_func(zeta, Lz, self.s2)
        return zeta + h1 * b1 + h2 * b2


class StretchedSleve(BaseTransform):
    """
    SLEVE Coordinate with exponential boundary-layer stretching.
    """
    def __init__(self, h1_func, stretch_kappa=2.5, s1=15000.0, s2=2500.0):
        self.h1_func = h1_func
        self.kappa = stretch_kappa
        self.s1 = s1
        self.s2 = s2

    def _b_func(self, zeta, Lz, s):
        return jnp.sinh((Lz - zeta)/s) / jnp.sinh(Lz/s)

    def __call__(self, xi, zeta, h, Lz):
        # Intermediate mapping (Computational Uniform -> Stretched Non-Uniform)
        eta = zeta / Lz
        
        # Safe evaluation: if kappa is very close to 0, default to uniform spacing
        zeta_stretched = jnp.where(
            jnp.abs(self.kappa) > 1e-5,
            Lz * (jnp.exp(self.kappa * eta) - 1.0) / (jnp.exp(self.kappa) - 1.0),
            zeta
        )

        # Terrain-following SLEVE transform using the stretched coordinate
        h1 = self.h1_func(xi)
        h2 = h - h1
        b1 = self._b_func(zeta_stretched, Lz, self.s1)
        b2 = self._b_func(zeta_stretched, Lz, self.s2)

        return zeta_stretched + h1 * b1 + h2 * b2


class IntegralNeuralTransform(BaseTransform):
    """
    Spatially Adaptive Robust Neural Coordinate.
    Predicts a positive vertical layer-density field and integrates it to
    obtain a normalized cumulative coordinate S.  The physical height is

        z = zeta + h (1 - S).

    This terrain-decay form is important: spatial variations learned by the
    network are scaled by terrain height, rather than by the full atmospheric
    depth.  With density in [0.3, 1.7], it guarantees positive vertical
    derivatives whenever h/Lz < 3/17, in addition to the exact endpoints
    z(0)=h and z(Lz)=Lz.  ``RegionalGrid3D`` also checks physical layer
    thicknesses when grids are constructed outside a traced calculation.
    """
    def __init__(self, nn_apply_fn, params, condition_on_terrain=True,
                 legacy_behavior=False):
        self.apply_fn = nn_apply_fn
        self.params = params
        self.condition_on_terrain = condition_on_terrain
        self.legacy_behavior = legacy_behavior

    def __call__(self, xi, zeta, h, Lz):
        # 12-point Gauss-Legendre Quadrature nodes and weights on [-1, 1]
        x_nodes = jnp.array([
            -0.9815606342467192, -0.9041172563704749, -0.7699026741943047,
            -0.5873179542866175, -0.3678314989981802, -0.1252334085114689,
             0.1252334085114689,  0.3678314989981802,  0.5873179542866175,
             0.7699026741943047,  0.9041172563704749,  0.9815606342467192
        ])
        w_nodes = jnp.array([
             0.0471753363865118,  0.1069393259953184,  0.1600783285433462,
             0.2031674267230659,  0.2334925365383548,  0.2491470458134028,
             0.2491470458134028,  0.2334925365383548,  0.2031674267230659,
             0.1600783285433462,  0.1069393259953184,  0.0471753363865118
        ])

        h_norm = h / 4000.0

        if zeta.ndim == 3:
            # 3D field case: shape (nx, ny, nz)
            h_2d = h_norm[:, :, 0:1, None]  # (nx, ny, 1, 1)
            # ``RegionalGrid3D`` currently passes only xi to transforms.  Its
            # NEUVE experiments use dx=dy, so infer that common physical grid
            # spacing from xi.  Scaling by the 4-km terrain normalization makes
            # slope and curvature features independent of horizontal resolution.
            spacing_norm = jnp.maximum(
                jnp.mean(jnp.abs(jnp.diff(xi[:, 0, 0]))) / 4000.0, 1.0e-8
            )
            feature_spacing = 1.0 if self.legacy_behavior else spacing_norm
            dh_dx = (
                jnp.gradient(h_norm[:, :, 0], axis=0) / feature_spacing
            )[:, :, None, None]
            dh_dy = (
                jnp.gradient(h_norm[:, :, 0], axis=1) / feature_spacing
            )[:, :, None, None]
            slope_2d = jnp.sqrt(dh_dx**2 + dh_dy**2 + 1e-8)
            d2h_dx2 = (
                jnp.gradient(dh_dx[:, :, 0, 0], axis=0) / feature_spacing
            )[:, :, None, None]
            d2h_dy2 = (
                jnp.gradient(dh_dy[:, :, 0, 0], axis=1) / feature_spacing
            )[:, :, None, None]
            laplacian_2d = d2h_dx2 + d2h_dy2

            if not self.condition_on_terrain:
                # Learn one shared vertical decay profile b(eta).  Terrain
                # enters only through z=zeta+h*b, so every coordinate surface
                # remains a smooth scaled copy of the physical topography.
                h_2d = jnp.zeros_like(h_2d)
                slope_2d = jnp.zeros_like(slope_2d)
                laplacian_2d = jnp.zeros_like(laplacian_2d)

            eta = zeta / Lz

            def eval_density(y_inputs, h_inputs, s_inputs, l_inputs):
                raw_out = self.apply_fn(self.params, y_inputs, h_inputs, s_inputs, l_inputs).squeeze(-1)
                if self.legacy_behavior:
                    return 0.08 + 1.90 * jax.nn.sigmoid(raw_out)
                # The ratio 1.7/0.3 = 5.67 is large enough to represent the
                # near-surface decay of the SLEVE comparator (~5.4), while the
                # PGF experiment's h/Lz < 3/17 bound still prevents tangling.
                return 0.3 + 1.4 * jax.nn.sigmoid(raw_out)

            # --- 1. Compute I(1) = \int_0^1 \rho(y) dy ---
            y_1 = (0.5 * x_nodes + 0.5)[None, None, None, :, None]
            h_1 = h_2d[:, :, :, None, :]
            s_1 = slope_2d[:, :, :, None, :]
            l_1 = laplacian_2d[:, :, :, None, :]
            y_1_b, h_1_b, s_1_b, l_1_b = jnp.broadcast_arrays(y_1, h_1, s_1, l_1)
            density_1 = eval_density(y_1_b, h_1_b, s_1_b, l_1_b)
            I_total = 0.5 * jnp.sum(w_nodes * density_1, axis=-1)  # (nx, ny, 1)

            # --- 2. Compute I(eta) = \int_0^eta \rho(y) dy ---
            y_eta = (0.5 * eta[..., None] * x_nodes + 0.5 * eta[..., None])[..., None]  # (nx, ny, nz, 12, 1)
            h_eta = h_2d[:, :, :, None, :]
            s_eta = slope_2d[:, :, :, None, :]
            l_eta = laplacian_2d[:, :, :, None, :]
            y_eta_b, h_eta_b, s_eta_b, l_eta_b = jnp.broadcast_arrays(y_eta, h_eta, s_eta, l_eta)
            density_eta = eval_density(y_eta_b, h_eta_b, s_eta_b, l_eta_b)
            I_eta = 0.5 * eta * jnp.sum(w_nodes * density_eta, axis=-1)  # (nx, ny, nz)

            S_eta = I_eta / I_total
            return zeta + h * (1.0 - S_eta)
        else:
            # 1D column / scalar case
            h_scalar = (
                jnp.mean(h_norm) if self.condition_on_terrain else jnp.asarray(0.0)
            )
            eta = zeta / Lz

            def eval_density_1d(y_inputs):
                h_in = jnp.full_like(y_inputs, h_scalar)
                s_in = jnp.zeros_like(y_inputs)
                l_in = jnp.zeros_like(y_inputs)
                raw_out = self.apply_fn(self.params, y_inputs, h_in, s_in, l_in).squeeze(-1)
                if self.legacy_behavior:
                    return 0.08 + 1.90 * jax.nn.sigmoid(raw_out)
                return 0.3 + 1.4 * jax.nn.sigmoid(raw_out)

            y_1 = (0.5 * x_nodes + 0.5)[:, None]
            density_1 = eval_density_1d(y_1)
            I_total = 0.5 * jnp.sum(w_nodes * density_1)

            y_eta = (0.5 * eta[..., None] * x_nodes + 0.5 * eta[..., None])[..., None]
            density_eta = eval_density_1d(y_eta)
            I_eta = 0.5 * eta * jnp.sum(w_nodes * density_eta, axis=-1)

            S_eta = I_eta / I_total
            return zeta + h * (1.0 - S_eta)


class NEUVEMLP(nn.Module):
    """
    Spatially adaptive neural network for NEUVE vertical coordinate density prediction.
    Takes non-dimensional vertical height y in [0, 1], normalized local terrain height h_norm,
    local terrain slope magnitude |nabla h|, and local terrain curvature/Laplacian nabla^2 h.
    Uses tanh activations for infinitely smooth adjoint gradients.
    """
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, y, h_norm=None, slope=None, laplacian=None):
        features = [y]
        if h_norm is not None:
            features.append(h_norm)
        else:
            features.append(jnp.zeros_like(y))
        if slope is not None:
            features.append(slope)
        else:
            features.append(jnp.zeros_like(y))
        if laplacian is not None:
            features.append(laplacian)
        else:
            features.append(jnp.zeros_like(y))
        inputs = jnp.concatenate(features, axis=-1)
        x = nn.Dense(self.hidden_dim)(inputs)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(
            1,
            kernel_init=jax.nn.initializers.normal(stddev=1e-3),
            bias_init=jax.nn.initializers.zeros,
        )(x)
        return x


class NEUVECoordinate(BaseTransform):
    """
    Self-contained Spatially Adaptive Neural Vertical Coordinate (NEUVE) wrapper.
    Can be passed directly as `transform` to RegionalGrid3D.
    """
    def __init__(self, params=None, hidden_dim=64, key_seed=42,
                 condition_on_terrain=True, legacy_behavior=False):
        self.model = NEUVEMLP(hidden_dim=hidden_dim)
        self.condition_on_terrain = condition_on_terrain
        self.legacy_behavior = legacy_behavior
        if params is None:
            key = jax.random.PRNGKey(key_seed)
            dummy_y = jnp.linspace(0, 1.0, 51)[:, None]
            dummy_h = jnp.zeros_like(dummy_y)
            dummy_slope = jnp.zeros_like(dummy_y)
            dummy_laplacian = jnp.zeros_like(dummy_y)
            self.params = self.model.init(key, dummy_y, dummy_h, dummy_slope, dummy_laplacian)
        else:
            self.params = params
        self.integral_transform = IntegralNeuralTransform(
            self.model.apply, self.params,
            condition_on_terrain=self.condition_on_terrain,
            legacy_behavior=self.legacy_behavior,
        )

    def with_params(self, params):
        """Returns a new NEUVECoordinate instance with updated weights."""
        return NEUVECoordinate(
            params=params,
            hidden_dim=self.model.hidden_dim,
            condition_on_terrain=self.condition_on_terrain,
            legacy_behavior=self.legacy_behavior,
        )

    def save_weights(self, filepath):
        """Saves the neural coordinate weights to binary file."""
        bytes_data = serialization.to_bytes(self.params)
        with open(filepath, 'wb') as f:
            f.write(bytes_data)

    @classmethod
    def from_file(cls, filepath, hidden_dim=64, condition_on_terrain=True,
                  legacy_behavior=False):
        """Loads a trained NEUVECoordinate from binary file."""
        instance = cls(
            hidden_dim=hidden_dim,
            condition_on_terrain=condition_on_terrain,
            legacy_behavior=legacy_behavior,
        )
        with open(filepath, 'rb') as f:
            bytes_data = f.read()
        loaded_params = serialization.from_bytes(instance.params, bytes_data)
        return cls(
            params=loaded_params,
            hidden_dim=hidden_dim,
            condition_on_terrain=condition_on_terrain,
            legacy_behavior=legacy_behavior,
        )


    def __call__(self, xi, zeta, h, Lz):
        return self.integral_transform(xi, zeta, h, Lz)
