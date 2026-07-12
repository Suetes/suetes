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
    Predicts a positive 'density' field conditioned on vertical height y AND local terrain h,
    and integrates it to get the decay function b(zeta, h).
    Guarantees z(0) = h and z(Lz) = Lz (Monotonic, No Crossing).
    """
    def __init__(self, nn_apply_fn, params):
        self.apply_fn = nn_apply_fn
        self.params = params

    def __call__(self, xi, zeta, h, Lz):
        y_eval = jnp.linspace(0, 1.0, 52)
        dy = 1.0 / 51.0
        y_grid = jnp.linspace(0, 1.0, 51)
        
        # Normalize h relative to 4000m reference mountain height
        h_norm = h / 4000.0
        
        if zeta.ndim == 3:
            # 3D field case: shape (nx, ny, nz)
            h_2d = h_norm[:, :, 0:1, None]  # (nx, ny, 1, 1)
            y_3d = jnp.broadcast_to(y_grid[None, None, :, None], (h.shape[0], h.shape[1], 51, 1))
            h_3d = jnp.broadcast_to(h_2d, (h.shape[0], h.shape[1], 51, 1))
            
            raw_out = self.apply_fn(self.params, y_3d, h_3d).squeeze(-1)  # (nx, ny, 51)
            density = jax.nn.softplus(raw_out) + 0.05
            
            cdf = jnp.cumsum(density, axis=-1) * dy
            zeros_init = jnp.zeros_like(cdf[..., :1])
            cdf_full = jnp.concatenate([zeros_init, cdf], axis=-1)
            cdf_norm = cdf_full / cdf_full[..., -1:]  # (nx, ny, 52)
            
            Y_actual = zeta / Lz
            
            def interp_col(y_col, cdf_col):
                return jnp.interp(y_col, y_eval, cdf_col)
            
            s_values = jax.vmap(jax.vmap(interp_col))(Y_actual, cdf_norm)
            b_vals = 1.0 - s_values
            return zeta + h * b_vals
        else:
            # 1D column / scalar case
            h_scalar = jnp.mean(h_norm)
            y_in = y_grid[:, None]
            h_in = jnp.full_like(y_in, h_scalar)
            raw_out = self.apply_fn(self.params, y_in, h_in).squeeze()
            density = jax.nn.softplus(raw_out) + 0.05
            cdf = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(density) * dy])
            cdf_norm = cdf / cdf[-1]
            Y_actual = zeta / Lz
            s_values = jnp.interp(Y_actual, y_eval, cdf_norm)
            b_vals = 1.0 - s_values
            return zeta + h * b_vals


class NEUVEMLP(nn.Module):
    """
    Spatially adaptive neural network for NEUVE vertical coordinate density prediction.
    Takes non-dimensional vertical height y in [0, 1] AND normalized local terrain height h_norm.
    Uses tanh activations for infinitely smooth adjoint gradients.
    """
    hidden_dim: int = 32

    @nn.compact
    def __call__(self, y, h_norm=None):
        if h_norm is not None:
            inputs = jnp.concatenate([y, h_norm], axis=-1)
        else:
            inputs = jnp.concatenate([y, jnp.zeros_like(y)], axis=-1)
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
    def __init__(self, params=None, hidden_dim=32, key_seed=42):
        self.model = NEUVEMLP(hidden_dim=hidden_dim)
        if params is None:
            key = jax.random.PRNGKey(key_seed)
            dummy_y = jnp.linspace(0, 1.0, 51)[:, None]
            dummy_h = jnp.zeros_like(dummy_y)
            self.params = self.model.init(key, dummy_y, dummy_h)
        else:
            self.params = params
        self.integral_transform = IntegralNeuralTransform(self.model.apply, self.params)

    def with_params(self, params):
        """Returns a new NEUVECoordinate instance with updated weights."""
        return NEUVECoordinate(params=params, hidden_dim=self.model.hidden_dim)

    def save_weights(self, filepath):
        """Saves the neural coordinate weights to binary file."""
        bytes_data = serialization.to_bytes(self.params)
        with open(filepath, 'wb') as f:
            f.write(bytes_data)

    @classmethod
    def from_file(cls, filepath, hidden_dim=32):
        """Loads a trained NEUVECoordinate from binary file."""
        instance = cls(hidden_dim=hidden_dim)
        with open(filepath, 'rb') as f:
            bytes_data = f.read()
        loaded_params = serialization.from_bytes(instance.params, bytes_data)
        return cls(params=loaded_params, hidden_dim=hidden_dim)


    def __call__(self, xi, zeta, h, Lz):
        return self.integral_transform(xi, zeta, h, Lz)