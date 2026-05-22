import jax
import jax.numpy as jnp

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
        b_s = jnp.sinh((Lz - zeta)/self.ss) / jnp.sinh(Lz/self.ss)
        return zeta + h * (b_s**self.n)


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

        # Simplified SLEVE decay using the stretched coordinate
        b_s = jnp.sinh((Lz - zeta_stretched) / self.ss) / jnp.sinh(Lz / self.ss)
        
        return zeta_stretched + h * (b_s ** self.n)


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
    Robust Neural Coordinate ensuring monotonicity.
    """
    def __init__(self, nn_apply_fn, params):
        self.apply_fn = nn_apply_fn
        self.params = params

    def __call__(self, xi, zeta, h, Lz):
        y_grid = jnp.linspace(0, 1.0, 101)[:, None]
        dy = 1.0 / 100.0
        
        raw_out = self.apply_fn(self.params, y_grid)
        density = jax.nn.softplus(raw_out.squeeze()) + 0.05
        
        cdf = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(density) * dy])
        cdf_norm = cdf / cdf[-1]
        
        Y_actual = zeta / Lz
        s_values = jnp.interp(Y_actual, jnp.linspace(0, 1, 102), cdf_norm)
        b_vals = 1.0 - s_values
        
        return zeta + h * b_vals