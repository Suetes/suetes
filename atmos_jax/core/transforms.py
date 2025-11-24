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
    z = zeta + h * (1 - zeta/Lz) * exp(-zeta / scale_height)
    """
    def __init__(self, scale_height=5000.0):
        self.scale_height = scale_height

    def __call__(self, xi, zeta, h, Lz):
        # Decay factor ensures boundary condition at top (z=Lz) is met
        decay = (1.0 - zeta/Lz) * jnp.exp(-zeta / self.scale_height)
        return zeta + h * decay

class SleveSimple(BaseTransform):
    """
    SLEVE (Smooth LEvel VErtical) Coordinate (Schär et al., 2002).
    Applies Sinh-based decay with exponent 'n' to smooth small-scale features.
    This is a simplified version with just a single topography scale
    """
    def __init__(self, scale_s=4000.0, scale_l=15000.0, n=1.35):
        self.ss = scale_s
        self.sl = scale_l
        self.n = n

    def __call__(self, xi, zeta, h, Lz):
        # Sinh-based decay function (b_s)
        # b(zeta) = sinh((Lz - zeta)/S) / sinh(Lz/S)
        b_s = jnp.sinh((Lz - zeta)/self.ss) / jnp.sinh(Lz/self.ss)
        
        # Apply decay with exponent n
        # Standard SLEVE: z = zeta + h * (b_s)^n
        return zeta + h * (b_s**self.n)

class Sleve(BaseTransform):
    """
    General SLEVE Coordinate (Schär et al., 2002).
    z(Z) = Z + h1 * b1(Z) + h2 * b2(Z)
    """
    def __init__(self, h1_func, s1=15000.0, s2=2500.0):
        """
        Args:
            h1_func: Function taking (x) -> returns large-scale topo h1.
            s1: Scale height for large-scale features (approx 15km).
            s2: Scale height for small-scale features (approx 2.5km).
        """
        self.h1_func = h1_func
        self.s1 = s1
        self.s2 = s2

    def _b_func(self, zeta, Lz, s):
        # Eq (15): b(Z) = sinh((H-Z)/s) / sinh(H/s)
        return jnp.sinh((Lz - zeta)/s) / jnp.sinh(Lz/s)

    def __call__(self, xi, zeta, h, Lz):
        # 1. Calculate h1 (Large Scale) at current x locations
        # xi matches the shape of h, so we can pass it to the function
        h1 = self.h1_func(xi)
        
        # 2. Derive h2 (Small Scale)
        # Eq (13): h = h1 + h2  =>  h2 = h - h1
        h2 = h - h1
        
        # 3. Calculate Decay Functions
        b1 = self._b_func(zeta, Lz, self.s1)
        b2 = self._b_func(zeta, Lz, self.s2)
        
        # 4. Eq (14): z = Z + h1*b1 + h2*b2
        return zeta + h1 * b1 + h2 * b2

class IntegralNeuralTransform(BaseTransform):
    """
    Robust Neural Coordinate.
    Predicts a positive 'density' field and integrates it to get the decay function b(zeta).
    Guarantees z(0) = h and z(Lz) = Lz (Monotonic, No Crossing).
    """
    def __init__(self, nn_apply_fn, params):
        self.apply_fn = nn_apply_fn
        self.params = params

    def __call__(self, xi, zeta, h, Lz):
        # 1. Define Integration Grid (Normalized 0->1)
        # We use a fixed grid for integration stability
        y_grid = jnp.linspace(0, 1.0, 101)[:, None]
        dy = 1.0 / 100.0
        
        # 2. Get Density from NN (Must be positive)
        # Softplus + epsilon ensures strictly positive density -> strict monotonicity
        raw_out = self.apply_fn(self.params, y_grid)
        density = jax.nn.softplus(raw_out.squeeze()) + 0.05
        
        # 3. Integrate to get Cumulative Distribution Function (CDF)
        # CDF(0) = 0, CDF(1) = Integral(density)
        cdf = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(density) * dy])
        
        # 4. Normalize CDF to get shape function s(y) where s(0)=0, s(1)=1
        cdf_norm = cdf / cdf[-1]
        
        # 5. Interpolate to actual vertical coordinate Y = zeta / Lz
        Y_actual = zeta / Lz
        s_values = jnp.interp(Y_actual, jnp.linspace(0, 1, 102), cdf_norm)
        
        # 6. Decay function b(y) = 1 - s(y)
        # b(0) = 1, b(1) = 0
        b_vals = 1.0 - s_values
        
        # 7. Transform: z = zeta + h * b(zeta)
        return zeta + h * b_vals