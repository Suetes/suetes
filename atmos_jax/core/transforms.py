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

class Sleve(BaseTransform):
    """
    SLEVE (Smooth LEvel VErtical) Coordinate (Schär et al., 2002).
    Applies Sinh-based decay with exponent 'n' to smooth small-scale features.
    
    For this general implementation, we apply the decay to the total 'h'.
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

class NeuralTransform(BaseTransform):
    """
    Neural Coordinate: The vertical decay function b(zeta) is learned.
    z = zeta + h * b_NN(zeta/Lz)
    
    Enforces boundary conditions: b(0) = 1, b(Lz) = 0
    """
    def __init__(self, nn_apply_fn, params):
        self.apply_fn = nn_apply_fn
        self.params = params

    def __call__(self, xi, zeta, h, Lz):
        # 1. Normalized vertical coordinate Y in [0, 1]
        Y = zeta / Lz
        
        # 2. Define scalar wrapper for the network
        # The NN predicts N(y). We must normalize it to get b(y).
        # b(y) = 1 - (N(y) - N(0)) / (N(1) - N(0))
        def get_decay(y_val):
            # Raw network output
            N_y = self.apply_fn(self.params, y_val)
            
            # Reference points for BCs
            N_0 = self.apply_fn(self.params, 0.0)
            N_1 = self.apply_fn(self.params, 1.0)
            
            numerator = N_y - N_0
            denominator = N_1 - N_0
            
            # Numerical safety for denominator
            denominator = jnp.where(jnp.abs(denominator) < 1e-6, 1e-6, denominator)
            
            return 1.0 - (numerator / denominator)

        # 3. Vectorize over the grid
        # This applies the NN logic element-wise to the 2D grid arrays
        b_vals = jnp.vectorize(get_decay)(Y)
        
        # 4. Final Transform
        return zeta + h * b_vals