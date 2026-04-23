import jax.numpy as jnp
import jax

def get_analytic_w(grid, h_params, flow_params):
    """
    Computes the Linear Non-Hydrostatic Analytic Solution.
    Target for the Neural Network training.
    """
    X = grid.X_m
    Z = grid.Z_m
    Lx = grid.Lx
    
    # Unpack parameters
    hc, ac, lam = h_params
    U, N = flow_params['U'], flow_params['N']
    
    # Discretize Wavenumber k
    k_min = 2.0 * jnp.pi / 100000.0
    k_max = 2.0 * jnp.pi / 250.0 
    num_k = 400
    ks = jnp.linspace(k_min, k_max, num_k)
    dk = ks[1] - ks[0]
    
    # 1. Fourier Transform of Topography (Gaussian * Cosine^2)
    k0 = 2.0 * jnp.pi / lam
    
    def gauss_ft(k):
        return jnp.exp(-(k * ac / 2.0)**2)
        
    factor = hc * jnp.sqrt(jnp.pi) * ac / 2.0
    H_k = factor * (gauss_ft(ks) + 0.5*gauss_ft(ks - k0) + 0.5*gauss_ft(ks + k0))
    
    # 2. Vertical Wavenumber m (Real-valued logic)
    # m^2 = N^2/U^2 - k^2
    # If m^2 > 0: Propagating (Oscillatory)
    # If m^2 < 0: Evanescent (Decaying)
    l_sq = (N / U)**2
    m_sq = l_sq - ks**2
    
    # Split into propagating (m) and decaying (mu) parts
    is_prop = m_sq > 0
    m_real = jnp.sqrt(jnp.maximum(0.0, m_sq))  # Vertical wavenumber
    mu_imag = jnp.sqrt(jnp.maximum(0.0, -m_sq)) # Decay rate
    
    # 3. Compute w(x,z) summation
    # Formula: w ~ Real( i * k * U * H_k * exp(i(kx + mz)) )
    # This simplifies to: -k * U * H_k * exp(-mu*z) * sin(kx + m*z)
    
    X_centered = X - Lx/2.0
    
    # Expand for broadcasting (nx, nz, num_k)
    X_exp = jnp.expand_dims(X_centered, -1)
    Z_exp = jnp.expand_dims(Z, -1)
    
    # Calculate terms without complex numbers
    # If propagating: exp(-0) * sin(kx + mz)
    # If evanescent: exp(-mu*z) * sin(kx + 0)
    decay_term = jnp.exp(-mu_imag * Z_exp)
    phase_term = jnp.sin(ks * X_exp + m_real * Z_exp)
    
    term = -ks * U * H_k * decay_term * phase_term * (dk / jnp.pi)
    
    # Sum over k
    return jnp.sum(term, axis=-1)