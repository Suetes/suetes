import jax
import jax.numpy as jnp

# Constants
MAX_MOUNTAINS = 10

def get_complex_topo(x, h0s, as_, lams, xcs, scale_factor=1.0):
    """
    Generates topography as a sum of multiple Schaer mountains.
    Supports broadcasting for batch generation.
    """
    # Expand x to (nx, 1)
    x_col = jnp.expand_dims(x, axis=-1)
    
    # Distances
    x_diff = x_col - xcs
    x_dist = jnp.abs(x_diff)
    
    # Mask for compact support
    mask = x_dist <= as_
    
    # Individual Mountains
    h_star = jnp.where(mask, 
                       h0s * jnp.cos(jnp.pi * x_diff / (2 * as_))**2, 
                       0.0)
    mountains = h_star * jnp.cos(jnp.pi * x_diff / lams)**2
    
    # Sum
    total_h = jnp.sum(mountains, axis=-1)
    
    # Safety Scaling (Limit max height to avoid unavoidable CFL crashes)
    # scale_factor allows curriculum learning (growing mountains)
    max_h = jnp.max(total_h)
    safety_scale = jnp.where(max_h > 4500.0, 4500.0 / (max_h + 1e-6), 1.0)
    
    return total_h * safety_scale * scale_factor

def create_mixed_dataset(num_samples=30, lx=300000.0):
    """
    Creates a balanced dataset of Smooth, Standard, and Jagged topographies.
    Returns list of parameter tuples (h0s, as_, lams, xcs).
    """
    key = jax.random.PRNGKey(123)
    dataset = []
    n_per_type = num_samples // 3
    
    for i in range(num_samples):
        key, k_n, k_h, k_a, k_l, k_x = jax.random.split(key, 6)
        num_m = jax.random.randint(k_n, (), 1, 8) # 1 to 7 mountains
        
        # Centers spread across domain
        active_x = jax.random.uniform(k_x, (MAX_MOUNTAINS,), 
                                      minval=lx*0.1, maxval=lx*0.9)
        
        # Define Types
        if i < n_per_type: # SMOOTH
            active_h = jax.random.uniform(k_h, (MAX_MOUNTAINS,), minval=500.0, maxval=3000.0)
            active_a = jax.random.uniform(k_a, (MAX_MOUNTAINS,), minval=40000.0, maxval=80000.0)
            active_l = jax.random.uniform(k_l, (MAX_MOUNTAINS,), minval=12000.0, maxval=25000.0)
        elif i < 2 * n_per_type: # STANDARD
            active_h = jax.random.uniform(k_h, (MAX_MOUNTAINS,), minval=500.0, maxval=4000.0)
            active_a = jax.random.uniform(k_a, (MAX_MOUNTAINS,), minval=15000.0, maxval=40000.0)
            active_l = jax.random.uniform(k_l, (MAX_MOUNTAINS,), minval=8000.0, maxval=12000.0)
        else: # JAGGED
            active_h = jax.random.uniform(k_h, (MAX_MOUNTAINS,), minval=500.0, maxval=4000.0)
            active_a = jax.random.uniform(k_a, (MAX_MOUNTAINS,), minval=8000.0, maxval=20000.0)
            active_l = jax.random.uniform(k_l, (MAX_MOUNTAINS,), minval=5000.0, maxval=9000.0)

        # Masking inactive mountains
        mask = jnp.arange(MAX_MOUNTAINS) < num_m
        h0s = jnp.where(mask, active_h, 0.0)
        as_ = jnp.where(mask, active_a, 1.0)
        lams = jnp.where(mask, active_l, 1.0)
        xcs = jnp.where(mask, active_x, 0.0)
        
        dataset.append((h0s, as_, lams, xcs))
        
    return dataset

def create_single_mountain_dataset(h0=3000.0, a=25000.0, lam=8000.0, lx=300000.0):
    """Returns a single-item dataset for the standard Schaer test."""
    h0s = jnp.zeros(MAX_MOUNTAINS).at[0].set(h0)
    as_ = jnp.ones(MAX_MOUNTAINS).at[0].set(a)
    lams = jnp.ones(MAX_MOUNTAINS).at[0].set(lam)
    xcs = jnp.zeros(MAX_MOUNTAINS).at[0].set(lx/2.0)
    return [(h0s, as_, lams, xcs)]