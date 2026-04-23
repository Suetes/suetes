import jax
import jax.numpy as jnp
import scipy.ndimage

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


def get_topo_components(x, h0s, as_, lams, xcs, scale_factor=1.0):
    """
    Returns (total_h, h1_large_scale) for the general SLEVE coordinate.
    h1 is defined as 0.5 * envelope (Eq 27 in Schaer et al., 2002).
    """
    # Expand x to (nx, 1)
    x_col = jnp.expand_dims(x, axis=-1)
    
    # Distances
    x_diff = x_col - xcs
    x_dist = jnp.abs(x_diff)
    
    # Mask for compact support
    mask = x_dist <= as_
    
    # Envelope (h_star)
    h_star = jnp.where(mask, 
                       h0s * jnp.cos(jnp.pi * x_diff / (2 * as_))**2, 
                       0.0)
    
    # 1. Calculate Total H (Standard)
    mountains = h_star * jnp.cos(jnp.pi * x_diff / lams)**2
    total_h = jnp.sum(mountains, axis=-1)
    
    # 2. Calculate Large Scale H1 (0.5 * Envelope)
    # Schaer 2002 Eq (27): h1 = 0.5 * h*(x)
    h1_mountains = 0.5 * h_star
    total_h1 = jnp.sum(h1_mountains, axis=-1)
    
    # 3. Apply Safety Scaling to BOTH
    # We must calculate scale based on total_h to preserve geometry
    max_h = jnp.max(total_h)
    safety_scale = jnp.where(max_h > 4500.0, 4500.0 / (max_h + 1e-6), 1.0)
    
    total_h_scaled = total_h * safety_scale * scale_factor
    total_h1_scaled = total_h1 * safety_scale * scale_factor
    
    return total_h_scaled, total_h1_scaled


def get_filtered_topo_components(x, h0s, as_, lams, xcs, sigma_meters=15000.0):
    """
    Splits topography using a Gaussian Low-Pass filter, consistent with 
    Schaer et al. (2002) real-case recommendations.
    """
    # 1. Generate the total topography first
    # (Re-use your existing logic to get the single H array)
    h_total = get_complex_topo(x, h0s, as_, lams, xcs)
    
    # 2. Determine filter width in grid points
    # sigma_meters should be roughly equal to s1 (e.g., 10-15km)
    dx = x[1] - x[0]
    sigma_pixels = sigma_meters / dx
    
    # 3. Compute h1 (Large Scale) via Gaussian Filter
    # This removes features smaller than sigma_meters
    h1 = scipy.ndimage.gaussian_filter1d(h_total, sigma=sigma_pixels, mode='wrap')
    
    # 4. Compute h2 (Small Scale Residual)
    h2 = h_total - h1
    
    return h_total, h1, h2


def get_auto_sleve_params(h1_array, h2_array, Lz, target_gamma=0.1):
    """
    Automatically determines BOTH s1 and s2 to create the flattest possible 
    grid that remains safe (invertible).
    """
    h1_max = jnp.max(jnp.abs(h1_array))
    h2_max = jnp.max(jnp.abs(h2_array))
    
    # --- 1. Tune s1 (Large Scale) ---
    # We want s1 as small as possible (flat grid) but large enough to avoid 
    # compressing layers too much at the bottom.
    # Heuristic: Allow h1 to consume ~40% of the invertibility budget.
    # term1 = (h1 / s1) ~ 0.4  => s1 ~ h1 / 0.4
    # We clamp s1 between 5km (very flat) and 15km (very safe).
    s1_ideal = h1_max / 0.4
    s1 = jnp.clip(s1_ideal, 5000.0, 15000.0)
    
    # --- 2. Calculate Used Budget ---
    # term1 = (h1/s1) * coth(Lz/s1)
    # coth(x) = 1/tanh(x)
    term1 = (h1_max / s1) * (1.0 / jnp.tanh(Lz / s1))
    
    # --- 3. Tune s2 (Small Scale) ---
    # Use remaining budget for h2
    budget_remaining = 1.0 - term1 - target_gamma
    
    # Safety check
    budget_remaining = jnp.maximum(budget_remaining, 0.05)
    
    # h2 / s2 = budget => s2 = h2 / budget
    s2 = h2_max / budget_remaining
    
    # Clamp s2: It shouldn't be larger than s1, and not tiny (<100m)
    s2 = jnp.clip(s2, 100.0, s1 * 0.5)
    
    return s1, s2


def create_mixed_dataset(key, num_samples=30, lx=300000.0):
    """
    Generates a balanced dataset of Smooth, Standard, and Jagged topographies.
    Returns a list of dictionaries containing parameters and type labels.
    """
    dataset = []
    n_per_type = int(jnp.ceil(num_samples / 3))
    
    for i in range(num_samples):
        key, k_n, k_h, k_a, k_l, k_x = jax.random.split(key, 6)
        
        # 1 to 6 mountains per sample
        num_m = jax.random.randint(k_n, (), 1, 7)
        
        # Random centers spread across domain and random heights from 500m to 3000m
        active_h = jax.random.uniform(k_h, (MAX_MOUNTAINS,), minval=500.0, maxval=3000.0)
        active_x = jax.random.uniform(k_x, (MAX_MOUNTAINS,), minval=50000.0, maxval=250000.0)
        
        # Determine Type
        if i < n_per_type: 
            type_label = "Smooth"
            active_a = jax.random.uniform(k_a, (MAX_MOUNTAINS,), minval=40000.0, maxval=80000.0)
            active_l = jax.random.uniform(k_l, (MAX_MOUNTAINS,), minval=12000.0, maxval=25000.0)
        elif i < 2 * n_per_type:
            type_label = "Standard"
            active_a = jax.random.uniform(k_a, (MAX_MOUNTAINS,), minval=15000.0, maxval=40000.0)
            active_l = jax.random.uniform(k_l, (MAX_MOUNTAINS,), minval=8000.0, maxval=12000.0)
        else:
            type_label = "Jagged"
            active_a = jax.random.uniform(k_a, (MAX_MOUNTAINS,), minval=8000.0, maxval=20000.0)
            active_l = jax.random.uniform(k_l, (MAX_MOUNTAINS,), minval=5000.0, maxval=9000.0)

        # Mask inactive mountains (h0=0)
        mask = jnp.arange(MAX_MOUNTAINS) < num_m
        h0s = jnp.where(mask, active_h, 0.0)
        as_ = jnp.where(mask, active_a, 1.0)
        lams = jnp.where(mask, active_l, 1.0)
        xcs = jnp.where(mask, active_x, 0.0)
        
        # Return params as a tuple for easy unpacking
        dataset.append({'params': (h0s, as_, lams, xcs), 'type': type_label})
        
    return dataset

def create_single_mountain_dataset(h0=3000.0, a=25000.0, lam=8000.0, lx=300000.0):
    """Returns a single-item dataset for the standard Schaer test."""
    h0s = jnp.zeros(MAX_MOUNTAINS).at[0].set(h0)
    as_ = jnp.ones(MAX_MOUNTAINS).at[0].set(a)
    lams = jnp.ones(MAX_MOUNTAINS).at[0].set(lam)
    xcs = jnp.zeros(MAX_MOUNTAINS).at[0].set(lx/2.0)
    return [(h0s, as_, lams, xcs)]