import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from suetes.core import StaggeredGrid
from suetes.core.transforms import BaseTransform

# --- 1. MODEL ---
class TripleOutputMLP(nn.Module):
    width: int
    depth: int
    @nn.compact
    def __call__(self, z_norm):
        x = z_norm
        for _ in range(self.depth):
            x = nn.Dense(self.width)(x)
            x = nn.tanh(x)
        
        # Output 3 values: [lambda1, lambda2, cutoff_signal]
        x = nn.Dense(3)(x)
        lambdas = jax.nn.softplus(x[..., :2]) * 1e-3 + 1e-5
        cutoff = jax.nn.sigmoid(x[..., 2:]) 
        return lambdas, cutoff

class AnchoredCutoffTransform(BaseTransform):
    def __init__(self, apply_fn, params, h1_func):
        self.apply_fn = apply_fn
        self.params = params
        self.h1_func = h1_func

    def __call__(self, xi, zeta, h_total, Lz):
        h1 = self.h1_func(xi)
        h2 = h_total - h1
        
        y_grid = zeta / Lz
        lambdas, raw_cutoff = self.apply_fn(self.params, y_grid[..., None])
        lambdas = lambdas.squeeze()
        raw_cutoff = raw_cutoff.squeeze()
        
        lam1 = lambdas[..., 0]
        lam2 = lambdas[..., 1]
        
        # --- CRITICAL FIX: ANCHOR CUTOFF ---
        # Force cutoff to 1.0 at surface (y=0)
        # We blend the network output with 1.0 using a fixed exponential decay
        # Anchor scale = 500m (approx 2% of domain)
        anchor_weight = jnp.exp(-zeta / 500.0)
        # smooth_cutoff starts at 1.0 and blends to raw_cutoff
        cutoff = anchor_weight * 1.0 + (1.0 - anchor_weight) * raw_cutoff
        
        linear_term = (1.0 - y_grid)
        
        b1 = linear_term * jnp.exp(-lam1 * zeta) * cutoff
        b2 = linear_term * jnp.exp(-lam2 * zeta) * cutoff
        
        return zeta + h1 * b1 + h2 * b2

# --- CONFIGURATION ---
Lx, Lz = 50000.0, 21000.0
nx, nz = 101, 33 

def get_jagged_mountain(key):
    k1, k2, k3 = jax.random.split(key, 3)
    hc = jax.random.uniform(k1, minval=900.0, maxval=1100.0)
    ac = jax.random.uniform(k2, minval=2000.0, maxval=3000.0)
    lam = jax.random.uniform(k3, minval=1800.0, maxval=2200.0)
    return (hc, ac, lam)

def get_h1_clean(x, h, sigma_meters=10000.0):
    dx = Lx / (nx - 1)
    sigma = sigma_meters / dx
    window = int(sigma * 4)
    r = jnp.arange(-window, window + 1)
    kernel = jnp.exp(-r**2 / (2 * sigma**2))
    kernel = kernel / jnp.sum(kernel)
    h_pad = jnp.pad(h, (window, window), mode='wrap')
    return jnp.convolve(h_pad, kernel, mode='valid')

def loss_fn_anchored(nn_params, model_apply, h_params):
    hc, ac, lam = h_params
    h_func = lambda x: hc * jnp.exp(-((x - Lx/2.0)/ac)**2) * jnp.cos(jnp.pi * (x - Lx/2.0) / lam)**2
    
    x_grid = jnp.linspace(0, Lx, nx)
    h_vals = h_func(x_grid)
    h1_vals = get_h1_clean(x_grid, h_vals)
    h1_func = lambda x: jnp.interp(x, x_grid, h1_vals)
    
    transform = AnchoredCutoffTransform(model_apply, nn_params, h1_func)
    grid = StaggeredGrid(nx, nz, Lx, Lz, h_func, transform=transform)
    
    J = grid.metrics['m']['z_zeta']
    
    # 1. Smoothness
    dJ_dx = (J[1:, :] - J[:-1, :]) / grid.dx
    z_weight = (grid.Z_m[1:, :] / Lz) ** 1.5
    loss_smooth = jnp.mean((dJ_dx * z_weight)**2) * 1e10
    
    # 2. Vertical Regularization (Smooth transition)
    dJ_dz = (J[:, 1:] - J[:, :-1])
    loss_vert = jnp.mean(dJ_dz**2) * 1e5
    
    # 3. Barrier
    loss_bar = jnp.mean(jax.nn.relu(0.02 - J)**2) * 1e12
    
    return loss_smooth + loss_vert + loss_bar, (loss_smooth, loss_vert)

def train():
    print("Initializing Anchored Cutoff Training...")
    model = TripleOutputMLP(width=64, depth=3)
    key = jax.random.PRNGKey(42)
    
    params = model.init(key, jnp.array([[0.5]]))
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(5e-3))
    opt_state = optimizer.init(params)
    
    dataset = [get_jagged_mountain(k) for k in jax.random.split(key, 30)]
    
    @jax.jit
    def step(p, opt, h):
        (loss, (sm, vert)), g = jax.value_and_grad(loss_fn_anchored, has_aux=True)(p, model.apply, h)
        u, opt = optimizer.update(g, opt)
        return optax.apply_updates(p, u), opt, loss, sm, vert

    print(f"{'Epoch':<6} | {'Smooth':<10} | {'Vert':<10} | {'Total':<10}")
    
    for epoch in range(501):
        t_loss, t_sm, t_vert = 0.0, 0.0, 0.0
        for h in dataset:
            params, opt_state, l, sm, vert = step(params, opt_state, h)
            t_loss += l; t_sm += sm; t_vert += vert
        
        if epoch % 50 == 0:
            print(f"{epoch:<6} | {t_sm/30:<10.4f} | {t_vert/30:<10.4f} | {t_loss/30:<10.4f}")
            
    with open("neural_sleve_anchored.pkl", "wb") as f:
        pickle.dump(params, f)
    print("Training complete.")

if __name__ == "__main__":
    train()