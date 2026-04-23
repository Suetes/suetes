import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import optax
from atmos_jax.core import StaggeredGrid
from atmos_jax.core.neural import StandardMLP
from atmos_jax.core.transforms import IntegralNeuralTransform

# --- CONFIGURATION ---
Lx, Lz = 50000.0, 21000.0
nx, nz = 101, 33 

def get_jagged_mountain(key):
    k1, k2, k3 = jax.random.split(key, 3)
    hc = jax.random.uniform(k1, minval=900.0, maxval=1100.0)
    ac = jax.random.uniform(k2, minval=2000.0, maxval=3000.0)
    lam = jax.random.uniform(k3, minval=1800.0, maxval=2200.0)
    return (hc, ac, lam)

def loss_fn_regularized(nn_params, model_apply, h_params):
    hc, ac, lam = h_params
    h_func = lambda x: hc * jnp.exp(-((x - Lx/2.0)/ac)**2) * jnp.cos(jnp.pi * (x - Lx/2.0) / lam)**2
    transform = IntegralNeuralTransform(model_apply, nn_params)
    grid = StaggeredGrid(nx, nz, Lx, Lz, h_func, transform=transform)
    
    # Jacobian J(x, z)
    J = grid.metrics['m']['z_zeta']
    
    # 1. HORIZONTAL SMOOTHNESS (Flatten lines)
    dJ_dx = (J[1:, :] - J[:-1, :]) / grid.dx
    z_weight = (grid.Z_m[1:, :] / Lz) ** 1.5
    smoothness_loss = jnp.mean((dJ_dx * z_weight)**2) * 1e10
    
    # 2. VERTICAL SMOOTHNESS (Prevent Layer Cake)
    # Penalize rapid changes in thickness between layers
    # dJ/d_zeta approx (J[k+1] - J[k])
    dJ_dz = (J[:, 1:] - J[:, :-1])
    # High weight to kill the sawtooth pattern immediately
    vertical_loss = jnp.mean(dJ_dz**2) * 1e5
    
    # 3. COMPRESSION KICKSTART (Beat SLEVE)
    # Guide J to 0.25 near the mountain
    h_topo = grid.h_func(grid.X_m[:, 0])
    topo_mask = jnp.where(h_topo > 100.0, 1.0, 0.0)[:, None]
    target_J = 0.25
    kickstart_loss = jnp.mean(topo_mask * ((J - target_J) * (1.0 - grid.Z_m/Lz))**2) * 1e3
    
    # 4. SAFETY BARRIER (Prevent Folding)
    # Use Softplus for smooth gradient near the cliff
    barrier_loss = jnp.mean(jax.nn.softplus(500.0 * (0.2 - J))) * 1e8
    
    total = smoothness_loss + vertical_loss + kickstart_loss + barrier_loss
    return total, (smoothness_loss, vertical_loss, kickstart_loss)

def train():
    print("Initializing Vertically Regularized Training...")
    model = StandardMLP(width=64, depth=3)
    key = jax.random.PRNGKey(42)
    
    # Start Standard (Random weights give flexibility)
    params = model.init(key, jnp.array([[0.5]]))
    
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(5e-3))
    opt_state = optimizer.init(params)
    
    dataset = [get_jagged_mountain(k) for k in jax.random.split(key, 20)]
    
    @jax.jit
    def step(p, opt, h):
        (loss, components), g = jax.value_and_grad(loss_fn_regularized, has_aux=True)(p, model.apply, h)
        u, opt = optimizer.update(g, opt)
        return optax.apply_updates(p, u), opt, loss, components

    print(f"{'Epoch':<6} | {'Smooth':<10} | {'Vert':<10} | {'Kick':<10} | {'Total':<10}")
    
    for epoch in range(501):
        t_loss, t_sm, t_vert, t_kick = 0.0, 0.0, 0.0, 0.0
        for h in dataset:
            params, opt_state, l, (sm, vert, kick) = step(params, opt_state, h)
            t_loss += l; t_sm += sm; t_vert += vert; t_kick += kick
        
        if epoch % 50 == 0:
            N = len(dataset)
            print(f"{epoch:<6} | {t_sm/N:<10.4f} | {t_vert/N:<10.4f} | {t_kick/N:<10.4f} | {t_loss/N:<10.4f}")
            
    with open("neural_euler_cpu_ultra.pkl", "wb") as f:
        pickle.dump(params, f)
    print("Training complete.")

if __name__ == "__main__":
    train()