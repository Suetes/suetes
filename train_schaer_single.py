import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

# Force CPU
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

from atmos_jax.core import StaggeredGrid
from atmos_jax.dynamics.advection import SchaerAdvection
from atmos_jax.core.transforms import BaseTransform

# ==============================================================================
# 1. Configuration (Standard Schär Case)
# ==============================================================================
Lx, Lz = 300000.0, 25000.0
nx, nz = 300, 50
dt, t_end = 25.0, 5000.0
u0 = 20.0

# --- Physics ---
def h_schaer(x):
    x_c = x - Lx/2.0
    h0 = 3000.0
    a = 25000.0
    lam = 8000.0
    h_star = jnp.where(jnp.abs(x_c) <= a, h0 * jnp.cos(jnp.pi * x_c / (2*a))**2, 0.0)
    return h_star * jnp.cos(jnp.pi * x_c / lam)**2

def get_u_profile(z):
    z1, z2 = 4000.0, 5000.0
    val_2 = u0 * jnp.sin(jnp.pi * (z - z1) / (2 * (z2 - z1)))**2
    u = jnp.where(z <= z1, 0.0, 0.0)
    u = jnp.where((z > z1) & (z < z2), val_2, u)
    u = jnp.where(z >= z2, u0, u)
    return u

def init_rho(grid):
    x_blob = 100000.0
    z0, Ax, Az = 9000.0, 25000.0, 3000.0
    r = jnp.sqrt( ((grid.X_m - x_blob)/Ax)**2 + ((grid.Z_m - z0)/Az)**2 )
    return jnp.where(r <= 1.0, jnp.cos(jnp.pi * r / 2.0)**2, 0.0)

def get_analytic_rho(grid):
    x_blob = 100000.0
    z0, Ax, Az = 9000.0, 25000.0, 3000.0
    u_z = get_u_profile(grid.Z_m)
    X_back = jnp.mod(grid.X_m - u_z * t_end, Lx)
    dx = jnp.abs(X_back - x_blob)
    dx = jnp.minimum(dx, Lx - dx)
    r = jnp.sqrt( (dx/Ax)**2 + ((grid.Z_m - z0)/Az)**2 )
    return jnp.where(r <= 1.0, jnp.cos(jnp.pi * r / 2.0)**2, 0.0)

# ==============================================================================
# 2. Model & Transform
# ==============================================================================
class StandardMLP(nn.Module):
    width: int
    depth: int
    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth - 1):
            x = nn.Dense(self.width)(x)
            x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x

class IntegralNeuralTransform(BaseTransform):
    def __init__(self, nn_apply_fn, params):
        self.apply_fn = nn_apply_fn
        self.params = params
    def __call__(self, xi, zeta, h, Lz):
        y_grid = jnp.linspace(0, 1.0, 101)[:, None]
        dy = 1.0 / 100.0
        density = jax.nn.softplus(self.apply_fn(self.params, y_grid).squeeze()) + 0.1
        cdf = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(density) * dy])
        cdf_norm = cdf / cdf[-1]
        Y_actual = zeta / Lz
        b_shape = jnp.interp(Y_actual, jnp.linspace(0, 1, 102), cdf_norm)
        return zeta + h * (1.0 - b_shape)

# ==============================================================================
# 3. Solver & Loss
# ==============================================================================
class TrainingGrid(StaggeredGrid):
    def _compute_analytic_metrics(self, Xi, Zeta):
        Z_phys = self._apply_transform(Xi, Zeta)
        z_xi = (jnp.roll(Z_phys, -1, axis=0) - jnp.roll(Z_phys, 1, axis=0)) / (2 * self.dx)
        Z_pad = jnp.pad(Z_phys, ((0,0), (1,1)), mode='edge')
        z_zeta = (Z_pad[:, 2:] - Z_pad[:, :-2]) / (2 * self.dz)
        return {'z_xi': z_xi, 'z_zeta': z_zeta}

def run_simulation(grid):
    model = SchaerAdvection(grid, get_u_profile)
    rho_init = init_rho(grid)
    def rk3_step(carry, _):
        rho = carry
        k1 = rho + dt * model.compute_rhs({'rho': rho})['rho']
        k2 = 0.75*rho + 0.25*k1 + 0.25*dt*model.compute_rhs({'rho': k1})['rho']
        k3 = (1.0/3.0)*rho + (2.0/3.0)*k2 + (2.0/3.0)*dt*model.compute_rhs({'rho': k2})['rho']
        return k3, None
    rho_final, _ = jax.lax.scan(rk3_step, rho_init, None, length=int(t_end/dt))
    return rho_final

def loss_fn(nn_params, model_apply):
    transform = IntegralNeuralTransform(model_apply, nn_params)
    grid = TrainingGrid(nx, nz, Lx, Lz, h_schaer, transform=transform)
    
    J = grid.metrics['m']['z_zeta']
    reg_loss = 1e-7 * jnp.sum(jax.nn.relu(-J))
    
    rho_final = run_simulation(grid)
    rho_true = get_analytic_rho(grid)
    l2 = jnp.sqrt(jnp.mean((rho_final - rho_true)**2))
    
    return l2 + reg_loss, l2

# ==============================================================================
# 4. Optimization Loop
# ==============================================================================
def train():
    print("Optimizing Single Schaer Case (1000 Epochs)...")
    
    # Increased Capacity
    model = StandardMLP(width=64, depth=3)
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.array([[0.5]]))
    
    # Cosine Decay Schedule
    total_epochs = 1500
    scheduler = optax.cosine_decay_schedule(init_value=2e-3, decay_steps=total_epochs)
    
    optimizer = optax.chain(
        optax.zero_nans(),
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=scheduler)
    )
    opt_state = optimizer.init(params)
    
    @jax.jit
    def step(params, opt_state):
        (loss, l2), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, model.apply)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss, l2

    print(f"{'Epoch':<6} | {'Loss':<12} | {'L2 Error':<12}")
    print("-" * 36)

    best_l2 = 1e9
    best_params = params

    for i in range(total_epochs + 1):
        params, opt_state, loss, l2 = step(params, opt_state)
        
        if l2 < best_l2:
            best_l2 = l2
            best_params = params

        if i % 50 == 0:
            print(f"{i:<6} | {loss:<12.6f} | {l2:<12.6f}")
            
    print(f"Best L2: {best_l2:.6e}")
    with open("neural_single_params.pkl", "wb") as f:
        pickle.dump(best_params, f)

if __name__ == "__main__":
    train()