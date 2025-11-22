import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax

# Force CPU
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

from atmos_jax.core import StaggeredGrid
from atmos_jax.dynamics.advection import SchaerAdvection
from atmos_jax.core.neural import MonotonicMLP
from atmos_jax.core.transforms import NeuralTransform

# ==============================================================================
# 1. Physics Configuration
# ==============================================================================
Lx = 300000.0
Lz = 25000.0
nx = 300
nz = 50
dt = 25.0
t_end = 5000.0
u0 = 20.0

def get_h_topo(x, h0, a, lam):
    x_c = x - Lx/2.0
    h_star = jnp.where(jnp.abs(x_c) <= a, h0 * jnp.cos(jnp.pi * x_c / (2*a))**2, 0.0)
    return h_star * jnp.cos(jnp.pi * x_c / lam)**2

def get_u_profile(z):
    z1, z2 = 4000.0, 5000.0
    val_2 = u0 * jnp.sin(jnp.pi * (z - z1) / (2.0 * (z2 - z1)))**2
    u = jnp.where(z <= z1, 0.0, 0.0)
    u = jnp.where((z > z1) & (z < z2), val_2, u)
    u = jnp.where(z >= z2, u0, u)
    return u

def init_rho(grid):
    x_blob = -50000.0 + Lx/2.0
    z0, Ax, Az = 9000.0, 25000.0, 3000.0
    r = jnp.sqrt( ((grid.X_m - x_blob)/Ax)**2 + ((grid.Z_m - z0)/Az)**2 )
    return jnp.where(r <= 1.0, jnp.cos(jnp.pi * r / 2.0)**2, 0.0)

def get_analytic_rho(grid):
    x_blob = -50000.0 + Lx/2.0
    z0, Ax, Az = 9000.0, 25000.0, 3000.0
    u_z = get_u_profile(grid.Z_m)
    X_back = jnp.mod(grid.X_m - u_z * t_end, Lx)
    dx = jnp.abs(X_back - x_blob)
    dx = jnp.minimum(dx, Lx - dx)
    r = jnp.sqrt( (dx/Ax)**2 + ((grid.Z_m - z0)/Az)**2 )
    return jnp.where(r <= 1.0, jnp.cos(jnp.pi * r / 2.0)**2, 0.0)

# ==============================================================================
# 2. Solvers
# ==============================================================================
class TrainingGrid(StaggeredGrid):
    def _compute_analytic_metrics(self, Xi, Zeta):
        Z_phys = self._apply_transform(Xi, Zeta)
        z_xi = (jnp.roll(Z_phys, -1, axis=0) - jnp.roll(Z_phys, 1, axis=0)) / (2 * self.dx)
        Z_pad = jnp.pad(Z_phys, ((0,0), (1,1)), mode='edge')
        z_zeta = (Z_pad[:, 2:] - Z_pad[:, :-2]) / (2 * self.dz)
        return {'z_xi': z_xi, 'z_zeta': z_zeta}

def run_simulation_scan(grid):
    model = SchaerAdvection(grid, get_u_profile)
    rho_init = init_rho(grid)
    def rk3_step(carry, _):
        rho = carry
        rhs1 = model.compute_rhs({'rho': rho})['rho']
        k1 = rho + dt * rhs1
        rhs2 = model.compute_rhs({'rho': k1})['rho']
        k2 = 0.75 * rho + 0.25 * k1 + 0.25 * dt * rhs2
        rhs3 = model.compute_rhs({'rho': k2})['rho']
        k3 = (1.0/3.0) * rho + (2.0/3.0) * k2 + (2.0/3.0) * dt * rhs3
        return k3, None
    steps = int(t_end / dt)
    rho_final, _ = jax.lax.scan(rk3_step, rho_init, None, length=steps)
    return rho_final

# ==============================================================================
# 3. Robust Loss Function (Always Differentiable)
# ==============================================================================
def loss_fn(nn_params, model_apply, h0, a, lam):
    def current_h(x): return get_h_topo(x, h0, a, lam)
    
    transform = NeuralTransform(model_apply, nn_params)
    grid = TrainingGrid(nx, nz, Lx, Lz, current_h, transform=transform)
    
    # --- 1. Geometric Loss (Calculated regardless of stability) ---
    # Jacobian J = dz/d_zeta. Target ~1.0
    # Penalize J < 0.1 (Collapse)
    J = grid.metrics['m']['z_zeta']
    
    # "Soft Barrier": 1/J blows up smoothly as J->0. This provides a gradient 
    # pointing AWAY from collapse, even if physics isn't running.
    reg_loss = 0.001 * jnp.mean(1.0 / (jnp.abs(J) + 1e-6)) 
    
    # Add penalty for negative J (Inverted grid)
    neg_penalty = 10.0 * jnp.sum(jax.nn.relu(-J))
    
    total_geo_loss = reg_loss + neg_penalty

    # Stability Flag
    is_stable = jnp.min(J) > 1e-4
    
    # --- 2. Physics Loss (Conditional) ---
    def compute_physics_loss(_):
        rho_final = run_simulation_scan(grid)
        rho_true = get_analytic_rho(grid)
        l2_err = jnp.sqrt(jnp.mean((rho_final - rho_true)**2))
        return l2_err

    def return_penalty(_):
        # Just return a large constant, but gradients will come from geo_loss
        return 100.0 

    # Execute conditional physics
    phys_loss = jax.lax.cond(is_stable, compute_physics_loss, return_penalty, operand=None)
    
    # NaN Safety for physics
    phys_loss = jnp.where(jnp.isnan(phys_loss), 100.0, phys_loss)
    
    # --- 3. Total Loss ---
    # Gradients flow through total_geo_loss even if phys_loss is constant
    total_loss = phys_loss + total_geo_loss
    
    return total_loss, phys_loss

# ==============================================================================
# 4. Training Loop
# ==============================================================================
def train():
    print("Initializing General Neural Coordinate (Differentiable Failure Mode)...")
    model = MonotonicMLP(width=32, depth=3)
    key = jax.random.PRNGKey(2024)
    
    # Cold Start
    params = model.init(key, jnp.array([0.5]))
    params = jax.tree_map(lambda x: x + 0.05 * jax.random.normal(key, x.shape), params)
    
    total_steps = 1000
    optimizer = optax.chain(
        optax.zero_nans(),
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=0.001)
    )
    opt_state = optimizer.init(params)
    
    @jax.jit
    def update_step(params, opt_state, h0, a, lam):
        (loss, l2), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, model.apply, h0, a, lam
        )
        g_norm = optax.global_norm(grads)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, l2, g_norm

    print(f"{'Step':<6} | {'Total Loss':<10} | {'Phys Error':<10} | {'Grad Norm':<10} | {'Topo (h0, ...)'}")
    print("-" * 80)
    
    training_key = jax.random.PRNGKey(999)
    
    for i in range(total_steps + 1):
        training_key, subkey = jax.random.split(training_key)
        
        # Curriculum: Ramp up difficulty
        difficulty = min(1.0, i / 1000.0)
        h_min = 100.0 + 1000.0 * difficulty
        h_max = 500.0 + 3000.0 * difficulty # End at 3500m
        
        h0 = jax.random.uniform(subkey, minval=h_min, maxval=h_max)
        a = jax.random.uniform(subkey, minval=15000.0, maxval=40000.0)
        lam = jax.random.uniform(subkey, minval=6000.0, maxval=10000.0)
        
        params, opt_state, loss, l2, g_norm = update_step(params, opt_state, h0, a, lam)
        
        if i % 50 == 0:
            print(f"{i:<6} | {loss:<10.4f} | {l2:<10.4f} | {g_norm:<10.4f} | ({h0:.0f}, {a:.0f}, {lam:.0f})")

    print("Saving generalized parameters...")
    with open("neural_general_params.pkl", "wb") as f:
        pickle.dump(params, f)
    print("Done.")

if __name__ == "__main__":
    train()