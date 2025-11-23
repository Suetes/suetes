import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import optax

# Force CPU
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

from atmos_jax.core import StaggeredGrid
from atmos_jax.dynamics.advection import SchaerAdvection
from atmos_jax.core.neural import StandardMLP
from atmos_jax.core.transforms import IntegralNeuralTransform
from atmos_jax.utils.datasets import get_complex_topo, create_mixed_dataset

# Configuration
Lx, Lz = 300000.0, 25000.0
nx, nz = 300, 50
dt, t_end = 25.0, 5000.0
u0 = 20.0

# Training parameters
epochs = 500
num_samples = 30

# Schaer test case specific logic
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

# --- Solvers & Loss ---
def run_simulation_scan(grid):
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

def loss_fn_single(nn_params, model_apply, topo_params):
    # Unpack just the numerical parameters
    h0s, as_, lams, xcs = topo_params
    
    def current_h(x): 
        return get_complex_topo(x, h0s, as_, lams, xcs)
    
    transform = IntegralNeuralTransform(model_apply, nn_params)
    grid = StaggeredGrid(nx, nz, Lx, Lz, current_h, transform=transform)
    
    J = grid.metrics['m']['z_zeta']
    is_stable = jnp.min(J) > 1e-5
    
    # Weak regularization
    reg_loss = 1e-7 * jnp.sum(jax.nn.relu(-J)) 
    
    def physics_loss(_):
        rho_final = run_simulation_scan(grid)
        rho_true = get_analytic_rho(grid)
        l2 = jnp.sqrt(jnp.mean((rho_final - rho_true)**2))
        return l2

    def crash_loss(_): 
        return 1.0

    phys_l2 = jax.lax.cond(is_stable, physics_loss, crash_loss, None)
    phys_l2 = jnp.where(jnp.isnan(phys_l2), 1.0, phys_l2)
    
    return phys_l2 + reg_loss, phys_l2

def train():
    print("Initializing training...")
    model = StandardMLP(width=64, depth=3)
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.array([[0.5]]))
    # Perturb initialization
    params = jax.tree_map(lambda x: x + 0.02 * jax.random.normal(key, x.shape), params)
    
    dataset = create_mixed_dataset(key=jax.random.PRNGKey(123), num_samples=num_samples)
    print(f"Generated {len(dataset)} samples.")
    
    optimizer = optax.chain(optax.zero_nans(), optax.clip_by_global_norm(1.0), optax.adam(1e-3))
    opt_state = optimizer.init(params)
    
    @jax.jit
    def train_step(params, opt_state, topo_params):
        (loss, l2), grads = jax.value_and_grad(loss_fn_single, has_aux=True)(
            params, model.apply, topo_params
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, l2

    print(f"{'Epoch':<6} | {'Avg Loss':<12} | {'Avg L2':<12} | {'Crashes':<8}")
    print("-" * 50)
    
    for epoch in range(epochs):
        epoch_losses, epoch_l2s, crashes = [], [], 0
        for data in dataset:
            params, opt_state, loss_val, l2_val = train_step(params, opt_state, data['params'])
            
            if l2_val >= 0.9: 
                crashes += 1
            else: 
                epoch_losses.append(loss_val)
                epoch_l2s.append(l2_val)
        
        avg_l2 = np.mean(epoch_l2s) if epoch_l2s else 999.0
        if epoch % 10 == 0:
            print(f"{epoch:<6} | {np.mean(epoch_losses):<12.6f} | {avg_l2:<12.6f} | {crashes:<8}")

    print("Saving parameters...")
    with open("neural_complex_params.pkl", "wb") as f: 
        pickle.dump(params, f)

if __name__ == "__main__":
    train()