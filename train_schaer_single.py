import os
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax

# --- Force CPU ---
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

def h_schaer(x):
    x_c = x - Lx/2.0
    h0 = 3000.0
    a = 25000.0
    lam = 8000.0
    h_star = jnp.where(jnp.abs(x_c) <= a, h0 * jnp.cos(jnp.pi * x_c / (2*a))**2, 0.0)
    return h_star * jnp.cos(jnp.pi * x_c / lam)**2

def get_u_profile(z):
    z1 = 4000.0
    z2 = 5000.0
    val_2 = u0 * jnp.sin(jnp.pi * (z - z1) / (2.0 * (z2 - z1)))**2
    u = jnp.where(z <= z1, 0.0, 0.0)
    u = jnp.where((z > z1) & (z < z2), val_2, u)
    u = jnp.where(z >= z2, u0, u)
    return u

def init_rho(grid):
    x_blob = -50000.0 + Lx/2.0
    z0 = 9000.0
    Ax, Az = 25000.0, 3000.0
    r = jnp.sqrt( ((grid.X_m - x_blob)/Ax)**2 + ((grid.Z_m - z0)/Az)**2 )
    return jnp.where(r <= 1.0, jnp.cos(jnp.pi * r / 2.0)**2, 0.0)

def get_analytic_rho(grid):
    x_blob = -50000.0 + Lx/2.0
    z0 = 9000.0
    Ax, Az = 25000.0, 3000.0
    u_z = get_u_profile(grid.Z_m)
    X_back = jnp.mod(grid.X_m - u_z * t_end, Lx)
    dx = jnp.abs(X_back - x_blob)
    dx = jnp.minimum(dx, Lx - dx)
    r = jnp.sqrt( (dx/Ax)**2 + ((grid.Z_m - z0)/Az)**2 )
    return jnp.where(r <= 1.0, jnp.cos(jnp.pi * r / 2.0)**2, 0.0)

# ==============================================================================
# 2. Solvers & Grid
# ==============================================================================
class TrainingGrid(StaggeredGrid):
    """Forces Finite Difference metrics to expose discretization errors."""
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
# 3. Robust Loss Function
# ==============================================================================
def loss_fn(nn_params, model_apply, reg_weight):
    transform = NeuralTransform(model_apply, nn_params)
    grid = TrainingGrid(nx, nz, Lx, Lz, h_schaer, transform=transform)
    
    J = grid.metrics['m']['z_zeta']
    is_stable = jnp.min(J) > 1e-4 
    
    def compute_physics_loss(_):
        rho_final = run_simulation_scan(grid)
        rho_true = get_analytic_rho(grid)
        
        l2_err = jnp.sqrt(jnp.mean((rho_final - rho_true)**2))
        
        # Barrier Regularization
        reg_loss = reg_weight * jnp.mean(1.0 / (J + 1e-6))
        
        total = jnp.where(jnp.isnan(l2_err), 1e5, l2_err + reg_loss)
        return total, l2_err, reg_loss, rho_final

    def return_penalty(_):
        penalty = 1e5 
        dummy_rho = jnp.zeros_like(grid.X_m)
        return penalty, penalty, penalty, dummy_rho

    total_loss, l2_loss, reg_loss, rho_final = jax.lax.cond(
        is_stable, compute_physics_loss, return_penalty, operand=None
    )
    
    return total_loss, (l2_loss, reg_loss, rho_final, grid.X_m, grid.Z_m)

# ==============================================================================
# 4. Main Loop (Cold Start)
# ==============================================================================
def train():
    print("Initializing Neural Coordinate (Cold Start)...")
    model = MonotonicMLP(width=16, depth=3)
    key = jax.random.PRNGKey(42)
    params = model.init(key, jnp.array([0.5]))
    
    # --- PERTURBATION: Break Symmetry ---
    # Increase noise to 0.05 to prevent getting stuck in Sigma valley
    params = jax.tree_map(lambda x: x + 0.05 * jax.random.normal(key, x.shape), params)
    
    print("Starting Training (From Scratch / No Pre-training)...")
    
    total_steps = 500
    optimizer = optax.chain(
        optax.zero_nans(), 
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=0.0005)
    )
    opt_state = optimizer.init(params)
    
    reg_w = 0.0001
    
    @jax.jit
    def update_step(params, opt_state):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, model.apply, reg_w)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    X_final, Z_final, rho_final = None, None, None
    loss_history = []
    
    print(f"{'Epoch':<6} | {'Total Loss':<12} | {'L2 Error':<12} | {'Reg Loss':<12}")
    print("-" * 54)

    for i in range(total_steps + 1):
        params, opt_state, loss, (l2, reg, rho_f, X_f, Z_f) = update_step(params, opt_state)
        
        if i == total_steps:
            X_final, Z_final, rho_final = X_f, Z_f, rho_f

        if i % 10 == 0:
            print(f"{i:<6} | {loss:<12.6f} | {l2:<12.6f} | {reg:<12.6f}")
            loss_history.append(l2)

    # --- Plotting ---
    print("\nPlotting result...")
    os.makedirs("figures", exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    X_m = np.array(X_final) / 1000.0
    Z_m = np.array(Z_final) / 1000.0
    rho_f = np.array(rho_final)
    
    ax = axes[0]
    ax.contour(X_m, Z_m, rho_f, levels=np.linspace(0.1, 1.0, 10), colors='k', linewidths=1)
    h_vals = h_schaer(X_final[:,0]) / 1000.0
    ax.fill_between(X_m[:,0], h_vals, 0, color='gray')
    ax.set_title(f"Learned Coordinate Solution (L2: {loss_history[-1]:.4f})")
    ax.set_ylim(0, 15)
    
    ax = axes[1]
    Z_lines = Z_m[:, ::5]
    X_lines = X_m[:, ::5]
    for k in range(Z_lines.shape[1]):
        ax.plot(X_lines[:, k], Z_lines[:, k], 'k-', linewidth=0.5)
    ax.fill_between(X_m[:,0], h_vals, 0, color='gray')
    ax.set_title(f"Learned Coordinate Layers (Epoch {total_steps})")
    ax.set_ylim(0, 15)
    
    plt.savefig("figures/neural_schaer_result.png")
    print("Saved to figures/neural_schaer_result.png")

    print("Saving trained parameters...")
    import pickle
    with open("neural_schaer_params.pkl", "wb") as f:
        pickle.dump(params, f)
    print("Done.")

if __name__ == "__main__":
    train()