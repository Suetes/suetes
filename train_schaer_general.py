import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

# Force CPU
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

from atmos_jax.core import StaggeredGrid
from atmos_jax.dynamics.advection import SchaerAdvection
from atmos_jax.core.neural import MonotonicMLP
from atmos_jax.core.transforms import BaseTransform

# ==============================================================================
# 1. Configuration
# ==============================================================================
Lx, Lz = 300000.0, 25000.0
nx, nz = 300, 50
dt, t_end = 25.0, 5000.0
u0 = 20.0
MAX_MOUNTAINS = 10

def get_complex_topo(x, h0s, as_, lams, xcs):
    x_col = jnp.expand_dims(x, axis=-1)
    x_diff = x_col - xcs
    x_dist = jnp.abs(x_diff)
    mask = x_dist <= as_
    h_star = jnp.where(mask, h0s * jnp.cos(jnp.pi * x_diff / (2 * as_))**2, 0.0)
    mountains = h_star * jnp.cos(jnp.pi * x_diff / lams)**2
    total_h = jnp.sum(mountains, axis=-1)
    max_h = jnp.max(total_h)
    scale = jnp.where(max_h > 4000.0, 4000.0 / (max_h + 1e-6), 1.0)
    return total_h * scale

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

def create_complex_dataset(num_samples=30):
    key = jax.random.PRNGKey(999)
    dataset = []
    for _ in range(num_samples):
        key, k_n, k_h, k_a, k_l, k_x = jax.random.split(key, 6)
        num_m = jax.random.randint(k_n, (), 1, 8)
        active_h = jax.random.uniform(k_h, (MAX_MOUNTAINS,), minval=500.0, maxval=3000.0)
        active_a = jax.random.uniform(k_a, (MAX_MOUNTAINS,), minval=15000.0, maxval=50000.0)
        active_l = jax.random.uniform(k_l, (MAX_MOUNTAINS,), minval=6000.0, maxval=15000.0)
        active_x = jax.random.uniform(k_x, (MAX_MOUNTAINS,), minval=50000.0, maxval=250000.0)
        mask = jnp.arange(MAX_MOUNTAINS) < num_m
        h0s = jnp.where(mask, active_h, 0.0)
        as_ = jnp.where(mask, active_a, 1.0)
        lams = jnp.where(mask, active_l, 1.0)
        xcs = jnp.where(mask, active_x, 0.0)
        dataset.append((h0s, as_, lams, xcs))
    return dataset

# ==============================================================================
# 2. Integral Transform (The Fix for Shape Error)
# ==============================================================================
class IntegralNeuralTransform(BaseTransform):
    def __init__(self, nn_apply_fn, params):
        self.apply_fn = nn_apply_fn
        self.params = params

    def __call__(self, xi, zeta, h, Lz):
        # --- FIX: Reshape to (101, 1) for batch processing ---
        y_grid = jnp.linspace(0, 1.0, 101)[:, None] 
        dy = 1.0 / 100.0
        
        # Apply NN (returns (101, 1)), then squeeze to (101,)
        density = jax.nn.softplus(self.apply_fn(self.params, y_grid).squeeze()) + 0.1
        
        # Integrate
        cdf = jnp.cumsum(density) * dy
        cdf = jnp.concatenate([jnp.array([0.0]), cdf])
        
        # Normalize
        cdf_norm = cdf / cdf[-1]
        
        # Interpolate to grid
        Y_actual = zeta / Lz
        b_shape = jnp.interp(Y_actual, jnp.linspace(0, 1, 102), cdf_norm)
        
        b_vals = 1.0 - b_shape
        return zeta + h * b_vals

# ==============================================================================
# 3. Solvers & Loss
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

def loss_fn_single(nn_params, model_apply, topo_params):
    h0s, as_, lams, xcs = topo_params
    def current_h(x): return get_complex_topo(x, h0s, as_, lams, xcs)
    
    transform = IntegralNeuralTransform(model_apply, nn_params)
    grid = TrainingGrid(nx, nz, Lx, Lz, current_h, transform=transform)
    
    # Regularization (Minimal, just for conditioning)
    J = grid.metrics['m']['z_zeta']
    is_stable = jnp.min(J) > 1e-5
    reg_loss = 1e-7 * jnp.sum(jax.nn.relu(-J))
    
    def physics_loss(_):
        rho_final = run_simulation_scan(grid)
        rho_true = get_analytic_rho(grid)
        return jnp.sqrt(jnp.mean((rho_final - rho_true)**2))

    def crash_loss(_): return 1.0

    phys_l2 = jax.lax.cond(is_stable, physics_loss, crash_loss, None)
    phys_l2 = jnp.where(jnp.isnan(phys_l2), 1.0, phys_l2)
    return phys_l2 + reg_loss, phys_l2

# ==============================================================================
# 4. Training Loop
# ==============================================================================
def train():
    print("Initializing Integral Network Training...")
    
    # Standard MLP (Monotonicity via Integral Transform)
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

    model = StandardMLP(width=64, depth=3)
    key = jax.random.PRNGKey(0)
    # Correct Input Shape for init: (1, 1)
    params = model.init(key, jnp.array([[0.5]]))
    
    dataset = create_complex_dataset(num_samples=30)
    print(f"Dataset: {len(dataset)} mixed samples.")
    
    optimizer = optax.chain(
        optax.zero_nans(),
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=2e-3)
    )
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
    
    best_l2 = 1e9
    best_params = params
    
    # 200 Epochs
    for epoch in range(1001):
        epoch_losses = []
        epoch_l2s = []
        crashes = 0
        
        for topo_params in dataset:
            params, opt_state, loss_val, l2_val = train_step(params, opt_state, topo_params)
            
            if l2_val >= 0.9:
                crashes += 1
            else:
                epoch_losses.append(loss_val)
                epoch_l2s.append(l2_val)
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else 999.0
        avg_l2 = np.mean(epoch_l2s) if epoch_l2s else 999.0
        
        if crashes == 0 and avg_l2 < best_l2:
            best_l2 = avg_l2
            best_params = params
            
        if epoch % 10 == 0:
            print(f"{epoch:<6} | {avg_loss:<12.6f} | {avg_l2:<12.6f} | {crashes:<8}")

    print(f"Best L2: {best_l2:.6e}")
    print("Saving params...")
    with open("neural_complex_params.pkl", "wb") as f:
        pickle.dump(best_params, f)
    
    # --- Verification Plot ---
    print("\nPlotting Verification...")
    h0s, as_, lams, xcs = dataset[0]
    def h_f(x): return get_complex_topo(x, h0s, as_, lams, xcs)
    transform = IntegralNeuralTransform(model.apply, best_params)
    grid = StaggeredGrid(nx, nz, Lx, Lz, h_f, transform=transform)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    shift = Lx / 2.0
    X_c = grid.X_corner - shift
    Z_c = grid.Z_corner
    for k in range(0, nz+1, 2):
        ax.plot(X_c[:, k]/1000.0, Z_c[:, k], 'k-', linewidth=0.5, alpha=0.6)
    h_vals = h_f(grid.X_corner[:,0])
    ax.fill_between(X_c[:,0]/1000.0, h_vals, 0, color='gray', alpha=0.5)
    ax.set_title("Learned Integral Coordinate (Sample 0)")
    ax.set_ylim(0, 15000)
    plt.savefig("figures/integral_check.png")
    print("Saved figures/integral_check.png")

if __name__ == "__main__":
    train()