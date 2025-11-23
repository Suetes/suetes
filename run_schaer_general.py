import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from flax import linen as nn

# Force CPU
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", False)

from atmos_jax.core import StaggeredGrid, RK4, Simulation
from atmos_jax.core.transforms import GalChenSigma, BaseTransform
from atmos_jax.dynamics.advection import SchaerAdvection

# --- 1. Define Model (Must match Training) ---
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

# --- 2. Define Transform (Must match Training) ---
class IntegralNeuralTransform(BaseTransform):
    """
    Matches the 'Integral' training logic. 
    NN predicts density > 0. We integrate density to get shape.
    """
    def __init__(self, nn_apply_fn, params):
        self.apply_fn = nn_apply_fn
        self.params = params

    def __call__(self, xi, zeta, h, Lz):
        # Integration grid (Fixed high-res)
        y_grid = jnp.linspace(0, 1.0, 101)[:, None]
        dy = 1.0 / 100.0
        
        # NN outputs "Density" (Must be positive)
        # We add 0.1 to ensure min slope > 0
        density = jax.nn.softplus(self.apply_fn(self.params, y_grid).squeeze()) + 0.1
        
        # Cumulative Sum (Integral)
        cdf = jnp.cumsum(density) * dy
        cdf = jnp.concatenate([jnp.array([0.0]), cdf])
        
        # Normalize to [0, 1]
        cdf_norm = cdf / cdf[-1]
        
        # Interpolate to actual zeta locations
        Y_actual = zeta / Lz
        b_shape = jnp.interp(Y_actual, jnp.linspace(0, 1, 102), cdf_norm)
        
        # Decay function: b(y) must go 1 -> 0
        b_vals = 1.0 - b_shape
        
        return zeta + h * b_vals

def run_evaluation():
    Lx, Lz = 300000.0, 25000.0
    nx, nz = 300, 50
    u0, dt = 20.0, 25.0
    t_end = 5000.0
    os.makedirs("figures_eval", exist_ok=True)

    # --- Load Trained Model ---
    try:
        with open("neural_complex_params.pkl", "rb") as f:
            nn_params = pickle.load(f)
        
        # FIX: Use StandardMLP (width=64)
        model = StandardMLP(width=64, depth=3)
        
        # FIX: Use Integral Transform
        neural_transform = IntegralNeuralTransform(model.apply, nn_params)
        print("Loaded trained neural model (Integral Transform).")
    except FileNotFoundError:
        print("Error: neural_complex_params.pkl not found.")
        return

    # --- Helper Functions ---
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

    class SleveExact:
        def __init__(self, s1=15000.0, s2=2500.0, n=1.35):
            self.s1, self.s2, self.n = s1, s2, n
        def __call__(self, xi, zeta, h, Lz):
            b2 = jnp.sinh((Lz - zeta)/self.s2) / jnp.sinh(Lz/self.s2)
            return zeta + h * (b2**self.n)

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

    print("Generating 5 random complex topographies...")
    key = jax.random.PRNGKey(999)
    dataset = []
    types = ["Smooth", "Standard", "Jagged", "Jagged", "Standard"]
    
    for type_name in types:
        key, k_n, k_h, k_a, k_l, k_x = jax.random.split(key, 6)
        num_m = jax.random.randint(k_n, (), 1, 7)
        
        if type_name == "Smooth":
            active_a = jax.random.uniform(k_a, (MAX_MOUNTAINS,), minval=40000.0, maxval=80000.0)
            active_l = jax.random.uniform(k_l, (MAX_MOUNTAINS,), minval=12000.0, maxval=25000.0)
        elif type_name == "Standard":
            active_a = jax.random.uniform(k_a, (MAX_MOUNTAINS,), minval=15000.0, maxval=40000.0)
            active_l = jax.random.uniform(k_l, (MAX_MOUNTAINS,), minval=8000.0, maxval=12000.0)
        else: 
            active_a = jax.random.uniform(k_a, (MAX_MOUNTAINS,), minval=8000.0, maxval=20000.0)
            active_l = jax.random.uniform(k_l, (MAX_MOUNTAINS,), minval=5000.0, maxval=9000.0)

        active_h = jax.random.uniform(k_h, (MAX_MOUNTAINS,), minval=500.0, maxval=3000.0)
        active_x = jax.random.uniform(k_x, (MAX_MOUNTAINS,), minval=50000.0, maxval=250000.0)
        
        mask = jnp.arange(MAX_MOUNTAINS) < num_m
        h0s = jnp.where(mask, active_h, 0.0)
        as_ = jnp.where(mask, active_a, 1.0)
        lams = jnp.where(mask, active_l, 1.0)
        xcs = jnp.where(mask, active_x, 0.0)
        
        dataset.append({'type': type_name, 'params': (h0s, as_, lams, xcs)})

    for i, data in enumerate(dataset):
        type_name = data['type']
        h0s, as_, lams, xcs = data['params']
        print(f"\nRunning {type_name} Sample {i+1}...")
        
        def h_func(x): return get_complex_topo(x, h0s, as_, lams, xcs)

        strategies = [
            ("Sigma", GalChenSigma()),
            ("SLEVE (Std)", SleveExact()),
            ("Neural (Learned)", neural_transform)
        ]
        
        fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
        shift = Lx / 2.0
        
        for idx, (strat_name, transform) in enumerate(strategies):
            grid = StaggeredGrid(nx, nz, Lx, Lz, h_func, transform=transform)
            model = SchaerAdvection(grid, get_u_profile)
            state = {'rho': init_rho(grid)}
            
            sim = Simulation(RK4(model, dt), lambda t: None, lambda s, a: s)
            final = sim.run(state, 0.0, t_end, dt, chunk_steps=200)
            
            u_z = get_u_profile(grid.Z_m)
            X_back = jnp.mod(grid.X_m - u_z * t_end, Lx)
            dx = jnp.abs(X_back - 100000.0)
            dx = jnp.minimum(dx, Lx - dx)
            r = jnp.sqrt( (dx/25000.0)**2 + ((grid.Z_m - 9000.0)/3000.0)**2 )
            rho_ana = jnp.where(r <= 1.0, jnp.cos(jnp.pi * r / 2.0)**2, 0.0)
            
            error = final['rho'] - rho_ana
            l2 = np.sqrt(np.mean(error**2))
            
            X_plot = (grid.X_m - shift) / 1000.0
            h_vals = h_func(grid.X_m[:,0])
            
            ax_l = axes[idx, 0]
            ax_l.contour(X_plot, grid.Z_m, final['rho'], levels=np.linspace(0.1, 1.0, 10), colors='k')
            ax_l.fill_between(X_plot[:,0], h_vals, 0, color='gray', alpha=0.5)
            
            # --- PLOT GRID LINES (DENSER) ---
            Z_corner = grid.Z_corner
            X_corner = grid.X_corner - shift
            
            stride_z = 2
            for k in range(0, nz+1, stride_z):
                ax_l.plot(X_corner[:, k]/1000.0, Z_corner[:, k], color='gray', alpha=0.5, linewidth=0.5)
            stride_x = 10
            for j in range(0, nx+1, stride_x):
                ax_l.plot(X_corner[j, :]/1000.0, Z_corner[j, :], color='gray', alpha=0.3, linewidth=0.4)

            ax_l.set_title(f"{strat_name} Solution")
            ax_l.set_ylim(0, 15000)
            
            ax_r = axes[idx, 1]
            cf = ax_r.contourf(X_plot, grid.Z_m, error, levels=np.linspace(-0.1, 0.1, 50), cmap='RdBu_r')
            ax_r.fill_between(X_plot[:,0], h_vals, 0, color='gray', alpha=0.5)
            ax_r.set_title(f"Error (L2: {l2:.2e})")
            ax_r.set_ylim(0, 15000)
            
            if idx == 2:
                fig.colorbar(cf, ax=axes[:, 1], shrink=0.6, label='Error')

        plt.suptitle(f"{type_name} Sample {i+1}", fontsize=16)
        filename = f"figures_eval/eval_vis_{i+1}.png"
        plt.savefig(filename)
        print(f"Saved {filename}")
        plt.close()

if __name__ == "__main__":
    run_evaluation()