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
from atmos_jax.core.transforms import GalChenSigma, Sleve, IntegralNeuralTransform
from atmos_jax.core.neural import StandardMLP
from atmos_jax.dynamics.advection import SchaerAdvection
from atmos_jax.utils.datasets import get_complex_topo, create_mixed_dataset

def run_evaluation():

    # How many samples to create
    num_samples = 9

    output_dir = "figures/SchaerGeneral/"
    os.makedirs(output_dir, exist_ok=True)

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

    print(f"Generating {num_samples} random complex topographies...")
    key = jax.random.PRNGKey(123)
    dataset = create_mixed_dataset(key, num_samples=num_samples)

    for i, data in enumerate(dataset):
        type_name = data['type']
        h0s, as_, lams, xcs = data['params']
        print(f"\nRunning {type_name} Sample {i+1}...")
        
        def h_func(x): return get_complex_topo(x, h0s, as_, lams, xcs)

        strategies = [
            ("Sigma", GalChenSigma()),
            ("SLEVE (Std)", Sleve()),
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
            
            # --- PLOT GRID LINES ---
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
        outfile = os.path.join(output_dir, f"Sample{i+1}.png")
        plt.savefig(outfile, dpi=150)
        print(f"Saved {outfile}")
        plt.close()

if __name__ == "__main__":
    run_evaluation()