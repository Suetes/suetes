import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", False)

from atmos_jax.core import StaggeredGrid, RK4, Simulation
from atmos_jax.core.transforms import GalChenSigma, HybridSigma, Sleve, IntegralNeuralTransform
from atmos_jax.core.neural import StandardMLP
from atmos_jax.dynamics.advection import SchaerAdvection

def run_schaer_single():
    Lx, Lz = 300000.0, 25000.0
    nx, nz = 300, 50
    u0, dt, t_end = 20.0, 25.0, 5000.0
    x_start_blob = 100000.0
    os.makedirs("figures", exist_ok=True)

    try:
        with open("neural_single_params.pkl", "rb") as f:
            nn_params = pickle.load(f)
        model = StandardMLP(width=64, depth=3)
        neural_transform = IntegralNeuralTransform(model.apply, nn_params)
        print("Loaded neural model.")
    except FileNotFoundError:
        print("Error: neural_single_params.pkl not found.")
        return

    def h_schaer(x):
        x_c = x - Lx/2.0
        h0, a, lam = 3000.0, 25000.0, 8000.0
        h_star = jnp.where(jnp.abs(x_c) <= a, h0 * jnp.cos(jnp.pi * x_c / (2*a))**2, 0.0)
        return h_star * jnp.cos(jnp.pi * x_c / lam)**2

    def get_u_profile(z):
        val_2 = u0 * jnp.sin(jnp.pi * (z - 4000.0) / 2000.0)**2
        u = jnp.where(z <= 4000.0, 0.0, 0.0)
        u = jnp.where((z > 4000.0) & (z < 5000.0), val_2, u)
        u = jnp.where(z >= 5000.0, u0, u)
        return u

    def init_rho(grid):
        r = jnp.sqrt( ((grid.X_m - x_start_blob)/25000.0)**2 + ((grid.Z_m - 9000.0)/3000.0)**2 )
        return jnp.where(r <= 1.0, jnp.cos(jnp.pi * r / 2.0)**2, 0.0)
    
    def get_analytic_rho(grid):
        u_z = get_u_profile(grid.Z_m)
        X_back = jnp.mod(grid.X_m - u_z * t_end, Lx)
        dx = jnp.abs(X_back - x_start_blob)
        dx = jnp.minimum(dx, Lx - dx)
        r = jnp.sqrt( (dx/25000.0)**2 + ((grid.Z_m - 9000.0)/3000.0)**2 )
        return jnp.where(r <= 1.0, jnp.cos(jnp.pi * r / 2.0)**2, 0.0)

    cases = [
        ('Sigma', GalChenSigma()),
        ('Hybrid', HybridSigma(scale_height=8000.0)),
        ('SLEVE', Sleve(scale_s=2500.0, scale_l=15000.0, n=1.35)), 
        ('Neural', neural_transform)
    ]
    
    fig, axes = plt.subplots(4, 2, figsize=(12, 16), constrained_layout=True)
    shift = Lx / 2.0

    for i, (name, transform) in enumerate(cases):
        grid = StaggeredGrid(nx, nz, Lx, Lz, h_schaer, transform=transform)
        model = SchaerAdvection(grid, get_u_profile)
        state = {'rho': init_rho(grid)}
        sim = Simulation(RK4(model, dt), lambda t: None, lambda s, a: s)
        final = sim.run(state, 0.0, t_end, dt, chunk_steps=200)
        
        rho_ana = get_analytic_rho(grid)
        error = final['rho'] - rho_ana
        l2 = np.sqrt(np.mean(error**2))
        
        X_plot = (grid.X_m - shift) / 1000.0
        # Use corner for plotting lines to match fill
        X_c_plot = (grid.X_corner - shift) / 1000.0
        h_vals = h_schaer(grid.X_corner[:,0])
        
        ax_l = axes[i, 0]
        ax_l.contour(X_plot, grid.Z_m, final['rho'], levels=np.linspace(0.1, 1.0, 10), colors='k')
        ax_l.fill_between(X_c_plot[:,0], h_vals, 0, color='gray', alpha=0.5)
        
        # Aligned Grid Lines
        for k in range(0, nz+1, 2):
            ax_l.plot(X_c_plot[:, k], grid.Z_corner[:, k], color='gray', alpha=0.5, linewidth=0.5)
            
        ax_l.set_title(f"{name} Solution")
        ax_l.set_ylim(0, 15000); ax_l.set_xlim(-75, 75)
        
        ax_r = axes[i, 1]
        cf = ax_r.contourf(X_plot, grid.Z_m, error, levels=np.linspace(-0.1, 0.1, 50), cmap='RdBu_r')
        ax_r.fill_between(X_c_plot[:,0], h_vals, 0, color='gray', alpha=0.5)
        ax_r.set_title(f"Error (L2: {l2:.2e})")
        ax_r.set_ylim(0, 15000); ax_r.set_xlim(-75, 75)
        if i == 3: fig.colorbar(cf, ax=axes[:, 1], shrink=0.6, label='Error')

    plt.savefig("figures/schaer_single_comparison.png")
    print("Saved figures/schaer_single_comparison.png")

if __name__ == "__main__":
    run_schaer_single()