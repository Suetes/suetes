import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from atmos_jax.core import StaggeredGrid
from atmos_jax.core.neural import StandardMLP
from atmos_jax.core.transforms import IntegralNeuralTransform

# --- CONFIGURATION ---
Lx, Lz = 50000.0, 21000.0
nx, nz = 101, 33 

def plot_grid_diagnostics():
    print("--- GRID DIAGNOSTICS ---")
    
    # 1. Setup Jagged Mountain
    hc, ac, lam = 1000.0, 2500.0, 2000.0
    h_func = lambda x: hc * jnp.exp(-((x - Lx/2.0)/ac)**2) * jnp.cos(jnp.pi * (x - Lx/2.0) / lam)**2
    
    # 2. Load Neural Model
    try:
        with open("neural_euler_cpu_ultra.pkl", "rb") as f:
            nn_params = pickle.load(f)
        model = StandardMLP(width=64, depth=3)
        transform = IntegralNeuralTransform(model.apply, nn_params)
    except FileNotFoundError:
        print("Model file not found!")
        return

    grid = StaggeredGrid(nx, nz, Lx, Lz, h_func, transform=transform)
    
    # 3. Extract Metric (Layer Thickness Jacobian)
    # shape (nx, nz)
    J = grid.metrics['m']['z_zeta'] 
    
    # 4. Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # A. Grid Lines (Physical Space)
    ax = axes[0, 0]
    X, Z = grid.X_corner/1000.0, grid.Z_corner
    for k in range(nz+1):
        ax.plot(X[:, k], Z[:, k], 'k-', lw=0.5)
    ax.set_title("Physical Grid Lines")
    ax.set_ylim(0, 15000)
    ax.fill_between(X[:,0], grid.h_func(grid.X_corner[:,0]), 0, color='gray')
    
    # B. Jacobian Heatmap (Computational Space)
    ax = axes[0, 1]
    im = ax.imshow(J.T, origin='lower', aspect='auto', cmap='viridis', vmin=0.0, vmax=1.2)
    ax.set_title("Jacobian J (Layer Thickness)\nLook for noisy stripes!")
    plt.colorbar(im, ax=ax, label='J')
    
    # C. Vertical Profile at Mountain Peak
    ax = axes[1, 0]
    peak_idx = nx // 2
    ax.plot(J[peak_idx, :], grid.Z_m[peak_idx, :]/1000.0, 'r-o')
    ax.set_title(f"Vertical Profile at Peak (x={Lx/2000:.1f}km)")
    ax.set_xlabel("Layer Thickness J")
    ax.set_ylabel("Height (km)")
    ax.grid(True)
    
    # D. Horizontal Gradient (Smoothness)
    ax = axes[1, 1]
    dJ_dx = (J[1:, :] - J[:-1, :]) / grid.dx
    im2 = ax.imshow(dJ_dx.T, origin='lower', aspect='auto', cmap='RdBu_r', vmin=-1e-4, vmax=1e-4)
    ax.set_title("Horizontal Gradient dJ/dx")
    plt.colorbar(im2, ax=ax)
    
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/grid_diagnosis.png")
    print("Saved diagnosis to figures/grid_diagnosis.png")
    plt.show()

if __name__ == "__main__":
    plot_grid_diagnostics()