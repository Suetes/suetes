import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

# Respect user setting
jax.config.update("jax_enable_x64", False) 

from atmos_jax.core import StaggeredGrid, RK4, Simulation
from atmos_jax.core.transforms import GalChenSigma, HybridSigma, NeuralTransform
from atmos_jax.core.neural import MonotonicMLP
from atmos_jax.dynamics.advection import SchaerAdvection

def run_schaer_test():
    Lx = 300000.0
    Lz = 25000.0
    nx = 300
    nz = 50
    
    u0 = 20.0
    t_end = 5000.0
    dt = 25.0
    x_start_blob = 100000.0 
    
    os.makedirs("figures", exist_ok=True)

    # Topography
    h0_val = 3000.0
    a_val = 25000.0
    lam_val = 8000.0

    def h_schaer(x):
        x_c = x - Lx/2.0
        h_star = jnp.where(jnp.abs(x_c) <= a_val, 
                           h0_val * jnp.cos(jnp.pi * x_c / (2*a_val))**2, 
                           0.0)
        return h_star * jnp.cos(jnp.pi * x_c / lam_val)**2

    # --- Custom SLEVE Split ---
    class SleveSplit:
        def __init__(self, s1=15000.0, s2=2500.0):
            self.s1, self.s2 = s1, s2
        def __call__(self, xi, zeta, h, Lz):
            x_c = xi - Lx/2.0
            h_star = jnp.where(jnp.abs(x_c) <= a_val, 
                               h0_val * jnp.cos(jnp.pi * x_c / (2*a_val))**2, 0.0)
            h1 = 0.5 * h_star
            h2 = h - h1
            b1 = jnp.sinh((Lz - zeta)/self.s1) / jnp.sinh(Lz/self.s1)
            b2 = jnp.sinh((Lz - zeta)/self.s2) / jnp.sinh(Lz/self.s2)
            return zeta + h1 * b1 + h2 * b2

    # --- Load Neural Coordinate ---
    try:
        with open("neural_schaer_params.pkl", "rb") as f:
            nn_params = pickle.load(f)
        
        # Reconstruct Model
        model = MonotonicMLP(width=16, depth=3)
        neural_transform = NeuralTransform(model.apply, nn_params)
        has_neural = True
        print("Loaded trained neural coordinate.")
    except FileNotFoundError:
        print("Neural parameters not found. Skipping Neural case.")
        has_neural = False

    # Wind Profile
    def get_u_profile(z):
        z1, z2 = 4000.0, 5000.0
        val_2 = u0 * jnp.sin(jnp.pi * (z - z1) / (2 * (z2 - z1)))**2
        u = jnp.where(z <= z1, 0.0, 0.0)
        u = jnp.where((z > z1) & (z < z2), val_2, u)
        u = jnp.where(z >= z2, u0, u)
        return u

    def init_rho(grid):
        z0 = 9000.0
        Ax, Az = 25000.0, 3000.0
        X, Z = grid.X_m, grid.Z_m
        r = jnp.sqrt( ((X - x_start_blob)/Ax)**2 + ((Z - z0)/Az)**2 )
        return jnp.where(r <= 1.0, jnp.cos(jnp.pi * r / 2.0)**2, 0.0)

    # ======================================================
    # Simulation Setup
    # ======================================================
    cases = [
        ('Sigma', GalChenSigma()),
        ('Hybrid', HybridSigma(scale_height=8000.0)),
        ('SLEVE', SleveSplit(s1=15000.0, s2=2500.0)),
    ]
    if has_neural:
        cases.append(('Neural', neural_transform))
    
    results = {}
    
    for name, transform in cases:
        print(f"Running Case: {name}")
        grid = StaggeredGrid(nx, nz, Lx, Lz, h_schaer, transform=transform)
        model = SchaerAdvection(grid, get_u_profile)
        state = {'rho': init_rho(grid)} 
        stepper = RK4(model, dt)
        sim = Simulation(stepper, lambda t: None, lambda s, a: s)
        
        snapshots = [state['rho']]
        times = [0.0, 2500.0, 5000.0]
        
        curr_state = sim.run(state, 0.0, times[1], dt, chunk_steps=100)
        snapshots.append(curr_state['rho'])
        
        curr_state = sim.run(curr_state, times[1], times[2], dt, chunk_steps=100)
        snapshots.append(curr_state['rho'])
        
        results[name] = {'snapshots': snapshots, 'X': grid.X_m, 'Z': grid.Z_m}

    # ======================================================
    # Plotting
    # ======================================================
    num_rows = len(cases)
    fig, axes = plt.subplots(num_rows, 2, figsize=(12, 4*num_rows), constrained_layout=True)
    shift = Lx / 2.0
    
    def get_analytic(X, Z, t):
        u = get_u_profile(Z)
        X_back = jnp.mod(X - u * t, Lx)
        dx = jnp.abs(X_back - x_start_blob)
        dx = jnp.minimum(dx, Lx - dx)
        z0 = 9000.0
        Ax, Az = 25000.0, 3000.0
        r = jnp.sqrt( (dx/Ax)**2 + ((Z - z0)/Az)**2 )
        return jnp.where(r <= 1.0, jnp.cos(jnp.pi * r / 2.0)**2, 0.0)

    for i, (name, _) in enumerate(cases):
        res = results[name]
        X, Z = res['X'], res['Z']
        rho_final = res['snapshots'][-1]
        
        rho_ana = get_analytic(X, Z, 5000.0)
        error = rho_final - rho_ana
        l2_err = np.sqrt(np.mean(error**2))
        
        X_plot = (X - shift) / 1000.0 
        Z_plot = Z / 1000.0 
        
        ax_l = axes[i, 0]
        levels = np.linspace(0.1, 1.0, 10)
        styles = ['--', ':', '-']
        for idx, snap in enumerate(res['snapshots']):
            ax_l.contour(X_plot, Z_plot, snap, levels=levels, colors='k', linewidths=0.8, linestyles=styles[idx])
        
        h_vals = h_schaer(X[:,0]) / 1000.0
        ax_l.plot(X_plot[:,0], h_vals, 'k-', lw=1)
        ax_l.fill_between(X_plot[:,0], h_vals, 0, color='gray', alpha=0.3)
        ax_l.set_title(f"{name} Solutions", loc='center')
        ax_l.set_ylim(0, 15); ax_l.set_xlim(-75, 75); ax_l.set_ylabel("z [km]")
        
        ax_r = axes[i, 1]
        v_max = 0.1
        levels_err = np.linspace(-v_max, v_max, 50)
        cf = ax_r.contourf(X_plot, Z_plot, error, levels=levels_err, cmap='RdBu_r', extend='both')
        fig.colorbar(cf, ax=ax_r, label='Error')
        ax_r.plot(X_plot[:,0], h_vals, 'k-', lw=1)
        ax_r.fill_between(X_plot[:,0], h_vals, 0, color='gray', alpha=0.3)
        ax_r.set_title(f"Error Field (L2: {l2_err:.2e})", loc='center')
        ax_r.set_ylim(0, 15); ax_r.set_xlim(-75, 75)
        
        if i == num_rows - 1:
            ax_l.set_xlabel("x [km]"); ax_r.set_xlabel("x [km]")

    plt.savefig("figures/recreate_schaer_fig6.png")
    print("Figure saved.")

if __name__ == "__main__":
    run_schaer_test()