import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", False) 

from atmos_jax.core import StaggeredGrid, RK4, Simulation
from atmos_jax.core.transforms import GalChenSigma, HybridSigma, NeuralTransform
from atmos_jax.core.neural import MonotonicMLP
from atmos_jax.dynamics.advection import SchaerAdvection

def run_generalization_test():
    Lx, Lz = 300000.0, 25000.0
    nx, nz = 300, 50
    u0, dt = 20.0, 25.0
    t_end = 5000.0
    
    os.makedirs("figures", exist_ok=True)

    # --- Load Generalized Neural Model ---
    try:
        with open("neural_general_params.pkl", "rb") as f:
            nn_params = pickle.load(f)
        model = MonotonicMLP(width=16, depth=3)
        neural_transform = NeuralTransform(model.apply, nn_params)
        print("Loaded generalized neural model.")
    except FileNotFoundError:
        print("Error: neural_general_params.pkl not found. Run training first.")
        return

    # --- Test Suite ---
    test_cases = [
        {"name": "Standard", "h0": 3000.0, "a": 25000.0, "lam": 8000.0},
        {"name": "Steep_Hard", "h0": 4000.0, "a": 20000.0, "lam": 6000.0}, # Harder than training mean
        {"name": "Shallow",  "h0": 1000.0, "a": 30000.0, "lam": 10000.0}
    ]

    # SLEVE Split implementation for comparison
    class SleveSplit:
        def __init__(self, s1=15000.0, s2=2500.0, h0=3000.0, a=25000.0):
            self.s1, self.s2 = s1, s2
            # Note: Analytic SLEVE usually needs tuning for 'h0'. 
            # Here we use fixed decay scales s1/s2 for fairness.
            self.h0_ref, self.a_ref = h0, a 

        def __call__(self, xi, zeta, h, Lz):
            # Dynamic split based on local h properties is hard analytically without global knowledge
            # We approximation split by smoothing h (conceptually)
            # Ideally SLEVE parameters s1/s2 are tuned per mountain.
            # We keep s1=15km, s2=2.5km fixed as "Standard SLEVE".
            b1 = jnp.sinh((Lz - zeta)/self.s1) / jnp.sinh(Lz/self.s1)
            b2 = jnp.sinh((Lz - zeta)/self.s2) / jnp.sinh(Lz/self.s2)
            # Simple blend for generic comparison: decay total h by b2 (fast) 
            # or try to replicate the split.
            # Let's use the standard "Decay total h by b2^n" or similar?
            # No, let's give SLEVE the advantage of the perfect split logic used in training
            # BUT: We have to calculate h_star inside here.
            # This is the limitation of analytic: it needs to know the functional form of h.
            # The NN does NOT know h_star, it only sees z/H.
            # Wait, our NN currently only sees z/H. It doesn't see h(x).
            # So it actually learns a single universal b(z) curve.
            # Let's compare against SLEVE using the exact split for that mountain.
            
            # We can't easily reconstruct h1/h2 inside this call without passing h0/a params.
            # So we will assume this class is instantiated per test case with correct params.
            return zeta + h * b2 # Fallback to fast decay if split unknown, or...
            
            # Actually, let's implement the Split properly in the loop below
            pass

    # Physics
    def get_u_profile(z):
        z1, z2 = 4000.0, 5000.0
        val_2 = u0 * jnp.sin(jnp.pi * (z - z1) / (2 * (z2 - z1)))**2
        u = jnp.where(z <= z1, 0.0, 0.0)
        u = jnp.where((z > z1) & (z < z2), val_2, u)
        u = jnp.where(z >= z2, u0, u)
        return u

    def init_rho(grid):
        x_blob = -50000.0 + Lx/2.0
        z0, Ax, Az = 9000.0, 25000.0, 3000.0
        r = jnp.sqrt( ((grid.X_m - x_blob)/Ax)**2 + ((grid.Z_m - z0)/Az)**2 )
        return jnp.where(r <= 1.0, jnp.cos(jnp.pi * r / 2.0)**2, 0.0)

    # --- Main Test Loop ---
    for case in test_cases:
        cname = case['name']
        print(f"\nTesting on Topography: {cname} (h0={case['h0']})")
        
        # Define Topography Function
        def h_func(x):
            x_c = x - Lx/2.0
            h_star = jnp.where(jnp.abs(x_c) <= case['a'], 
                               case['h0'] * jnp.cos(jnp.pi * x_c / (2*case['a']))**2, 0.0)
            return h_star * jnp.cos(jnp.pi * x_c / case['lam'])**2

        # Define SLEVE Split specific to this mountain (Best case analytical)
        class SleveSplitExact:
            def __call__(self, xi, zeta, h, Lz):
                x_c = xi - Lx/2.0
                h_star = jnp.where(jnp.abs(x_c) <= case['a'], 
                                   case['h0'] * jnp.cos(jnp.pi * x_c / (2*case['a']))**2, 0.0)
                h1 = 0.5 * h_star
                h2 = h - h1
                b1 = jnp.sinh((Lz - zeta)/15000.0) / jnp.sinh(Lz/15000.0)
                b2 = jnp.sinh((Lz - zeta)/2500.0) / jnp.sinh(Lz/2500.0)
                return zeta + h1 * b1 + h2 * b2

        strategies = [
            ("Sigma", GalChenSigma()),
            ("SLEVE (Exact)", SleveSplitExact()),
            ("Neural (General)", neural_transform)
        ]
        
        # Run Simulations
        fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
        shift = Lx / 2.0
        
        # Analytic solution (same for all, just depends on grid Z)
        # We compute it dynamically per grid below
        
        for i, (strat_name, transform) in enumerate(strategies):
            grid = StaggeredGrid(nx, nz, Lx, Lz, h_func, transform=transform)
            model = SchaerAdvection(grid, get_u_profile)
            state = {'rho': init_rho(grid)}
            
            sim = Simulation(RK4(model, dt), lambda t: None, lambda s, a: s)
            final = sim.run(state, 0.0, t_end, dt, chunk_steps=200)
            
            # Analytic
            u_z = get_u_profile(grid.Z_m)
            X_back = jnp.mod(grid.X_m - u_z * t_end, Lx)
            dx = jnp.abs(X_back - (-50000.0 + Lx/2.0))
            dx = jnp.minimum(dx, Lx - dx)
            r = jnp.sqrt( (dx/25000.0)**2 + ((grid.Z_m - 9000.0)/3000.0)**2 )
            rho_ana = jnp.where(r <= 1.0, jnp.cos(jnp.pi * r / 2.0)**2, 0.0)
            
            error = final['rho'] - rho_ana
            l2 = np.sqrt(np.mean(error**2))
            
            # Plot
            X_plot = (grid.X_m - shift) / 1000.0
            
            ax_l = axes[i, 0]
            ax_l.contour(X_plot, grid.Z_m, final['rho'], levels=np.linspace(0.1, 1.0, 10), colors='k')
            h_vals = h_func(grid.X_m[:,0])
            ax_l.fill_between(X_plot[:,0], h_vals, 0, color='gray')
            ax_l.set_title(f"{strat_name} Solution")
            ax_l.set_ylim(0, 15000)
            
            ax_r = axes[i, 1]
            cf = ax_r.contourf(X_plot, grid.Z_m, error, levels=np.linspace(-0.1, 0.1, 50), cmap='RdBu_r')
            ax_r.fill_between(X_plot[:,0], h_vals, 0, color='gray')
            ax_r.set_title(f"Error (L2: {l2:.2e})")
            ax_r.set_ylim(0, 15000)
            
        plt.suptitle(f"Generalization Test: {cname} Topography", fontsize=16)
        plt.savefig(f"figures/general_test_{cname}.png")
        print(f"Saved figures/general_test_{cname}.png")

if __name__ == "__main__":
    run_generalization_test()