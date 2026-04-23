import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import jax
import jax.numpy as jnp
from flax import linen as nn
import jax.tree_util # Ensure this is used for compatibility

from atmos_jax.core import StaggeredGrid
from atmos_jax.dynamics import EulerSet1
from atmos_jax.core.transforms import BaseTransform, GalChenSigma, Sleve

# --- CLASSES ---
class ResidualCorrectionMLP(nn.Module):
    width: int
    depth: int
    @nn.compact
    def __call__(self, z_norm):
        x = z_norm
        for _ in range(self.depth):
            x = nn.Dense(self.width)(x)
            x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x

class ResidualSleveTransform(BaseTransform):
    def __init__(self, apply_fn, params, h1_func):
        self.apply_fn = apply_fn
        self.params = params
        self.h1_func = h1_func
        self.s1 = 15000.0
        self.s2 = 2500.0

    def __call__(self, xi, zeta, h_total, Lz):
        h1 = self.h1_func(xi)
        h2 = h_total - h1
        b1 = jnp.sinh((Lz - zeta)/self.s1) / jnp.sinh(Lz/self.s1)
        b2 = jnp.sinh((Lz - zeta)/self.s2) / jnp.sinh(Lz/self.s2)
        z_base = zeta + h1 * b1 + h2 * b2
        
        y_grid = zeta / Lz
        correction = self.apply_fn(self.params, y_grid[..., None]).squeeze()
        return z_base + h_total * correction

# --- CONFIGURATION ---
Lx, Lz = 100000.0, 21000.0
nx, nz = 201, 65  
dt = 0.1          
t_end = 2.0  
FLOW_PARAMS = {'U': 10.0, 'N': 0.01, 'theta0': 280.0}
CONSTANTS = {'g': 9.81, 'cp': 1004.5, 'cv': 717.5, 'R': 287.0, 'p0': 1.0e5, 'nu4': 2.0e7, 'nu': 0.0}

# --- HELPERS (Omitted for space, assume full code copy) ---
def get_analytic_w_cpu(grid, h_params, flow_params):
    # ... (Full analytic solution logic) ...
    X, Z = grid.X_m, grid.Z_m
    hc, ac, lam = h_params
    U, N = flow_params['U'], flow_params['N']
    k_min, k_max = 2.0*jnp.pi/Lx, 2.0*jnp.pi/500.0 
    ks = jnp.linspace(k_min, k_max, 400) 
    dk = ks[1] - ks[0]
    k0 = 2.0 * jnp.pi / lam
    gauss_ft = lambda k: jnp.exp(-(k * ac / 2.0)**2)
    H_k = (hc * jnp.sqrt(jnp.pi) * ac / 2.0) * (gauss_ft(ks) + 0.5*gauss_ft(ks - k0) + 0.5*gauss_ft(ks + k0))
    l_sq = (N / U)**2
    m_sq = l_sq - ks**2
    m_real = jnp.sqrt(jnp.maximum(0.0, m_sq))
    mu_imag = jnp.sqrt(jnp.maximum(0.0, -m_sq))
    X_exp = jnp.expand_dims(X - Lx/2.0, -1)
    Z_exp = jnp.expand_dims(Z, -1)
    decay = jnp.exp(-mu_imag * Z_exp)
    phase = jnp.sin(ks * X_exp + m_real * Z_exp)
    return jnp.sum(-ks * U * H_k * decay * phase * (dk / jnp.pi), axis=-1)

def run_simulation(grid, name, nu4_val):
    # ... (Full simulation logic using jax.tree_util.tree_map) ...
    run_consts = CONSTANTS.copy()
    run_consts['nu4'] = nu4_val
    model_euler = EulerSet1(grid, run_consts)
    if jnp.min(grid.metrics['m']['z_zeta']) <= 0: return None
    
    N, th0, g, cp = FLOW_PARAMS['N'], FLOW_PARAMS['theta0'], run_consts['g'], run_consts['cp']
    Z_m = grid.Z_m
    theta_bg = th0 * jnp.exp((N**2/g) * Z_m)
    pi_bg = 1.0 + (g**2 / (cp*th0*N**2)) * (jnp.exp(-(N**2/g) * Z_m) - 1.0)
    bg = {'theta': theta_bg, 'pi': pi_bg, 'dtheta_dz': (N**2/g)*theta_bg, 'dpi_dz': -g/(cp*theta_bg)}
    state = {'u': jnp.zeros((grid.nx+1, grid.nz)), 'w': jnp.zeros((grid.nx, grid.nz+1)), 
             'pi_p': jnp.zeros_like(Z_m), 'theta_p': jnp.zeros_like(Z_m), 'background': bg}
    
    U, RAMP = FLOW_PARAMS['U'], 2000.0
    steps = int(t_end / dt)
    
    @jax.jit
    def step_fn(state, t):
        u_ref = jnp.where(t < RAMP, U * jnp.sin(0.5*jnp.pi*t/RAMP)**2, U)
        forcing = {'u_ref': u_ref, 'u_acc': 0.0}
        state['u'] = state['u'].at[0, :].set(u_ref).at[-1, :].set(state['u'][-2, :])
        slope = grid.metrics['w']['z_xi'][:, 0]
        u_at_w = 0.5*(state['u'][:, 0][:-1] + state['u'][:, 0][1:])
        state['w'] = state['w'].at[:, 0].set(u_at_w * slope).at[:, -1].set(0.0)
        prog = {k: state[k] for k in ['u', 'w', 'pi_p', 'theta_p']}
        k1 = model_euler.compute_rhs(state, forcing)
        s2 = {**state, **jax.tree_util.tree_map(lambda x, d: x + 0.5*dt*d, prog, k1)}
        k2 = model_euler.compute_rhs(s2, forcing)
        return {**state, **jax.tree_util.tree_map(lambda x, d: x + dt*d, prog, k2)}

    print(f"Simulating {name} (nu4={nu4_val:.1e})...")
    for i in range(steps):
        state = step_fn(state, i*dt)
    return state

# --- MAIN EVALUATION ---

def evaluate_final():
    print("--- UDE RESIDUAL EVALUATION ---")
    
    hc, ac, lam = 250.0, 5000.0, 4000.0
    h_params = (hc, ac, lam)
    h_func = lambda x: hc * jnp.exp(-((x - Lx/2.0)/ac)**2) * jnp.cos(jnp.pi * (x - Lx/2.0) / lam)**2
    
    x_grid = np.linspace(0, Lx, nx)
    h_vals = h_func(x_grid)
    h1_vals = 0.5 * hc * np.exp(-((x_grid - Lx/2.0)/ac)**2)
    h1_interp = lambda x: jnp.interp(x, x_grid, h1_vals)
    
    strategies = []
    
    # 1. NEURAL RESIDUAL
    try:
        with open("neural_residual_sleve.pkl", "rb") as f: params = pickle.load(f)
        model = ResidualCorrectionMLP(width=64, depth=3)
        strategies.append(("Neural Residual", ResidualSleveTransform(model.apply, params, h1_interp), 2.0e7))
    except FileNotFoundError: print("Skipping Neural Residual")
    
    # 2. SLEVE (Optimized) - Benchmark
    strategies.append(("SLEVE (Opt)", Sleve(h1_func=h1_interp, s1=5000.0, s2=2000.0), 2.0e7))
    
    # 3. SIGMA (Worst Case Baseline)
    strategies.append(("Sigma", GalChenSigma(), 2.0e7))
    
    strategies.append(("Analytic", None, 0.0))
    
    results = {}
    dummy_grid = StaggeredGrid(nx, nz, Lx, Lz, h_func, transform=GalChenSigma())
    w_analytic_true = get_analytic_w_cpu(dummy_grid, h_params, FLOW_PARAMS)
    
    for name, transform, nu4 in strategies:
        if name == "Analytic":
            # FIX: Populate all coordinate keys for the Analytic run
            results[name] = {
                'w': w_analytic_true, 'err': np.zeros_like(w_analytic_true), 'l2': 0.0,
                'X': dummy_grid.X_m, 'Z': dummy_grid.Z_m,
                'X_c': dummy_grid.X_m, 'Z_c': dummy_grid.Z_m
            }
            continue
            
        grid = StaggeredGrid(nx, nz, Lx, Lz, h_func, transform=transform)
        final_state = run_simulation(grid, name, nu4)
        if final_state is None: continue
        
        w = 0.5 * (final_state['w'][:, 1:] + final_state['w'][:, :-1])
        w_t = get_analytic_w_cpu(grid, h_params, FLOW_PARAMS)
        
        mask = jnp.where(grid.Z_m < 15000.0, 1.0, 0.0)
        err = (w - w_t) * mask
        l2 = jnp.sqrt(jnp.mean(err**2))
        
        # CORRECTED POPULATION for Simulation Runs
        results[name] = {
            'w': w, 'err': err, 'l2': l2, 
            'X': grid.X_m, 'Z': grid.Z_m,
            'X_c': grid.X_corner, 'Z_c': grid.Z_corner
        }
        print(f"  {name} L2: {l2:.5f}")

    # Plotting loop (Fixed unpacking)
    cols = len(results)
    fig, axes = plt.subplots(3, cols, figsize=(3.5*cols, 10), constrained_layout=True)
    if cols == 1: axes = axes[:, None]

    # --- PLOTTING LOOP FIX ---
    for i, (name, res) in enumerate(results.items()):
        
        X, Z = res['X']/1000.0, res['Z']
        Xc, Zc = res['X_c']/1000.0, res['Z_c'] # Accessing the fixed keys

        # ... (plotting code remains the same) ...
        V_RANGE = 0.05 
        LEVELS = 20
        
        ax = axes[0, i]
        cf = ax.contourf(X, Z, res['w'], levels=np.linspace(-V_RANGE, V_RANGE, LEVELS), cmap='RdBu_r', extend='both')
        ax.set_title(f"{name}\nL2: {res['l2']:.5f}"); ax.set_ylim(0, 15000)
        
        ax = axes[1, i]
        if name == "Analytic":
            ax.text(0.5, 0.5, "Ref", ha='center', transform=ax.transAxes)
        else:
            cf2 = ax.contourf(X, Z, res['err'], levels=np.linspace(-0.015, 0.015, LEVELS), cmap='RdBu_r', extend='both')
            ax.set_title(f"Error"); ax.set_ylim(0, 15000)
        
        ax = axes[2, i]
        if name == "Analytic":
             ax.text(0.5, 0.5, "-", ha='center', transform=ax.transAxes)
        else:
            # Grid Structure Plot
            for k in range(0, nz, 4): ax.plot(Xc[:, k], Zc[:, k], 'k-', lw=0.8, alpha=0.6)
            ax.set_title("Grid Structure"); ax.set_ylim(0, 5000)
            ax.set_xlim(40, 60)

    fig.colorbar(cf, ax=axes[0, :], shrink=0.6, label='w (m/s)')
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/euler_residual_final.png", dpi=150)
    print("Saved figures/euler_residual_final.png")
    plt.show()

if __name__ == "__main__":
    evaluate_final()