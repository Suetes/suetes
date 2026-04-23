import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import jax
import jax.numpy as jnp

# Force CPU
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", False)

from atmos_jax.core import StaggeredGrid
from atmos_jax.dynamics import EulerSet1
from atmos_jax.core.neural import StandardMLP
from atmos_jax.core.transforms import IntegralNeuralTransform, GalChenSigma, Sleve

# --- CONFIGURATION ---
Lx, Lz = 50000.0, 21000.0
nx, nz = 101, 33  
dt = 0.05        
t_end = 15000.0  

FLOW_PARAMS = {'U': 10.0, 'N': 0.01, 'theta0': 280.0}

# --- HELPERS ---
def get_filtered_h1(x_grid, h_total, sigma_meters=10000.0):
    dx = x_grid[1] - x_grid[0]
    sigma_pixels = sigma_meters / dx
    return scipy.ndimage.gaussian_filter1d(h_total, sigma=sigma_pixels, mode='wrap')

def get_analytic_w_cpu(grid, h_params, flow_params):
    X, Z = grid.X_m, grid.Z_m
    hc, ac, lam = h_params
    U, N = flow_params['U'], flow_params['N']
    k_min, k_max = 2.0*jnp.pi/100000.0, 2.0*jnp.pi/250.0
    ks = jnp.linspace(k_min, k_max, 200) 
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
    term = -ks * U * H_k * decay * phase * (dk / jnp.pi)
    return jnp.sum(term, axis=-1)

def run_simulation(grid, name, nu4_val):
    constants = {
        'g': 9.81, 'cp': 1004.5, 'cv': 717.5, 'R': 287.0, 'p0': 1.0e5, 
        'nu4': nu4_val, 
        'nu': 0.0 
    }
    model_euler = EulerSet1(grid, constants)
    
    J = grid.metrics['m']['z_zeta']
    min_J = jnp.min(J)
    print(f"  Grid Check ({name}): Min J = {min_J:.4f}")
    if min_J <= 0: return None

    N, th0, g, cp = FLOW_PARAMS['N'], FLOW_PARAMS['theta0'], constants['g'], constants['cp']
    Z_m = grid.Z_m
    theta_bg = th0 * jnp.exp((N**2/g) * Z_m)
    pi_bg = 1.0 + (g**2 / (cp*th0*N**2)) * (jnp.exp(-(N**2/g) * Z_m) - 1.0)
    bg = {'theta': theta_bg, 'pi': pi_bg, 'dtheta_dz': (N**2/g)*theta_bg, 'dpi_dz': -g/(cp*theta_bg)}
    state = {
        'u': jnp.zeros((grid.nx+1, grid.nz)), 'w': jnp.zeros((grid.nx, grid.nz+1)), 
        'pi_p': jnp.zeros_like(Z_m), 'theta_p': jnp.zeros_like(Z_m), 'background': bg
    }
    
    U, RAMP = FLOW_PARAMS['U'], 2000.0 
    steps = int(t_end / dt)
    print(f"  Running {steps} steps (nu4={nu4_val:.1e})...")
    
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
        s2 = {**state, **jax.tree_map(lambda x, d: x + 0.5*dt*d, prog, k1)}
        k2 = model_euler.compute_rhs(s2, forcing)
        return {**state, **jax.tree_map(lambda x, d: x + dt*d, prog, k2)}

    for step in range(steps):
        t = step * dt
        if step % 20000 == 0: print(f"    Step {step}/{steps}")
        state = step_fn(state, t)
    return state

def evaluate_final():
    print("--- FINAL COMPARISON (FIXED SLEVE) ---")
    
    hc, ac, lam = 1000.0, 2500.0, 2000.0
    h_params = (hc, ac, lam)
    h_func = lambda x: hc * jnp.exp(-((x - Lx/2.0)/ac)**2) * jnp.cos(jnp.pi * (x - Lx/2.0) / lam)**2
    
    x_grid = np.linspace(0, Lx, nx)
    h_vals = h_func(x_grid)
    h1_vals = get_filtered_h1(x_grid, h_vals, sigma_meters=10000.0)
    
    # FIXED SLEVE PARAMETERS (Schaer Section 5b)
    s1_fixed, s2_fixed = 5000.0, 2000.0
    print(f"Using Fixed SLEVE: s1={s1_fixed:.0f}m, s2={s2_fixed:.0f}m")
    
    strategies = []
    try:
        with open("neural_euler_cpu_ultra.pkl", "rb") as f:
            nn_params = pickle.load(f)
        model_nn = StandardMLP(width=64, depth=3)
        strategies.append(("Neural", IntegralNeuralTransform(model_nn.apply, nn_params), 5.0e7))
    except FileNotFoundError: pass

    strategies.append(("Sigma", GalChenSigma(), 2.0e8))
    
    # SLEVE with fixed params and high diffusion
    strategies.append(("SLEVE", Sleve(h1_func=lambda x: jnp.interp(x, x_grid, h1_vals), s1=s1_fixed, s2=s2_fixed), 2.0e8))
    
    strategies.append(("Analytic", None, 0.0))
    
    results = {}
    
    print("Computing Analytic Solution...")
    dummy_grid = StaggeredGrid(nx, nz, Lx, Lz, h_func, transform=GalChenSigma())
    w_analytic_true = get_analytic_w_cpu(dummy_grid, h_params, FLOW_PARAMS)
    
    for name, transform, nu4_val in strategies:
        if name == "Analytic":
            results[name] = {
                'w': w_analytic_true, 'err': np.zeros_like(w_analytic_true), 'l2': 0.0,
                'X': dummy_grid.X_m, 'Z': dummy_grid.Z_m, 'X_c': dummy_grid.X_corner, 'Z_c': dummy_grid.Z_corner
            }
            continue

        print(f"\nSimulating: {name}...")
        grid = StaggeredGrid(nx, nz, Lx, Lz, h_func, transform=transform)
        final_state = run_simulation(grid, name, nu4_val)
        if final_state is None: continue
            
        w_num = 0.5 * (final_state['w'][:, 1:] + final_state['w'][:, :-1])
        w_true_local = get_analytic_w_cpu(grid, h_params, FLOW_PARAMS)
        mask = jnp.where(grid.Z_m < 12000.0, 1.0, 0.0)
        err = (w_num - w_true_local) * mask
        l2 = jnp.sqrt(jnp.mean(err**2))
        
        results[name] = {
            'w': w_num, 'err': err, 'l2': l2,
            'X': grid.X_m, 'Z': grid.Z_m, 'X_c': grid.X_corner, 'Z_c': grid.Z_corner
        }
        print(f"  {name} L2 Error: {l2:.5f}")

    cols = len(results)
    if cols == 0: return
    fig, axes = plt.subplots(3, cols, figsize=(4*cols, 10), constrained_layout=True)
    if cols == 1: axes = axes[:, None]

    V_RANGE = 0.15 
    LEVELS = 20

    for i, (name, _, _) in enumerate(strategies):
        if name not in results: continue
        res = results[name]
        X, Z = res['X']/1000.0, res['Z']
        Xc, Zc = res['X_c']/1000.0, res['Z_c']
        
        # Row 1: W
        ax = axes[0, i]
        cf = ax.contourf(X, Z, res['w'], levels=np.linspace(-V_RANGE, V_RANGE, LEVELS), cmap='RdBu_r', extend='both')
        ax.set_title(f"{name}\nL2: {res['l2']:.4f}")
        ax.set_ylim(0, 12000)
        if name != "Analytic":
            for k in range(0, nz+1, 2): ax.plot(Xc[:, k], Zc[:, k], 'k-', lw=0.4, alpha=0.3)

        # Row 2: Error
        ax = axes[1, i]
        if name == "Analytic":
            ax.text(0.5, 0.5, "Ref", ha='center', transform=ax.transAxes)
        else:
            cf2 = ax.contourf(X, Z, res['err'], levels=np.linspace(-0.1, 0.1, LEVELS), cmap='RdBu_r', extend='both')
            ax.set_title(f"Error")
        ax.set_ylim(0, 12000)
        
        # Row 3: Zoom
        ax = axes[2, i]
        if name == "Analytic":
             ax.text(0.5, 0.5, "-", ha='center', transform=ax.transAxes)
        else:
            for k in range(0, nz+1): ax.plot(Xc[:, k], Zc[:, k], 'k-', lw=0.8)
            ax.set_title("Grid Zoom")
            ax.set_ylim(0, 8000); ax.set_xlim(20, 30)

    fig.colorbar(cf, ax=axes[0, :], shrink=0.6, label='w (m/s)')
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/euler_jagged_fixed_sleve.png", dpi=150)
    print("Saved figures/euler_jagged_fixed_sleve.png")
    plt.show()

if __name__ == "__main__":
    evaluate_final()