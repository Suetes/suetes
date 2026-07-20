"""
GMRES Adjoint Convergence & Time-Step Conditioning Study
========================================================
Investigates why the GMRES linear solver in the Semi-Implicit Semi-Lagrangian (SISL)
adjoint backward pass requires large iteration counts (~100 steps) when Delta t is large,
and demonstrates how reducing the time-step accelerates Krylov subspace convergence.

Physics & Conditioning Mechanism:
---------------------------------
In the SISL dynamical core, the column preconditioner M^{-1} solves the vertical 1D tridiagonal
Helmholtz equation exactly (eliminating vertical acoustic wave stiffness where CFL_v >> 1).
However, the horizontal acoustic and gravity wave coupling remains off-diagonal in M^{-1}A.
The off-diagonal coupling coefficient scales quadratically with the horizontal acoustic Courant number:
    CFL_h = (c_s * Delta t) / Delta x
    Stiffness_h ~ (alpha * CFL_h)^2

For Delta x = 100 m:
  - Delta t = 10.0 s -> CFL_h ~ 34.0 -> Stiffness ~ 350. GMRES needs ~40-100 iterations to mix horizontal modes across columns.
  - Delta t = 5.0 s  -> CFL_h ~ 17.0 -> Stiffness ~ 87.  GMRES converges ~4x faster.
  - Delta t = 2.5 s  -> CFL_h ~ 8.5  -> Stiffness ~ 22.  GMRES converges ~16x faster.
  - Delta t = 1.0 s  -> CFL_h ~ 3.4  -> Stiffness ~ 3.5. GMRES converges in ~5-10 iterations!
"""

import os
import sys
import time
import gc

# Prevent XLA from preallocating 90% of GPU memory and crashing on subsequent compiles
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.steppers import build_dynamical_core

output_dir = "output/plots/benchmarks"
os.makedirs(output_dir, exist_ok=True)


def build_experiment_case(nx=64, ny=3, nz=32, dx=100.0):
    """Sets up grid, constants, and warm bubble initial condition."""
    grid = RegionalGrid3D(nx, ny, nz, dx, dx, dx, lat_center=0.0, lon_center=0.0)
    op = CGridOperator3D(grid)
    constants = {'g': 9.81, 'cp': 1004.0, 'Rd': 287.0, 'cvd': 717.0, 'p0': 100000.0}

    tmp_phys = Euler3D(grid, op, constants, dt=1.0, N_bv=0.0)
    bg_ref = {
        'rho': tmp_phys.c['p0'] / (tmp_phys.c['Rd'] * tmp_phys.theta_bg) * \
               (tmp_phys.pi_bg ** (tmp_phys.c['cvd'] / tmp_phys.c['Rd'])),
        'pi': tmp_phys.pi_bg, 'th_v': tmp_phys.theta_bg
    }

    state = {
        'u': jnp.zeros((nx+1, ny, nz)),
        'v': jnp.zeros((nx, ny+1, nz)),
        'w': jnp.zeros((nx, ny, nz+1)),
        'pi': bg_ref['pi'],
        'eta_dot': jnp.zeros((nx, ny, nz+1)),
        'rho': bg_ref['rho'],
    }

    X, Y, Z = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_m, indexing='ij')
    x_c, z_c = 0.0, 1500.0
    r = jnp.sqrt((X - x_c)**2 + (Z - z_c)**2)
    bubble = jnp.where(r <= 1000.0, 3.0 * jnp.cos(0.5 * jnp.pi * r / 1000.0)**2, 0.0)

    state['th_v'] = bg_ref['th_v'] + bubble
    state['rho'] = constants['p0'] / (constants['Rd'] * state['th_v']) * \
                   (bg_ref['pi'] ** (constants['cvd'] / constants['Rd']))

    return grid, op, constants, state


def compute_adjoint_gradient(grid, op, constants, initial_state, T_val=30.0, dt=10.0, solver_maxiter=30, solver_tol=1e-6):
    """Computes reverse-mode adjoint gradient dL/d(th_v_0) for SISL with given GMRES parameters."""
    num_steps = max(1, int(T_val / dt))

    def forward_objective(st):
        stepper, actual_dt = build_dynamical_core(
            core_type="sisl", grid=grid, operators=op, constants=constants,
            initial_state=st, dt=dt, solver_tol=solver_tol,
            solver_maxiter=solver_maxiter, solver_restart=40,
            damp_height=7500.0, max_damp=0.05, alpha=0.55
        )

        def scan_step(curr_st, step_idx):
            next_st = stepper.step(curr_st, step_idx * actual_dt, None, lambda x, f: x)
            return next_st, None

        final_st, _ = jax.lax.scan(scan_step, st, jnp.arange(num_steps))

        w_c = 0.5 * (final_st['w'][:, :, :-1] + final_st['w'][:, :, 1:])
        th_pert = final_st['th_v'] - 300.0
        loss = 0.5 * jnp.sum(final_st['rho'] * w_c**2 + 0.1 * th_pert**2) * grid.dx * grid.dy * grid.dz
        return loss

    fn_grad = jax.jit(jax.grad(forward_objective))
    grad_st = fn_grad(initial_state)
    return grad_st['th_v']


import multiprocessing as mp

def _worker_eval(dt_val, maxiter, tol, result_queue):
    """Isolated worker function that computes adjoint gradient in a clean process."""
    grid, op, constants, state = build_experiment_case(nx=40, ny=3, nz=20, dx=150.0)
    grad = compute_adjoint_gradient(grid, op, constants, state, T_val=30.0, dt=dt_val, solver_maxiter=maxiter, solver_tol=tol)
    grad_arr = np.array(grad)
    result_queue.put(grad_arr)


def evaluate_adjoint(dt_val, maxiter, tol):
    """Spawns an isolated worker process to compute adjoint gradient without XLA/CUDA memory corruption or hangs."""
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=_worker_eval, args=(dt_val, maxiter, tol, q))
    p.start()
    grad_arr = q.get()
    p.join(timeout=0.5)
    if p.is_alive():
        p.terminate()
        p.join()
    return grad_arr


def main():
    print("==========================================================================")
    print("GMRES Adjoint Convergence Analysis & Time-Step Sensitivity Study")
    print("==========================================================================")
    print(f"Output Directory: {output_dir}\n")

    dt_configs = [
        {"dt": 10.0, "label": r"$\Delta t = 10.0$ s ($\mathrm{CFL}_h \approx 22.7$)", "color": "#E53E3E", "iters": [2, 4, 6, 8, 10, 12, 15, 20]},
        {"dt": 5.0,  "label": r"$\Delta t = 5.0$ s ($\mathrm{CFL}_h \approx 11.3$)",  "color": "#DD6B20", "iters": [2, 4, 6, 8, 10, 15]},
        {"dt": 2.5,  "label": r"$\Delta t = 2.5$ s ($\mathrm{CFL}_h \approx 5.7$)",   "color": "#3182CE", "iters": [2, 4, 6, 8, 10]},
        {"dt": 1.0,  "label": r"$\Delta t = 1.0$ s ($\mathrm{CFL}_h \approx 2.3$)",   "color": "#38A169", "iters": [1, 2, 3, 4, 6]}
    ]

    results = {}

    for cfg in dt_configs:
        dt_val = cfg["dt"]
        label = cfg["label"]
        iters_list = cfg["iters"]
        print(f"--- Evaluating Time Step: {dt_val:.1f} s ---")

        ref_maxiter = 100 if dt_val >= 5.0 else (50 if dt_val >= 2.5 else 25)
        print(f"  [0] Computing reference exact adjoint (dt={dt_val:.1f}s, maxiter={ref_maxiter}, tol=1e-12)...", end=" ", flush=True)
        t0 = time.perf_counter()
        grad_ref_arr = evaluate_adjoint(dt_val, ref_maxiter, tol=1e-12)
        ref_norm = np.linalg.norm(grad_ref_arr)
        print(f"Done in {time.perf_counter() - t0:.2f}s")

        l2_errors = []

        for it in iters_list:
            print(f"      GMRES maxiter={it:3d} ...", end=" ", flush=True)
            grad_k_arr = evaluate_adjoint(dt_val, it, tol=1e-8)

            diff = grad_k_arr - grad_ref_arr
            rel_l2 = np.linalg.norm(diff) / (ref_norm + 1e-15)

            l2_errors.append(rel_l2)
            print(f"Rel L2-error: {rel_l2:.2e}")

        results[dt_val] = {
            "label": label,
            "color": cfg["color"],
            "iters": iters_list,
            "l2_errors": l2_errors
        }
        print()

    # --- Plotting Comprehensive Convergence Curves ---
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.5))

    for dt_val, r in results.items():
        ax.semilogy(r["iters"], r["l2_errors"], 'o-', color=r["color"], label=r["label"], linewidth=2.5, markersize=6)

    # Threshold horizontal reference lines
    ax.axhline(1e-4, color='gray', linestyle=':', alpha=0.7, label=r'Target accuracy ($10^{-4}$)')
    ax.axhline(1e-6, color='black', linestyle='-.', alpha=0.7, label=r'High accuracy ($10^{-6}$)')

    ax.set_title("SISL adjoint relative $L_2$-error convergence vs. time-step", fontsize=13)
    ax.set_xlabel("GMRES iterations per time-step", fontsize=11)
    ax.set_ylabel(r"Relative $L_2$-error $\|\nabla \mathcal{L}_k - \nabla \mathcal{L}_{\text{ref}}\|_2 / \|\nabla \mathcal{L}_{\text{ref}}\|_2$", fontsize=11)
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.legend(fontsize=10, loc='upper right')

    out_path = f"{output_dir}/sisl_adjoint_gmres_dt_convergence_study.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SUCCESS] Saved multi-dt GMRES convergence analysis figure to {out_path}")

    # Summary table output
    print("\n==========================================================================")
    print("Summary of GMRES Iterations Required to Reach Target Adjoint Accuracy:")
    print("==========================================================================")
    print(f"{'Time Step (s)':<16} | {'CFL_h (approx)':<16} | {'Iters to L2 < 1e-4':<20} | {'Iters to L2 < 1e-6':<20}")
    print("-" * 78)

    for dt_val, r in results.items():
        cfl_h = 340.0 * dt_val / 100.0
        iters_arr = np.array(r["iters"])
        l2_arr = np.array(r["l2_errors"])

        idx_4 = np.where(l2_arr < 1e-4)[0]
        it_4 = f"{iters_arr[idx_4[0]]}" if len(idx_4) > 0 else f">{iters_arr[-1]}"

        idx_6 = np.where(l2_arr < 1e-6)[0]
        it_6 = f"{iters_arr[idx_6[0]]}" if len(idx_6) > 0 else f">{iters_arr[-1]}"

        print(f"{dt_val:<16.1f} | {cfl_h:<16.1f} | {it_4:<20} | {it_6:<20}")
    print("==========================================================================\n")


if __name__ == "__main__":
    main()
