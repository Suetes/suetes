import os
# Suppress JAX/XLA C++ warnings (like the cuda_executor driver version warning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import json
import subprocess
import time

def run_single_worker(core_mode, T_val, grid_size):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from suetes.regional3d.geometry import RegionalGrid3D
    from suetes.regional3d.operators import CGridOperator3D
    from suetes.regional3d.euler import Euler3D
    from suetes.regional3d.steppers import build_dynamical_core

    nx, ny, nz = grid_size
    dx = 100.0
    grid = RegionalGrid3D(nx, ny, nz, dx, dx, dx, lat_center=0.0, lon_center=0.0)
    constants = {'g': 9.81, 'cp': 1004.0, 'Rd': 287.0, 'cvd': 717.0, 'p0': 100000.0}

    tmp_phys = Euler3D(grid, CGridOperator3D(grid), constants, dt=1.0, N_bv=0.0)
    bg_ref = {
        'rho': tmp_phys.c['p0'] / (tmp_phys.c['Rd'] * tmp_phys.theta_bg) * \
               (tmp_phys.pi_bg ** (tmp_phys.c['cvd'] / tmp_phys.c['Rd'])),
        'pi': tmp_phys.pi_bg, 'th_v': tmp_phys.theta_bg
    }

    state = {
        'u': jnp.zeros((grid.nx+1, grid.ny, grid.nz)), 'v': jnp.zeros((grid.nx, grid.ny+1, grid.nz)), 
        'w': jnp.zeros((grid.nx, grid.ny, grid.nz+1)),
        'pi': bg_ref['pi'], 'eta_dot': jnp.zeros((grid.nx, grid.ny, grid.nz+1)), 'rho': bg_ref['rho'],
    }
    
    # Initialize a warm bubble in the center of the 3D domain
    X, Y, Z = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_m, indexing='ij')
    x_c, y_c, z_c = (nx * dx) / 2.0, (ny * dx) / 2.0, 1500.0
    r = jnp.sqrt((X - x_c)**2 + (Y - y_c)**2 + (Z - z_c)**2)
    bubble = jnp.where(r <= 1000.0, 3.0 * jnp.cos(0.5 * jnp.pi * r / 1000.0)**2, 0.0)
    
    state['th_v'] = bg_ref['th_v'] + bubble
    state['rho'] = constants['p0'] / (constants['Rd'] * state['th_v']) * \
                   (bg_ref['pi'] ** (constants['cvd'] / constants['Rd']))
    target_th = state['th_v'] + 1.0

    core_type = "sisl" if "sisl" in core_mode else "split-explicit"
    use_checkpoint = True 

    dt = 1.0 if core_type == "split-explicit" else 10.0
    num_steps = max(1, int(T_val / dt))

    op = CGridOperator3D(grid)
    if core_type == "sisl":
        # Toggle iterations based on the specific SISL mode
        iters = 10 if core_mode == "sisl-fast" else 30
        core_kwargs = {
            "dt": dt, "nu_div_factor": 0.0, "nu_h_factor": 0.0, 
            "damp_height": 7500.0, "max_damp": 0.05, "N_bv": 0.0,
            "solver_tol": 1e-4, "solver_maxiter": iters, "solver_restart": iters, "alpha": 0.55
        }
    else:
        ns = max(12, int(dt / 0.1))
        core_kwargs = {
            "dt": dt, "ns": ns, "nu_div_factor": 0.0, "nu_h_factor": 0.0, 
            "damp_height": 7500.0, "max_damp": 0.05, "N_bv": 0.0, "alpha": 0.55
        }

    def forward_sim(initial_state):
        stepper, actual_dt = build_dynamical_core(
            core_type=core_type, grid=grid, operators=op, constants=constants,
            initial_state=initial_state, **core_kwargs
        )
        def scan_step(st, step_idx):
            next_st = stepper.step(st, step_idx * actual_dt, None, lambda x, f: x)
            return next_st, None
        
        if use_checkpoint and core_type == "split-explicit":
            scan_step = jax.checkpoint(scan_step)

        final_state, _ = jax.lax.scan(scan_step, initial_state, jnp.arange(num_steps))
        diff = final_state['th_v'] - target_th
        return 0.5 * jnp.sum(diff**2) * grid.dx * grid.dy * grid.dz

    fn = jax.jit(jax.value_and_grad(forward_sim))
    
    # Warmup
    for _ in range(2):
        val, grads = fn(state)
        val.block_until_ready()
        grads['th_v'].block_until_ready()

    dev = jax.local_devices()[0]
    base_mem = dev.memory_stats().get('bytes_in_use', 0) if hasattr(dev, 'memory_stats') and dev.memory_stats() else 0

    # Timed runs
    runtimes = []
    for _ in range(3):
        t0 = time.perf_counter()
        val, grads = fn(state)
        val.block_until_ready()
        grads['th_v'].block_until_ready()
        runtimes.append((time.perf_counter() - t0) * 1000.0)

    t_runtime_ms = min(runtimes)
    peak_mem = dev.memory_stats().get('peak_bytes_in_use', 0) if hasattr(dev, 'memory_stats') and dev.memory_stats() else 0
    mem_used_mb = max(0.0, (peak_mem - base_mem) / (1024.0 * 1024.0))

    print(json.dumps({
        "core_mode": core_mode, "T": T_val, "grid_cells": nx*ny*nz,
        "runtime_ms": t_runtime_ms, "vram_mb": mem_used_mb, "oom": False
    }))

if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        core_mode = sys.argv[2]
        T_val, nx, ny, nz = map(float, sys.argv[3].split(","))
        run_single_worker(core_mode, T_val, (int(nx), int(ny), int(nz)))
        sys.exit(0)

    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs('output/plots', exist_ok=True)
    
    print("\n==========================================================================")
    print("EXPERIMENT: 3D DOMAIN SIZE SCALING ON A SINGLE GPU (T = 30s)")
    print("==========================================================================")
    
    grids = [(64, 64, 32), (96, 96, 32), (128, 128, 32), (192, 192, 32)]
    grid_cells = [g[0]*g[1]*g[2] for g in grids]
    modes = ["split-explicit-ckpt", "sisl-fast", "sisl"]
    
    results_dom = {m: {"runtimes": [], "vrams": [], "oom": []} for m in modes}

    for mode in modes:
        print(f"\n--- Benchmarking Domain Scaling: {mode.upper()} ---")
        for i, g in enumerate(grids):
            cmd = [sys.executable, "-u", __file__, "--worker", mode, f"30.0,{g[0]},{g[1]},{g[2]}"]
            try:
                out = subprocess.check_output(cmd, env=dict(os.environ, XLA_PYTHON_CLIENT_MEM_FRACTION="0.85"))
                lines = out.decode('utf-8').strip().split("\n")
                data = json.loads(lines[-1])
                results_dom[mode]["runtimes"].append(data["runtime_ms"])
                results_dom[mode]["vrams"].append(data["vram_mb"])
                results_dom[mode]["oom"].append(False)
                print(f"  Grid: {g[0]}x{g[1]}x{g[2]} ({data['grid_cells']:7d} cells) | Runtime: {data['runtime_ms']:7.1f} ms | VRAM: {data['vram_mb']:6.1f} MB")
            except subprocess.CalledProcessError:
                results_dom[mode]["runtimes"].append(np.nan)
                results_dom[mode]["vrams"].append(np.nan)
                results_dom[mode]["oom"].append(True)
                print(f"  Grid: {g[0]}x{g[1]}x{g[2]} | [OOM: VRAM EXHAUSTED]")

    fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Time Plot
    ax_time.plot(grid_cells, results_dom["split-explicit-ckpt"]["runtimes"], 'o-', color='#e377c2', label=r'Split-explicit', linewidth=2.5, markersize=8)
    ax_time.plot(grid_cells, results_dom["sisl-fast"]["runtimes"], '^--', color='#2ca02c', label=r'SISL ($\Delta t=10.0$s, 10 GMRES Iters)', linewidth=2.5, markersize=8)
    ax_time.plot(grid_cells, results_dom["sisl"]["runtimes"], 's-', color='#1f77b4', label=r'SISL ($\Delta t=10.0$s, 30 GMRES Iters)', linewidth=2.5, markersize=8)
    ax_time.set_title('(a) Reverse-mode adjoint execution time', fontsize=14)
    ax_time.set_xlabel('Total number of 3D grid cells', fontsize=12)
    ax_time.set_ylabel('Adjoint pass wall-clock time (ms)', fontsize=12)
    ax_time.grid(True, which="both", ls="--", alpha=0.5)
    ax_time.legend(fontsize=11)

    # Mem Plot
    ax_mem.plot(grid_cells, results_dom["split-explicit-ckpt"]["vrams"], 'o-', color='#e377c2', label=r'Split-explicit', linewidth=2.5, markersize=8)
    ax_mem.plot(grid_cells, results_dom["sisl-fast"]["vrams"], '^--', color='#2ca02c', label=r'SISL ($\Delta t=10.0$s, 10 GMRES Iters)', linewidth=2.5, markersize=8)
    ax_mem.plot(grid_cells, results_dom["sisl"]["vrams"], 's-', color='#1f77b4', label=r'SISL ($\Delta t=10.0$s, 30 GMRES Iters)', linewidth=2.5, markersize=8)
    ax_mem.set_title('(b) Peak reverse-mode VRAM overhead', fontsize=14)
    ax_mem.set_xlabel('Total number of 3D grid cells', fontsize=12)
    ax_mem.set_ylabel('Peak VRAM (MB)', fontsize=12)
    ax_mem.grid(True, which="both", ls="--", alpha=0.5)
    ax_mem.legend(fontsize=11)

    fig.tight_layout()
    fig.savefig('output/plots/single_gpu_domain_scaling.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("\n[SUCCESS] Benchmark complete. Plot saved to output/plots/single_gpu_domain_scaling.png\n")