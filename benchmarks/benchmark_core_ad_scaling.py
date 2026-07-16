import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import json
import subprocess
import time

def run_single_worker(core_mode, T_val, grid_size, worker_type="total"):
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

    if worker_type == "fwd":
        fn_fwd = jax.jit(forward_sim)
        for _ in range(2):
            val = fn_fwd(state)
            val.block_until_ready()

        fwd_runtimes = []
        for _ in range(3):
            t0 = time.perf_counter()
            val = fn_fwd(state)
            val.block_until_ready()
            fwd_runtimes.append((time.perf_counter() - t0) * 1000.0)

        print(json.dumps({"runtime_fwd_ms": min(fwd_runtimes)}))
        return

    # Worker type: "total"
    fn_total = jax.jit(jax.value_and_grad(forward_sim))
    dev = jax.local_devices()[0]
    base_mem = dev.memory_stats().get('bytes_in_use', 0) if hasattr(dev, 'memory_stats') and dev.memory_stats() else 0
    
    for _ in range(2):
        val, grads = fn_total(state)
        val.block_until_ready()
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), grads)

    total_runtimes = []
    for _ in range(3):
        t0 = time.perf_counter()
        val, grads = fn_total(state)
        val.block_until_ready()
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), grads)
        total_runtimes.append((time.perf_counter() - t0) * 1000.0)

    t_total_ms = min(total_runtimes)
    peak_mem = dev.memory_stats().get('peak_bytes_in_use', 0) if hasattr(dev, 'memory_stats') and dev.memory_stats() else 0
    mem_used_mb = max(0.0, (peak_mem - base_mem) / (1024.0 * 1024.0))

    print(json.dumps({
        "runtime_total_ms": t_total_ms,
        "vram_mb": mem_used_mb
    }))


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] in ("--worker-fwd", "--worker-total"):
        worker_type = sys.argv[1].replace("--worker-", "")
        core_mode = sys.argv[2]
        T_val = float(sys.argv[3])
        nx, ny, nz = map(int, sys.argv[4].split(","))
        run_single_worker(core_mode, T_val, (nx, ny, nz), worker_type=worker_type)
        sys.exit(0)

    import matplotlib.pyplot as plt
    import numpy as np

    output_dir = "output/plots/benchmarks"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n==========================================================================")
    print("EXPERIMENT: 3D DOMAIN SIZE SCALING ON A SINGLE GPU (T = 30s)")
    print("==========================================================================")
    
    grids = [(64, 64, 32), (96, 96, 32), (128, 128, 32), (160, 160, 32)]
    grid_cells = [g[0]*g[1]*g[2] for g in grids]
    modes = ["split-explicit-ckpt", "sisl-fast", "sisl"]
    
    results_dom = {m: {"runtimes_fwd": [], "runtimes_bwd": [], "runtimes_total": [], "ratios": [], "vrams": [], "oom": []} for m in modes}

    for mode in modes:
        print(f"\n--- Benchmarking Domain Scaling: {mode.upper()} ---")
        for i, g in enumerate(grids):
            cmd_fwd = [sys.executable, "-u", __file__, "--worker-fwd", mode, "30.0", f"{g[0]},{g[1]},{g[2]}"]
            cmd_tot = [sys.executable, "-u", __file__, "--worker-total", mode, "30.0", f"{g[0]},{g[1]},{g[2]}"]
            try:
                out_fwd = subprocess.check_output(cmd_fwd, env=dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false"))
                data_fwd = json.loads(out_fwd.decode('utf-8').strip().split("\n")[-1])
                t_fwd = data_fwd["runtime_fwd_ms"]

                out_tot = subprocess.check_output(cmd_tot, env=dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false"))
                data_tot = json.loads(out_tot.decode('utf-8').strip().split("\n")[-1])
                t_tot = data_tot["runtime_total_ms"]
                vram = data_tot["vram_mb"]

                t_bwd = max(0.0, t_tot - t_fwd)
                ratio = t_bwd / t_fwd if t_fwd > 0 else np.nan

                results_dom[mode]["runtimes_fwd"].append(t_fwd)
                results_dom[mode]["runtimes_bwd"].append(t_bwd)
                results_dom[mode]["runtimes_total"].append(t_tot)
                results_dom[mode]["ratios"].append(ratio)
                results_dom[mode]["vrams"].append(vram)
                results_dom[mode]["oom"].append(False)

                print(f"  Grid: {g[0]:3d}x{g[1]:3d}x{g[2]:2d} ({g[0]*g[1]*g[2]:7d} cells) | Fwd: {t_fwd:6.1f} ms | Bwd Adjoint: {t_bwd:6.1f} ms (Ratio: {ratio:4.1f}x) | Total: {t_tot:6.1f} ms | VRAM: {vram:6.1f} MB", flush=True)
            except subprocess.CalledProcessError:
                results_dom[mode]["runtimes_fwd"].append(np.nan)
                results_dom[mode]["runtimes_bwd"].append(np.nan)
                results_dom[mode]["runtimes_total"].append(np.nan)
                results_dom[mode]["ratios"].append(np.nan)
                results_dom[mode]["vrams"].append(np.nan)
                results_dom[mode]["oom"].append(True)
                print(f"  Grid: {g[0]}x{g[1]}x{g[2]} | [OOM: VRAM EXHAUSTED]", flush=True)

    fig, (ax_time, ax_ratio, ax_mem) = plt.subplots(1, 3, figsize=(18, 5.5))
    
    # Panel (a): Execution Time Breakdown (Fwd, Adjoint Bwd, Total)
    colors = {"split-explicit-ckpt": "#e377c2", "sisl-fast": "#2ca02c", "sisl": "#1f77b4"}
    labels = {
        "split-explicit-ckpt": r"Split-explicit",
        "sisl-fast": r"SISL ($\Delta t=10.0$s, 10 GMRES Iters)",
        "sisl": r"SISL ($\Delta t=10.0$s, 30 GMRES Iters)"
    }

    for mode in modes:
        c = colors[mode]
        # Total time (solid)
        ax_time.plot(grid_cells, results_dom[mode]["runtimes_total"], 'o-', color=c, label=f"{labels[mode]} (Total)", linewidth=2.5, markersize=7)
        # Backward adjoint time (dashed)
        ax_time.plot(grid_cells, results_dom[mode]["runtimes_bwd"], 's--', color=c, label=f"{labels[mode]} (Adjoint)", linewidth=2.0, markersize=6, alpha=0.85)
        # Forward time (dotted)
        ax_time.plot(grid_cells, results_dom[mode]["runtimes_fwd"], '^:', color=c, label=f"{labels[mode]} (Forward)", linewidth=1.5, markersize=5, alpha=0.7)

    ax_time.set_title('(a) Execution time breakdown (Fwd / Adjoint / Total)', fontsize=13)
    ax_time.set_xlabel('Total number of 3D grid cells', fontsize=12)
    ax_time.set_ylabel('Execution time (ms)', fontsize=12)
    ax_time.grid(True, which="both", ls="--", alpha=0.5)
    ax_time.legend(fontsize=9, loc='upper left')

    # Panel (b): Adjoint-to-Forward Cost Ratio (T_bwd / T_fwd)
    for mode in modes:
        c = colors[mode]
        ax_ratio.plot(grid_cells, results_dom[mode]["ratios"], 'o-', color=c, label=labels[mode], linewidth=2.5, markersize=7)
    ax_ratio.set_title(r'(b) Adjoint-to-Forward cost ratio ($T_{\mathrm{bwd}} / T_{\mathrm{fwd}}$)', fontsize=13)
    ax_ratio.set_xlabel('Total number of 3D grid cells', fontsize=12)
    ax_ratio.set_ylabel(r'Cost ratio $T_{\mathrm{bwd}} / T_{\mathrm{fwd}}$', fontsize=12)
    ax_ratio.grid(True, which="both", ls="--", alpha=0.5)
    ax_ratio.legend(fontsize=10)

    # Panel (c): Peak Reverse-Mode VRAM Overhead
    for mode in modes:
        c = colors[mode]
        ax_mem.plot(grid_cells, results_dom[mode]["vrams"], 'o-', color=c, label=labels[mode], linewidth=2.5, markersize=7)
    ax_mem.set_title('(c) Peak reverse-mode VRAM overhead', fontsize=13)
    ax_mem.set_xlabel('Total number of 3D grid cells', fontsize=12)
    ax_mem.set_ylabel('Peak VRAM (MB)', fontsize=12)
    ax_mem.grid(True, which="both", ls="--", alpha=0.5)
    ax_mem.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(f'{output_dir}/single_gpu_domain_scaling.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\n[SUCCESS] Benchmark complete. Plot saved to {output_dir}/single_gpu_domain_scaling.png\n")