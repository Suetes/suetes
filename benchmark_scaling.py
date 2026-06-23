#!/usr/bin/env python3
"""
Massive scaling benchmark for Suêtes.
Pushes grids to OOM and demonstrates the SISL large-timestep advantage.
"""
import os
import time
import csv
import jax
# Force float32 for realistic performance benchmarking on GPUs
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.steppers import build_dynamical_core

def get_gpu_memory_mb():
    """Attempt to get JAX device memory stats (depends on backend)."""
    try:
        stats = jax.local_devices()[0].memory_stats()
        return stats.get('bytes_in_use', 0) / (1024 ** 2)
    except Exception:
        return 0.0

def run_benchmark(core_type, N, dt_multiplier=1.0, num_warmup=5, num_steps=30):
    """Run a performance benchmark for a given NxNxN grid."""
    
    nx, ny, nz = N, N, N
    dx = dy = dz = 125.0
    
    # Apply the timestep multiplier (SISL can bypass CFL limits)
    dt = 2.5 * dt_multiplier
    
    if core_type == "sisl":
        core_kwargs = {"dt": dt, "nu_div_factor": 0.0, "nu_h_factor": 0.0, "alpha": 0.55}
    elif core_type == "split-explicit":
        core_kwargs = {"dt": dt, "ns": 6, "nu_div_factor": 0.0, "nu_h_factor": 0.0, "alpha": 0.55}
    else:
        raise ValueError("Unknown core type")

    grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_center=0.0, lon_center=0.0)
    op = CGridOperator3D(grid)
    constants = {'g': 9.81, 'cp': 1004.0, 'Rd': 287.0, 'cvd': 717.0, 'p0': 100000.0}

    tmp_phys = Euler3D(grid, op, constants, dt=dt, N_bv=0.01) # Stratified background
    bg_ref = {
        'rho': tmp_phys.c['p0'] / (tmp_phys.c['Rd'] * tmp_phys.theta_bg) * \
               (tmp_phys.pi_bg ** (tmp_phys.c['cvd'] / tmp_phys.c['Rd'])),
        'pi': tmp_phys.pi_bg, 'th_v': tmp_phys.theta_bg
    }

    state = {
        'u': jnp.zeros((nx+1, ny, nz)), 'v': jnp.zeros((nx, ny+1, nz)), 'w': jnp.zeros((nx, ny, nz+1)),
        'pi': bg_ref['pi'], 'eta_dot': jnp.zeros((nx, ny, nz+1)), 'rho': bg_ref['rho'],
        'th_v': bg_ref['th_v']
    }

    stepper, actual_dt = build_dynamical_core(
        core_type=core_type, grid=grid, operators=op, constants=constants,
        initial_state=state, **core_kwargs
    )

    # Wrap the stepper in a scan for fast execution on device
    @jax.jit(static_argnames=['steps'])
    def run_chunk(st, steps):
        def body(s, _):
            return stepper.step(s, 0.0, forcing=None, bc_fn=lambda x, f: x), None
        final_state, _ = jax.lax.scan(body, st, jnp.arange(steps))
        return final_state

    print(f"\n--- Benchmarking {core_type.upper()} | Grid: {N}x{N}x{N} | dt: {actual_dt}s ---")
    
    # Warm-up to trigger JIT compilation
    print("  -> JIT Compiling and warming up...")
    t_comp_start = time.time()
    state = run_chunk(state, num_warmup)
    jax.block_until_ready(state['w'])
    comp_time = time.time() - t_comp_start
    print(f"  -> Compilation + Warmup Time: {comp_time:.2f}s")
    
    mem_mb = get_gpu_memory_mb()

    # Timed execution
    print(f"  -> Executing {num_steps} steps...")
    t_start = time.perf_counter()
    
    final_state = run_chunk(state, num_steps)
    jax.block_until_ready(final_state['w'])  # Force sync with GPU
    
    t_end = time.perf_counter()
    wall_time = t_end - t_start
    
    # Compute metrics
    time_per_step = wall_time / num_steps
    sim_time_total = num_steps * actual_dt
    
    # SYPD: (Simulated time per wall-second) mapped to years
    sim_seconds_per_wall_second = sim_time_total / wall_time
    sypd = (sim_seconds_per_wall_second * 86400) / (365.25 * 86400)

    print(f"  -> Wall Time: {wall_time:.4f}s ({time_per_step*1000:.2f} ms/step)")
    print(f"  -> SYPD: {sypd:.4f}")
    if mem_mb > 0:
        print(f"  -> VRAM Used: {mem_mb:.1f} MB")

    return {
        "core": core_type, "dt_multiplier": dt_multiplier, "N": N, "grid_points": N**3, "dt": actual_dt,
        "comp_time_s": comp_time, "wall_time_s": wall_time, 
        "ms_per_step": time_per_step * 1000, "sypd": sypd, "vram_mb": mem_mb
    }

def main():
    import sys
    gpu_name = jax.devices()[0].device_kind.replace(" ", "_")
    out_csv = f"benchmark_results_massive_{gpu_name}.csv"
    
    if "--test" in sys.argv:
        grid_sizes = [16, 32]
        out_csv = "benchmark_results_massive_test.csv"
    else:
        # Sweeping from 128^3 up to 256^3 to find the VRAM ceiling
        grid_sizes = [128, 160, 192, 224, 256] 
    
    results = []
    for N in grid_sizes:
        # Split-Explicit gets standard 1x timestep
        try:
            res_se = run_benchmark("split-explicit", N=N, dt_multiplier=1.0, num_warmup=5, num_steps=30)
            results.append(res_se)
        except Exception as e:
            print(f"  -> SE FAILED (Likely OOM): {e}")
            
        # SISL gets a 1x timestep to compare under same physical constraints
        try:
            res_sisl_1x = run_benchmark("sisl", N=N, dt_multiplier=1.0, num_warmup=5, num_steps=30)
            results.append(res_sisl_1x)
        except Exception as e:
            print(f"  -> SISL 1x FAILED (Likely OOM): {e}")

        # SISL gets a 6x timestep to bypass advective CFL
        try:
            res_sisl_6x = run_benchmark("sisl", N=N, dt_multiplier=6.0, num_warmup=5, num_steps=30)
            results.append(res_sisl_6x)
        except Exception as e:
            print(f"  -> SISL 6x FAILED (Likely OOM): {e}")
                
        # Save intermediate results to CSV to prevent loss in case of later OOMs
        if results:
            keys = results[0].keys()
            with open(out_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)
            print(f"  -> Intermediate results saved to {out_csv}")
    
    if not results:
        print("\n[ERROR] All benchmark runs failed. Nothing to save.")
        return
    
    print(f"\n[DONE] Final results saved to {out_csv}")

if __name__ == "__main__":
    main()