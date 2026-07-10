#!/usr/bin/env python3
"""
Performance scaling benchmark for Suêtes: 3D Rising Thermal Bubble.
Measures throughput (SYPD), latency, and VRAM scaling on a single GPU.
"""
import os
import time
import csv
import functools
import gc
import jax
# Toggle to True if strict double-precision validation is required
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

def run_bubble_benchmark(core_type, N, dt_multiplier=1.0, num_warmup=5, num_steps=30):
    """Run a performance benchmark for a fixed 10km^3 domain at NxNxN resolution."""
    
    # Lock physical domain size to 10 km
    domain_size = 10000.0
    dx = dy = dz = domain_size / N
    
    # Base stable timestep scaled by resolution (Reference: 125m -> 2.5s)
    ref_dx = 125.0
    ref_dt = 2.5
    base_dt = ref_dt * (dx / ref_dx)
    
    # Apply the timestep multiplier (SISL bypasses advective CFL limits)
    dt = base_dt * dt_multiplier
    
    if core_type == "sisl":
        # Neutral stability (N_bv=0) for pure buoyant bubble
        core_kwargs = {
            "dt": dt, "nu_div_factor": 0.05, "nu_h_factor": 0.05, 
            "damp_height": 7500.0, "max_damp": 0.05, "N_bv": 0.0,
            "solver_tol": 1e-4, "solver_maxiter": 20, "solver_restart": 20, "alpha": 0.55
        }
    elif core_type == "split-explicit":
        # Keep acoustic sub-stepping proportionally stable
        ns = max(6, int(dt / 0.2)) 
        core_kwargs = {
            "dt": dt, "ns": ns, "nu_div_factor": 0.0, "nu_h_factor": 0.0, 
            "damp_height": 7500.0, "max_damp": 0.05, "N_bv": 0.0, "alpha": 0.55
        }
    else:
        raise ValueError("Unknown core type")

    grid = RegionalGrid3D(N, N, N, dx, dy, dz, lat_center=0.0, lon_center=0.0)
    op = CGridOperator3D(grid)
    constants = {'g': 9.81, 'cp': 1004.0, 'Rd': 287.0, 'cvd': 717.0, 'p0': 100000.0}

    # Initialize neutral background
    tmp_phys = Euler3D(grid, op, constants, dt=dt, N_bv=0.0) 
    bg_ref = {
        'rho': tmp_phys.c['p0'] / (tmp_phys.c['Rd'] * tmp_phys.theta_bg) * \
               (tmp_phys.pi_bg ** (tmp_phys.c['cvd'] / tmp_phys.c['Rd'])),
        'pi': tmp_phys.pi_bg, 'th_v': tmp_phys.theta_bg
    }

    state = {
        'u': jnp.zeros((N+1, N, N)), 'v': jnp.zeros((N, N+1, N)), 'w': jnp.zeros((N, N, N+1)),
        'pi': bg_ref['pi'], 'eta_dot': jnp.zeros((N, N, N+1)), 'rho': bg_ref['rho'],
    }

    # Inject the +2K Cosine-squared Warm Bubble
    X, Y, Z = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_m, indexing='ij')
    # Using a 3D spherical bubble centered in the domain
    x_c, y_c, z_c = domain_size / 2.0, domain_size / 2.0, 2000.0
    r = jnp.sqrt((X - x_c)**2 + (Y - y_c)**2 + (Z - z_c)**2)
    bubble = jnp.where(r <= 1500.0, 2.0 * jnp.cos(0.5 * jnp.pi * r / 1500.0)**2, 0.0)

    state['th_v'] = bg_ref['th_v'] + bubble
    state['rho'] = constants['p0'] / (constants['Rd'] * state['th_v']) * \
                   (bg_ref['pi'] ** (constants['cvd'] / constants['Rd']))

    stepper, actual_dt = build_dynamical_core(
        core_type=core_type, grid=grid, operators=op, constants=constants,
        initial_state=state, **core_kwargs
    )

    # Wrap the stepper in a scan for fast execution on device
    @functools.partial(jax.jit, static_argnums=(1,))
    def run_chunk(st, steps):
        def body(s, _):
            return stepper.step(s, 0.0, forcing=None, bc_fn=lambda x, f: x), None
        final_state, _ = jax.lax.scan(body, st, None, length=steps)
        return final_state

    print(f"\n--- Benchmarking {core_type.upper()} | Grid: {N}^3 (dx={dx:.1f}m) | dt: {actual_dt:.2f}s ---")
    
    # Warm-up twice to trigger both initial JIT compilation and device-array layout specialization
    print("  -> JIT Compiling and warming up...")
    t_comp_start = time.time()
    state = run_chunk(state, num_steps)
    jax.block_until_ready(state['w'])
    state = run_chunk(state, num_steps)
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
        "core": core_type, "dt_multiplier": dt_multiplier, "N": N, "dx": dx, 
        "grid_points": N**3, "dt": actual_dt, "comp_time_s": comp_time, 
        "wall_time_s": wall_time, "ms_per_step": time_per_step * 1000, 
        "sypd": sypd, "vram_mb": mem_mb
    }

def main():
    gpu_name = jax.devices()[0].device_kind.replace(" ", "_")
    out_csv = f"output/benchmark_bubble_scaling_{gpu_name}.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    
    # Practical sweeping grid resolutions (up to 160^3 = 4.1 million grid points)
    grid_sizes = [64, 96, 128, 160]
    num_steps = 15
    
    results = []
    for N in grid_sizes:
        # Split-Explicit gets standard 1x timestep
        try:
            res_se = run_bubble_benchmark("split-explicit", N=N, dt_multiplier=1.0, num_steps=num_steps)
            results.append(res_se)
        except Exception as e:
            print(f"  -> SE FAILED: {e}")
        finally:
            jax.clear_caches()
            gc.collect()
            
        # SISL gets a 1x timestep to compare under same physical constraints
        try:
            res_sisl_1x = run_bubble_benchmark("sisl", N=N, dt_multiplier=1.0, num_steps=num_steps)
            results.append(res_sisl_1x)
        except Exception as e:
            print(f"  -> SISL 1x FAILED: {e}")
        finally:
            jax.clear_caches()
            gc.collect()

        # SISL gets a 6x timestep to bypass advective CFL constraints
        try:
            res_sisl_6x = run_bubble_benchmark("sisl", N=N, dt_multiplier=6.0, num_steps=num_steps)
            results.append(res_sisl_6x)
        except Exception as e:
            print(f"  -> SISL 6x FAILED: {e}")
        finally:
            jax.clear_caches()
            gc.collect()
                
        # Iterative saving to preserve data up to the OOM crash
        if results:
            keys = results[0].keys()
            with open(out_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)
            print(f"  -> Intermediate results saved to {out_csv}")
    
    if not results:
        print("\n[ERROR] All benchmark runs failed.")
        return
    
    print(f"\n[DONE] Final results saved to {out_csv}")

if __name__ == "__main__":
    main()