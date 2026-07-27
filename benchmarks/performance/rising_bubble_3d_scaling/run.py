#!/usr/bin/env python3
"""Single-GPU scaling benchmark for the 3-D rising thermal bubble.

Each configuration runs in a fresh subprocess so compilation caches and GPU
allocator peak counters cannot leak between measurements. Reported SYPD is
dynamical-core kernel throughput; it excludes initialization, I/O, ERA5
coupling, diagnostics, and rendering.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from suetes.shared.experiment import add_experiment_args, setup_experiment_directories

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output"


def _memory_stats(device) -> dict[str, int]:
    try:
        return device.memory_stats() or {}
    except Exception:
        return {}


def _initialize_sisl_history(state, jnp):
    state = dict(state)
    state["u_prev"] = state["u"]
    state["v_prev"] = state["v"]
    state["w_prev"] = state["w"]
    state["eta_dot_prev"] = state["eta_dot"]
    state["tend_th_v_prev"] = jnp.zeros_like(state["th_v"])
    # Benchmark the steady integration path rather than the one-time cold-start
    # branch. Equal current/previous fields make the extrapolated trajectory
    # identical to the initial field while retaining a fixed carry structure.
    state["is_first_step"] = jnp.asarray(0.0, dtype=state["th_v"].dtype)
    return state


def run_worker(
    core_type: str, n: int, dt_multiplier: float, num_steps: int, warmup_repeats: int, timing_repeats: int
) -> dict:
    # These must be set before importing JAX in the worker process.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    os.environ.setdefault("JAX_ENABLE_X64", "false")

    import jax
    import jax.numpy as jnp
    import numpy as np

    from suetes.regional3d.euler import Euler3D
    from suetes.regional3d.geometry import RegionalGrid3D
    from suetes.regional3d.operators import CGridOperator3D
    from suetes.regional3d.steppers import build_dynamical_core

    jax.config.update("jax_enable_x64", False)

    domain_size = 10_000.0
    dx = domain_size / n
    base_dt = 2.5 * (dx / 125.0)
    requested_dt = base_dt * dt_multiplier

    if core_type == "sisl":
        ns = 0
        core_kwargs = {
            "dt": requested_dt,
            "nu_div_factor": 0.05,
            "nu_h_factor": 0.05,
            "damp_height": 7500.0,
            "max_damp": 0.05,
            "N_bv": 0.0,
            "solver_tol": 1.0e-4,
            "solver_maxiter": 20,
            "solver_restart": 20,
            "alpha": 0.55,
        }
    elif core_type == "split-explicit":
        # The outer timestep scales with dx, so a fixed number of acoustic
        # substeps also makes dtau scale with dx and preserves one acoustic-CFL
        # family across every resolution. ns=16 is the stable N=128 setting
        # verified by the finite-state diagnostic.
        ns = 16
        core_kwargs = {
            "dt": requested_dt,
            "ns": ns,
            "nu_div_factor": 0.0,
            "nu_h_factor": 0.0,
            "damp_height": 7500.0,
            "max_damp": 0.05,
            "N_bv": 0.0,
            "alpha": 0.55,
        }
    else:
        raise ValueError(f"Unknown core type: {core_type}")

    device = jax.local_devices()[0]
    baseline_stats = _memory_stats(device)
    baseline_bytes = baseline_stats.get("bytes_in_use", 0)

    grid = RegionalGrid3D(n, n, n, dx, dx, dx, lat_center=0.0, lon_center=0.0)
    op = CGridOperator3D(grid)
    constants = {"g": 9.81, "cp": 1004.0, "Rd": 287.0, "cvd": 717.0, "p0": 100000.0}
    tmp_phys = Euler3D(grid, op, constants, dt=requested_dt, N_bv=0.0)
    bg_ref = {
        "rho": constants["p0"]
        / (constants["Rd"] * tmp_phys.theta_bg)
        * tmp_phys.pi_bg ** (constants["cvd"] / constants["Rd"]),
        "pi": tmp_phys.pi_bg,
        "th_v": tmp_phys.theta_bg,
    }
    state = {
        "u": jnp.zeros((n + 1, n, n)),
        "v": jnp.zeros((n, n + 1, n)),
        "w": jnp.zeros((n, n, n + 1)),
        "pi": bg_ref["pi"],
        "eta_dot": jnp.zeros((n, n, n + 1)),
        "rho": bg_ref["rho"],
    }

    # RegionalGrid3D is centred on zero in x and y.
    x, y, z = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_m, indexing="ij")
    radius = jnp.sqrt(x**2 + y**2 + (z - 2000.0) ** 2)
    bubble = jnp.where(radius <= 1500.0, 2.0 * jnp.cos(0.5 * jnp.pi * radius / 1500.0) ** 2, 0.0)
    state["th_v"] = bg_ref["th_v"] + bubble
    state["rho"] = (
        constants["p0"] / (constants["Rd"] * state["th_v"]) * (bg_ref["pi"] ** (constants["cvd"] / constants["Rd"]))
    )
    if core_type == "sisl":
        state = _initialize_sisl_history(state, jnp)

    stepper, actual_dt = build_dynamical_core(
        core_type=core_type, grid=grid, operators=op, constants=constants, initial_state=state, **core_kwargs
    )

    def run_chunk_impl(chunk_state, steps):
        def body(scan_state, step_index):
            next_state = stepper.step(
                scan_state, step_index * actual_dt, forcing=None, bc_fn=lambda value, forcing: value
            )
            return next_state, None

        return jax.lax.scan(body, chunk_state, jnp.arange(steps))[0]

    run_chunk = jax.jit(run_chunk_impl, static_argnames=("steps",))

    compile_start = time_perf_counter()
    compiled_state = run_chunk(state, num_steps)
    jax.block_until_ready(compiled_state["w"])
    compile_and_first_run_s = time_perf_counter() - compile_start
    del compiled_state

    for _ in range(warmup_repeats):
        warmup_state = run_chunk(state, num_steps)
        jax.block_until_ready(warmup_state["w"])
        del warmup_state

    # Reuse the same initial bubble for every timing repetition and every core.
    # Warm-up results are deliberately discarded: compilation should not alter
    # the physical workload that is subsequently measured.
    timing_state = state
    samples = []
    for _ in range(timing_repeats):
        start = time_perf_counter()
        final_state = run_chunk(timing_state, num_steps)
        jax.block_until_ready(final_state["w"])
        samples.append(time_perf_counter() - start)

    wall_median = statistics.median(samples)
    wall_min = min(samples)
    wall_max = max(samples)
    q25, q75 = np.percentile(np.asarray(samples), [25.0, 75.0])
    simulated_seconds = num_steps * float(actual_dt)
    kernel_sypd = simulated_seconds / wall_median / 365.25

    max_w = float(jnp.max(jnp.abs(final_state["w"])))
    leaves = jax.tree_util.tree_leaves(final_state)
    finite_state = bool(all(bool(jnp.all(jnp.isfinite(v))) for v in leaves))
    if not finite_state:
        raise FloatingPointError("Non-finite state produced by timed benchmark")

    final_stats = _memory_stats(device)
    current_bytes = final_stats.get("bytes_in_use", 0)
    peak_bytes = final_stats.get("peak_bytes_in_use", 0)
    mib = 1024.0**2

    return {
        "device": device.device_kind,
        "precision": "float32",
        "core": core_type,
        "dt_multiplier": dt_multiplier,
        "N": n,
        "dx": dx,
        "grid_points": n**3,
        "dt": float(actual_dt),
        "ns": ns,
        "dt_acoustic": float(actual_dt / ns) if ns else 0.0,
        "num_steps": num_steps,
        "timing_repeats": timing_repeats,
        "compile_and_first_run_s": compile_and_first_run_s,
        "wall_time_median_s": wall_median,
        "wall_time_min_s": wall_min,
        "wall_time_max_s": wall_max,
        "wall_time_iqr_s": float(q75 - q25),
        "ms_per_step": wall_median / num_steps * 1000.0,
        "kernel_sypd": kernel_sypd,
        # Backward-compatible aliases with explicit semantics in new columns.
        "sypd": kernel_sypd,
        "vram_mb": max(0.0, peak_bytes - baseline_bytes) / mib if peak_bytes else float("nan"),
        "vram_baseline_mib": baseline_bytes / mib,
        "vram_current_mib": current_bytes / mib if current_bytes else float("nan"),
        "vram_peak_mib": peak_bytes / mib if peak_bytes else float("nan"),
        "vram_peak_increment_mib": (max(0.0, peak_bytes - baseline_bytes) / mib if peak_bytes else float("nan")),
        "max_abs_w": max_w,
        "finite_state": finite_state,
    }


def time_perf_counter():
    # Kept outside the JAX-heavy worker body to make timing calls visually clear.
    import time

    return time.perf_counter()


def run_configuration(args, core, n, multiplier) -> dict:
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--core",
        core,
        "--grid-size",
        str(n),
        "--dt-multiplier",
        str(multiplier),
        "--num-steps",
        str(args.num_steps),
        "--warmup-repeats",
        str(args.warmup_repeats),
        "--timing-repeats",
        str(args.timing_repeats),
    ]
    env = dict(os.environ)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    output = subprocess.check_output(command, cwd=REPO_ROOT, env=env, text=True, stderr=subprocess.STDOUT)
    return json.loads(output.strip().splitlines()[-1])


def write_results(path: Path, results: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--core", choices=("sisl", "split-explicit"))
    parser.add_argument("--grid-size", type=int)
    parser.add_argument("--dt-multiplier", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=5, help="Steps per compiled/timed chunk (default: 5)")
    parser.add_argument(
        "--warmup-repeats", type=int, default=1, help="Additional untimed chunks after compilation (default: 1)"
    )
    parser.add_argument(
        "--timing-repeats", type=int, default=5, help="Repeated timed chunks used for the median (default: 5)"
    )
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=(64, 96, 128, 160, 192))
    add_experiment_args(parser, default_output_root=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.worker:
        if args.core is None or args.grid_size is None:
            raise SystemExit("--worker requires --core and --grid-size")
        result = run_worker(
            args.core, args.grid_size, args.dt_multiplier, args.num_steps, args.warmup_repeats, args.timing_repeats
        )
        print(json.dumps(result, allow_nan=True))
        return

    output_dir, _ = setup_experiment_directories(
        args, kind="benchmarks", case="rising_bubble_3d_scaling"
    )

    results = []
    output_path = None
    configurations = (("split-explicit", 1.0), ("sisl", 1.0), ("sisl", 10.0))
    for n in args.grid_sizes:
        for core, multiplier in configurations:
            label = f"{core}, N={n}, dt_multiplier={multiplier:g}"
            print(f"\n--- {label} ---", flush=True)
            try:
                result = run_configuration(args, core, n, multiplier)
            except subprocess.CalledProcessError as error:
                print(error.output, flush=True)
                print(f"FAILED: {label} (exit={error.returncode})", flush=True)
                continue

            results.append(result)
            if output_path is None:
                device_name = result["device"].replace(" ", "_")
                output_path = output_dir / f"benchmark_bubble_scaling_{device_name}.csv"
            write_results(output_path, results)
            print(
                f"dt={result['dt']:.5g}s, ns={result['ns']}, "
                f"median={result['ms_per_step']:.3f} ms/step, "
                f"kernel SYPD={result['kernel_sypd']:.4g}, "
                f"peak increment={result['vram_peak_increment_mib']:.1f} MiB, "
                f"max|w|={result['max_abs_w']:.3g}",
                flush=True,
            )

    if not results:
        raise SystemExit("All benchmark configurations failed")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
