#!/usr/bin/env python3
"""Compare short Suetes forward and reverse-mode rollouts on CPU and GPU.

Each device/grid configuration runs in a fresh process. Compilation and warm-up
are excluded from the reported synchronized timings. The benchmark is intended
to demonstrate accelerator speedup, not hardware-independent model throughput.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform as host_platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "output" / "benchmarks" / "cpu_gpu_short_scaling"


def _cpu_model() -> str:
    model = host_platform.processor().strip()
    try:
        with Path("/proc/cpuinfo").open(encoding="utf-8") as stream:
            for line in stream:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return model or "unknown CPU"


def _initial_state(n, nz, dx, jnp, grid, physics, constants):
    pi = physics.pi_bg
    theta = physics.theta_bg
    rho = constants["p0"] / (constants["Rd"] * theta) * pi ** (constants["cvd"] / constants["Rd"])
    x, y, z = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_m, indexing="ij")
    radius = jnp.sqrt(x**2 + y**2 + (z - 1500.0) ** 2)
    perturbation = jnp.where(radius < 1000.0, 2.0 * jnp.cos(0.5 * jnp.pi * radius / 1000.0) ** 2, 0.0)
    theta = theta + perturbation
    rho = constants["p0"] / (constants["Rd"] * theta) * pi ** (constants["cvd"] / constants["Rd"])
    return {
        "u": jnp.zeros((n + 1, n, nz)),
        "v": jnp.zeros((n, n + 1, nz)),
        "w": jnp.zeros((n, n, nz + 1)),
        "pi": pi,
        "th_v": theta,
        "eta_dot": jnp.zeros((n, n, nz + 1)),
        "rho": rho,
    }


def run_worker(args):
    os.environ.setdefault("JAX_ENABLE_X64", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp

    from suetes.regional3d.euler import Euler3D
    from suetes.regional3d.geometry import RegionalGrid3D
    from suetes.regional3d.operators import CGridOperator3D
    from suetes.regional3d.steppers import build_dynamical_core

    jax.config.update("jax_enable_x64", False)
    device = jax.devices()[0]
    n = args.grid_size
    nz = args.vertical_levels if args.vertical_levels is not None else n
    dx, dz = 10_000.0 / n, 5_000.0 / nz
    split_dt = 0.5 * dx / 125.0
    dt = split_dt if args.core == "split-explicit" else 10.0 * split_dt
    constants = {"g": 9.81, "cp": 1004.0, "Rd": 287.0, "cvd": 717.0, "p0": 100000.0}
    grid = RegionalGrid3D(n, n, nz, dx, dx, dz, lat_center=0.0, lon_center=0.0)
    operators = CGridOperator3D(grid)
    physics = Euler3D(grid, operators, constants, dt=dt, N_bv=0.0)
    state = _initial_state(n, nz, dx, jnp, grid, physics, constants)

    if args.core == "sisl":
        state["u_prev"] = state["u"]
        state["v_prev"] = state["v"]
        state["w_prev"] = state["w"]
        state["eta_dot_prev"] = state["eta_dot"]
        state["tend_th_v_prev"] = jnp.zeros_like(state["th_v"])
        # Benchmark the steady two-time-level path with a fixed carry shape.
        state["is_first_step"] = jnp.asarray(0.0, dtype=state["th_v"].dtype)

    core_kwargs = {
        "dt": dt,
        "nu_div_factor": 0.0,
        "nu_h_factor": 0.0,
        "damp_height": 4000.0,
        "max_damp": 0.05,
        "N_bv": 0.0,
        "alpha": 0.55,
    }
    if args.core == "split-explicit":
        core_kwargs["ns"] = 8
    else:
        core_kwargs.update(
            {
                "solver_tol": 1.0e-4,
                "solver_maxiter": 10,
                "solver_restart": 10,
                "use_checkpointing": True,
            }
        )

    stepper, actual_dt = build_dynamical_core(
        core_type=args.core,
        grid=grid,
        operators=operators,
        constants=constants,
        initial_state=state,
        **core_kwargs,
    )

    def objective(initial_state):
        def body(carry, step_index):
            next_state = stepper.step(carry, step_index * actual_dt, None, lambda value, forcing: value)
            return next_state, None

        final_state, _ = jax.lax.scan(body, initial_state, jnp.arange(args.num_steps))
        return jnp.mean(final_state["w"] ** 2) + 0.01 * jnp.mean((final_state["th_v"] - physics.theta_bg) ** 2)

    functions = {
        "forward": jax.jit(objective),
        "value_and_gradient": jax.jit(jax.value_and_grad(objective)),
    }
    results = []
    for workload, function in functions.items():
        compiled = function(state)
        jax.block_until_ready(compiled)
        for _ in range(args.warmup_repeats):
            jax.block_until_ready(function(state))

        samples = []
        for _ in range(args.timing_repeats):
            start = time.perf_counter()
            result = function(state)
            jax.block_until_ready(result)
            samples.append(time.perf_counter() - start)
        median_s = statistics.median(samples)
        results.append(
            {
                "platform": device.platform,
                "device": _cpu_model() if device.platform == "cpu" else device.device_kind,
                "logical_cpu_count": os.cpu_count() if device.platform == "cpu" else "",
                "precision": "float32",
                "core": args.core,
                "workload": workload,
                "nx": n,
                "ny": n,
                "nz": nz,
                "grid_cells": n * n * nz,
                "num_steps": args.num_steps,
                "dt_s": float(actual_dt),
                "timing_repeats": args.timing_repeats,
                "median_s": median_s,
                "min_s": min(samples),
                "max_s": max(samples),
                "million_cell_steps_per_s": n * n * nz * args.num_steps / median_s / 1.0e6,
            }
        )
    print(json.dumps(results))


def run_configuration(args, platform, core, grid_size):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--grid-size",
        str(grid_size),
        "--core",
        core,
        "--num-steps",
        str(args.num_steps),
        "--warmup-repeats",
        str(args.warmup_repeats),
        "--timing-repeats",
        str(args.timing_repeats),
    ]
    if args.vertical_levels is not None:
        command.extend(("--vertical-levels", str(args.vertical_levels)))
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = platform
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    output = subprocess.check_output(command, cwd=REPO_ROOT, env=env, text=True, stderr=subprocess.STDOUT)
    return json.loads(output.strip().splitlines()[-1])


def write_results(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cpu_gpu_short_scaling.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    return csv_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--core", choices=("split-explicit", "sisl"))
    parser.add_argument("--cores", nargs="+", choices=("split-explicit", "sisl"), default=("split-explicit", "sisl"))
    parser.add_argument("--grid-size", type=int)
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=(24, 32, 48, 64, 96))
    parser.add_argument(
        "--vertical-levels",
        type=int,
        default=None,
        help="Fixed vertical level count; omit for cubic N x N x N refinement",
    )
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--retries", type=int, default=1, help="Retries for a failed isolated configuration")
    parser.add_argument("--platforms", nargs="+", choices=("cpu", "cuda"), default=("cpu", "cuda"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.worker:
        if args.grid_size is None or args.core is None:
            raise SystemExit("--worker requires --grid-size and --core")
        run_worker(args)
        return

    results = []
    for platform in args.platforms:
        for core in args.cores:
            for grid_size in args.grid_sizes:
                rows = None
                for attempt in range(args.retries + 1):
                    try:
                        rows = run_configuration(args, platform, core, grid_size)
                        break
                    except subprocess.CalledProcessError as error:
                        print(
                            f"Failed {platform} {core} N={grid_size} "
                            f"(attempt {attempt + 1}/{args.retries + 1}):\n{error.output}",
                            file=sys.stderr,
                        )
                if rows is None:
                    print(f"Skipping {platform} {core} N={grid_size} after retries", file=sys.stderr)
                    continue
                results.extend(rows)
                csv_path = write_results(results, args.output_dir)
                print(f"Completed {platform} {core} N={grid_size}", flush=True)
    if not results:
        raise SystemExit("No benchmark configurations completed")
    print(f"Results: {csv_path}")


if __name__ == "__main__":
    main()
