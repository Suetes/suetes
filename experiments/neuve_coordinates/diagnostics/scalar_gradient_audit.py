#!/usr/bin/env python3
"""Audit the scalar coordinate gradient against centered finite differences."""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from experiments._shared.neuve_coordinate import (
    load_dataset,
    run_target_case,
    terrains_from_dataset,
)
from experiments._shared.neuve_learned_coordinate import (
    LearnedDensityCoordinate,
)


def _values(text, converter=float):
    return [converter(value) for value in text.split(",") if value.strip()]


def _transform(amplitude, basis_count):
    return LearnedDensityCoordinate(
        {"global": jnp.linspace(amplitude, -amplitude, basis_count)}
    )


def _worker(args):
    dataset = load_dataset(args.dataset)
    terrain = terrains_from_dataset(dataset)[args.terrain_index]

    def objective(amplitude):
        metric, _, grid, _ = run_target_case(
            "pgf_rest",
            _transform(amplitude, args.basis_count),
            terrain,
            args.worker_steps,
        )
        return metric, jnp.min(grid.dz_m_full)

    value_gradient = jax.jit(jax.value_and_grad(objective, has_aux=True))
    forward = jax.jit(objective)
    amplitude = jnp.asarray(args.worker_amplitude)
    (value, minimum), gradient = value_gradient(amplitude)
    value = float(value)
    row = {
        "amplitude": args.worker_amplitude,
        "steps": args.worker_steps,
        "objective": value,
        "minimum_layer_m": float(minimum),
        "ad_gradient": float(gradient),
    }
    for epsilon in _values(args.epsilons):
        plus = float(forward(amplitude + epsilon)[0])
        minus = float(forward(amplitude - epsilon)[0])
        finite_difference = (plus - minus) / (2.0 * epsilon)
        remainder_plus = abs(plus - value - epsilon * row["ad_gradient"])
        remainder_minus = abs(minus - value + epsilon * row["ad_gradient"])
        suffix = f"{epsilon:.0e}"
        row[f"fd_{suffix}"] = finite_difference
        row[f"relative_error_{suffix}"] = abs(
            finite_difference - row["ad_gradient"]
        ) / max(
            abs(finite_difference),
            abs(row["ad_gradient"]),
            1.0e-14,
        )
        row[f"remainder_plus_{suffix}"] = remainder_plus
        row[f"remainder_minus_{suffix}"] = remainder_minus
    with open(args.worker_result, "w") as stream:
        json.dump(row, stream, indent=2)


def _run_worker(args, amplitude, steps, destination):
    command = [
        sys.executable,
        os.path.abspath(__file__),
        "--dataset",
        os.path.abspath(args.dataset),
        "--basis-count",
        str(args.basis_count),
        "--terrain-index",
        str(args.terrain_index),
        "--epsilons",
        args.epsilons,
        "--worker-amplitude",
        str(amplitude),
        "--worker-steps",
        str(steps),
        "--worker-result",
        str(destination),
    ]
    environment = os.environ.copy()
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    for attempt in range(1, 4):
        completed = subprocess.run(command, check=False, env=environment)
        if completed.returncode == 0 and destination.exists():
            with open(destination) as stream:
                return json.load(stream)
        print(
            f"  worker failed (attempt {attempt}/3, "
            f"exit={completed.returncode}); retrying"
        )
    raise RuntimeError(f"Gradient worker failed for a={amplitude}, steps={steps}")


def _plot(rows, output_dir, reference_amplitude, full_steps, epsilon):
    suffix = f"{epsilon:.0e}"
    horizon = sorted(
        (row for row in rows if np.isclose(row["amplitude"], reference_amplitude)),
        key=lambda row: row["steps"],
    )
    amplitude = sorted(
        (row for row in rows if row["steps"] == full_steps),
        key=lambda row: row["amplitude"],
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.7))
    axes[0].plot(
        [row["steps"] for row in horizon],
        [row["ad_gradient"] for row in horizon],
        "o-",
        label="Reverse-mode AD",
    )
    axes[0].plot(
        [row["steps"] for row in horizon],
        [row[f"fd_{suffix}"] for row in horizon],
        "s--",
        label="Centered finite difference",
    )
    axes[0].axhline(0.0, color="0.4", linewidth=0.8)
    axes[0].set(
        xlabel="Integration steps",
        ylabel=r"$\partial\mathcal{J}/\partial a$",
    )
    axes[0].legend(frameon=False)

    axes[1].plot(
        [row["amplitude"] for row in amplitude],
        [row["ad_gradient"] for row in amplitude],
        "o-",
        label="Reverse-mode AD",
    )
    axes[1].plot(
        [row["amplitude"] for row in amplitude],
        [row[f"fd_{suffix}"] for row in amplitude],
        "s--",
        label="Centered finite difference",
    )
    axes[1].axhline(0.0, color="0.4", linewidth=0.8)
    axes[1].set(
        xlabel=r"Aggressiveness $a$",
        ylabel=r"$\partial\mathcal{J}/\partial a$",
    )
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "scalar_gradient_audit.png", dpi=250)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/neuve_scalar_gradient_audit"),
    )
    parser.add_argument("--basis-count", type=int, default=16)
    parser.add_argument("--terrain-index", type=int, default=0)
    parser.add_argument(
        "--amplitudes",
        default="0,0.5,1,2",
        help="Full-horizon amplitudes",
    )
    parser.add_argument(
        "--horizons",
        default="25,50,100,250,500",
        help="Integration steps audited at the reference amplitude",
    )
    parser.add_argument("--reference-amplitude", type=float, default=0.5)
    parser.add_argument("--epsilons", default="1e-3,3e-3,1e-2")
    parser.add_argument("--worker-amplitude", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--worker-steps", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_result is not None:
        _worker(args)
        return

    dataset = load_dataset(args.dataset)
    full_steps = int(dataset["integration"]["steps"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "scalar_gradient_audit_progress.json"
    if progress_path.exists():
        with open(progress_path) as stream:
            rows = json.load(stream)
    else:
        rows = []

    cases = {(args.reference_amplitude, steps) for steps in _values(args.horizons, int)}
    cases.update((amplitude, full_steps) for amplitude in _values(args.amplitudes))

    def existing(amplitude, steps):
        return any(
            row["steps"] == steps and np.isclose(row["amplitude"], amplitude)
            for row in rows
        )

    with tempfile.TemporaryDirectory(prefix="suetes-gradient-audit-") as temporary:
        temporary = Path(temporary)
        for index, (amplitude, steps) in enumerate(sorted(cases)):
            if existing(amplitude, steps):
                print(f"Resuming a={amplitude:g}, steps={steps}")
                continue
            print(f"Auditing a={amplitude:g}, steps={steps}")
            rows.append(
                _run_worker(
                    args,
                    amplitude,
                    steps,
                    temporary / f"case_{index}.json",
                )
            )
            with open(progress_path, "w") as stream:
                json.dump(rows, stream, indent=2)

    rows.sort(key=lambda row: (row["steps"], row["amplitude"]))
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with open(
        args.output_dir / "scalar_gradient_audit.csv",
        "w",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(args.output_dir / "scalar_gradient_audit.json", "w") as stream:
        json.dump(rows, stream, indent=2)
    epsilon = min(_values(args.epsilons))
    _plot(
        rows,
        args.output_dir,
        args.reference_amplitude,
        full_steps,
        epsilon,
    )

    suffix = f"{epsilon:.0e}"
    print("\nScalar gradient audit")
    print(" amplitude  steps       AD gradient       FD gradient    relative error")
    for row in rows:
        print(
            f" {row['amplitude']:9.3f} {row['steps']:6d} "
            f"{row['ad_gradient']:17.8e} {row[f'fd_{suffix}']:17.8e} "
            f"{row[f'relative_error_{suffix}']:17.8e}"
        )
    print(f"Saved audit to {args.output_dir}")


if __name__ == "__main__":
    main()
