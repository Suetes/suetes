#!/usr/bin/env python3
"""Scan the full-horizon PGF loss across increasingly aggressive coordinates."""

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
from suetes.shared.transforms import GalChenSigma, SleveSimple


def _parse_values(text):
    return [float(value) for value in text.split(",") if value.strip()]


def _direct_transform(amplitude, basis_count):
    # Bernstein coefficients sampled from a linear function reproduce
    # log(rho)=a(1-2 eta) exactly for any basis_count >= 2.
    coefficients = jnp.linspace(amplitude, -amplitude, basis_count)
    return LearnedDensityCoordinate({"global": coefficients})


def _evaluate(transform, terrains, steps):
    values, minima = [], []
    for terrain in terrains:
        try:
            metric, _, grid, _ = run_target_case("pgf_rest", transform, terrain, steps)
        except (ValueError, FloatingPointError) as error:
            return {
                "mean_tke": np.inf,
                "std_tke": np.nan,
                "minimum_layer_m": np.nan,
                "samples": len(values),
                "error": str(error),
            }
        values.append(float(metric))
        minima.append(float(jnp.min(grid.dz_m_full)))
    return {
        "mean_tke": float(np.mean(values)),
        "std_tke": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "minimum_layer_m": float(np.min(minima)),
        "samples": len(values),
    }


def _worker(args, dataset, terrains, steps):
    if args.worker_kind == "direct":
        transform = _direct_transform(args.worker_amplitude, args.basis_count)
        result = {
            "kind": "direct",
            "amplitude": args.worker_amplitude,
            **_evaluate(transform, terrains, steps),
        }
    elif args.worker_kind == "galchen":
        result = {
            "kind": "galchen",
            "amplitude": np.nan,
            **_evaluate(GalChenSigma(), terrains, steps),
        }
    else:
        with open(args.sleve_config) as stream:
            config = json.load(stream)
        transform = SleveSimple(
            scale_s=float(config["scale_s"]),
            scale_l=15000.0,
            n=float(config["n"]),
        )
        result = {
            "kind": "sleve",
            "amplitude": np.nan,
            **_evaluate(transform, terrains, steps),
        }
    result["target"] = dataset.get("target", "pgf_rest")
    with open(args.worker_result, "w") as stream:
        json.dump(result, stream, indent=2)


def _run_worker(args, kind, result_path, amplitude=None):
    command = [
        sys.executable,
        os.path.abspath(__file__),
        "--dataset",
        os.path.abspath(args.dataset),
        "--sleve-config",
        os.path.abspath(args.sleve_config),
        "--basis-count",
        str(args.basis_count),
        "--worker-kind",
        kind,
        "--worker-result",
        str(result_path),
    ]
    if amplitude is not None:
        command.extend(("--worker-amplitude", str(amplitude)))
    environment = os.environ.copy()
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    for attempt in range(1, 4):
        completed = subprocess.run(command, check=False, env=environment)
        if completed.returncode == 0 and result_path.exists():
            with open(result_path) as stream:
                return json.load(stream)
        print(
            f"  worker failed (attempt {attempt}/3, "
            f"exit={completed.returncode}); retrying"
        )
    raise RuntimeError(f"Repeated worker failure for {kind}, a={amplitude}")


def _plot(rows, output_dir, minimum_layer):
    direct = [
        row for row in rows if row["kind"] == "direct" and np.isfinite(row["mean_tke"])
    ]
    galchen = next(row for row in rows if row["kind"] == "galchen")
    sleve = next(row for row in rows if row["kind"] == "sleve")
    amplitudes = np.asarray([row["amplitude"] for row in direct])
    tke = np.asarray([row["mean_tke"] for row in direct])
    layers = np.asarray([row["minimum_layer_m"] for row in direct])

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.7))
    axes[0].plot(amplitudes, tke, "o-", label="Direct density profile")
    axes[0].axhline(
        galchen["mean_tke"], color="0.45", linestyle="--", label="Gal--Chen"
    )
    axes[0].axhline(
        sleve["mean_tke"], color="tab:orange", linestyle="--", label="Tuned SLEVE"
    )
    axes[0].set(
        xlabel=r"Aggressiveness $a$",
        ylabel=r"Spurious TKE (m$^2$ s$^{-2}$)",
    )
    axes[0].legend(frameon=False)

    axes[1].plot(amplitudes, layers, "o-", color="tab:green")
    axes[1].axhline(
        minimum_layer,
        color="tab:red",
        linestyle="--",
        label="Acceptance threshold",
    )
    axes[1].axhline(
        sleve["minimum_layer_m"],
        color="tab:orange",
        linestyle=":",
        label="Tuned SLEVE",
    )
    axes[1].set(
        xlabel=r"Aggressiveness $a$",
        ylabel=r"Minimum layer thickness (m)",
    )
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "aggressiveness_scan.png", dpi=250)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sleve-config", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/neuve_aggressiveness_scan")
    )
    parser.add_argument(
        "--amplitudes",
        default="0,0.25,0.5,0.75,1,1.5,2,3,4,5",
        help="Comma-separated values in log(rho)=a(1-2 eta)",
    )
    parser.add_argument("--basis-count", type=int, default=16)
    parser.add_argument("--minimum-layer-m", type=float, default=150.0)
    parser.add_argument(
        "--worker-kind",
        choices=("direct", "galchen", "sleve"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-amplitude", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    if dataset.get("target", "pgf_rest") != "pgf_rest":
        parser.error("The aggressiveness scan supports only pgf_rest")
    terrains = terrains_from_dataset(dataset)
    steps = int(dataset["integration"]["steps"])
    if args.worker_kind:
        if args.worker_result is None:
            parser.error("--worker-result is required internally")
        _worker(args, dataset, terrains, steps)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "aggressiveness_scan_progress.json"
    if progress_path.exists():
        with open(progress_path) as stream:
            rows = json.load(stream)
    else:
        rows = []

    def completed(kind, amplitude=None):
        return next(
            (
                row
                for row in rows
                if row["kind"] == kind
                and (amplitude is None or np.isclose(row["amplitude"], amplitude))
            ),
            None,
        )

    def save_progress():
        with open(progress_path, "w") as stream:
            json.dump(rows, stream, indent=2)

    with tempfile.TemporaryDirectory(prefix="suetes-neuve-scan-") as temporary:
        temporary = Path(temporary)
        for kind in ("galchen", "sleve"):
            if completed(kind):
                print(f"Resuming {kind}")
                continue
            print(f"Evaluating {kind}")
            rows.append(_run_worker(args, kind, temporary / f"{kind}.json"))
            save_progress()
        for index, amplitude in enumerate(_parse_values(args.amplitudes)):
            if completed("direct", amplitude):
                print(f"Resuming aggressiveness a={amplitude:g}")
                continue
            print(f"Evaluating aggressiveness a={amplitude:g}")
            rows.append(
                _run_worker(
                    args,
                    "direct",
                    temporary / f"direct_{index}.json",
                    amplitude,
                )
            )
            save_progress()

    for row in rows:
        row["feasible"] = bool(
            np.isfinite(row["minimum_layer_m"])
            and row["minimum_layer_m"] >= args.minimum_layer_m
        )
    with open(args.output_dir / "aggressiveness_scan.csv", "w", newline="") as stream:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(args.output_dir / "aggressiveness_scan.json", "w") as stream:
        json.dump(rows, stream, indent=2)
    _plot(rows, args.output_dir, args.minimum_layer_m)

    print("\nFull-horizon loss landscape")
    for row in rows:
        label = f"a={row['amplitude']:g}" if row["kind"] == "direct" else row["kind"]
        print(
            f"{label:<12} TKE={row['mean_tke']:.6e} "
            f"min_dz={row['minimum_layer_m']:.2f} m "
            f"feasible={row['feasible']}"
        )
    print(f"Saved scan to {args.output_dir}")


if __name__ == "__main__":
    main()
