#!/usr/bin/env python3
"""Evaluate fixed coordinates across off-reference resting stratifications."""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import numpy as np

from experiments._shared.neuve_coordinate import REFERENCE_N_BV, load_dataset, run_target_case, terrains_from_dataset
from experiments._shared.neuve_learned_coordinate import load_learned_coordinate
from suetes.shared.transforms import GalChenSigma, SleveSimple


def _parse_values(text):
    values = [float(value) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("At least one buoyancy frequency is required")
    return values


def _coordinates(args):
    with open(args.sleve_config) as stream:
        sleve = json.load(stream)
    return {
        "Gal-Chen": GalChenSigma(),
        "Tuned SLEVE": SleveSimple(scale_s=float(sleve["scale_s"]), scale_l=15000.0, n=float(sleve["n"])),
        "NEUVE": load_learned_coordinate(args.neuve_weights),
    }


def _run_worker(args):
    dataset = load_dataset(args.dataset)
    if dataset.get("target", "pgf_rest") != "pgf_rest":
        raise ValueError("The stratification sweep requires a PGF-rest dataset")
    terrains = terrains_from_dataset(dataset)
    steps = args.steps or int(dataset["integration"]["steps"])
    rows = []
    for coordinate_name, transform in _coordinates(args).items():
        print(f"N={args.worker_n_bv:.5f} s^-1, {coordinate_name}", flush=True)
        for sample, (seed, terrain) in enumerate(zip(dataset["seeds"], terrains)):
            metric, _, _, diagnostics = run_target_case(
                "pgf_rest", transform, terrain, steps, resting_n_bv=args.worker_n_bv, reference_n_bv=args.reference_n_bv
            )
            rows.append(
                {
                    "resting_n_bv_s-1": args.worker_n_bv,
                    "reference_n_bv_s-1": args.reference_n_bv,
                    "sample": sample,
                    "seed": seed,
                    "coordinate": coordinate_name,
                    "mean_tke": float(metric),
                    "minimum_layer_m": float(diagnostics[0]),
                }
            )
    with open(args.worker_result, "w") as stream:
        json.dump(rows, stream, indent=2)


def _worker_command(args, n_bv, result):
    command = [
        sys.executable,
        os.path.abspath(__file__),
        "--dataset",
        os.path.abspath(args.dataset),
        "--neuve-weights",
        os.path.abspath(args.neuve_weights),
        "--sleve-config",
        os.path.abspath(args.sleve_config),
        "--reference-n-bv",
        str(args.reference_n_bv),
        "--worker-n-bv",
        str(n_bv),
        "--worker-result",
        str(result),
        "--output-dir",
        str(args.output_dir),
    ]
    if args.steps:
        command.extend(("--steps", str(args.steps)))
    return command


def _collect(args, values):
    rows = []
    environment = os.environ.copy()
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    with tempfile.TemporaryDirectory(prefix="suetes-neuve-n-sweep-") as temporary:
        for index, n_bv in enumerate(values):
            result = Path(temporary) / f"n_{index}.json"
            for attempt in range(1, 4):
                completed = subprocess.run(_worker_command(args, n_bv, result), check=False, env=environment)
                if completed.returncode == 0 and result.exists():
                    with open(result) as stream:
                        rows.extend(json.load(stream))
                    break
                print(f"Worker for N={n_bv:g} failed (attempt {attempt}/3, exit={completed.returncode})")
            else:
                raise RuntimeError(f"Repeated worker failure for N={n_bv:g}")
    return rows


def _summarize(rows, relative_floor, reference_n_bv):
    summary = []
    n_values = sorted({row["resting_n_bv_s-1"] for row in rows})
    for n_bv in n_values:
        selected_n = [row for row in rows if row["resting_n_bv_s-1"] == n_bv]
        arrays = {}
        for coordinate in ("Gal-Chen", "Tuned SLEVE", "NEUVE"):
            values = np.asarray([row["mean_tke"] for row in selected_n if row["coordinate"] == coordinate])
            arrays[coordinate] = values
            summary.append(
                {
                    "resting_n_bv_s-1": n_bv,
                    "coordinate": coordinate,
                    "mean_tke": float(np.mean(values)),
                    "std_tke": (float(np.std(values, ddof=1)) if len(values) > 1 else 0.0),
                    "samples": len(values),
                }
            )
        sleve = arrays["Tuned SLEVE"]
        neuve = arrays["NEUVE"]
        matched_reference = np.isclose(n_bv, reference_n_bv, rtol=0.0, atol=1.0e-12)
        reliable = float(np.mean(sleve)) > relative_floor and not matched_reference
        reduction = 100.0 * (1.0 - neuve / sleve) if reliable else None
        summary.append(
            {
                "resting_n_bv_s-1": n_bv,
                "coordinate": "NEUVE vs SLEVE",
                "mean_reduction_percent": (float(np.mean(reduction)) if reliable else None),
                "neuve_wins": int(np.count_nonzero(neuve < sleve)),
                "samples": len(neuve),
                "relative_metric_reliable": reliable,
                "matched_reference_negative_control": bool(matched_reference),
            }
        )
    return summary


def _plot(summary, output_dir, reference_n_bv):
    coordinates = ("Gal-Chen", "Tuned SLEVE", "NEUVE")
    colors = dict(zip(coordinates, ("0.35", "tab:orange", "tab:blue")))
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for coordinate in coordinates:
        selected = [row for row in summary if row["coordinate"] == coordinate]
        x = np.asarray([row["resting_n_bv_s-1"] for row in selected])
        mean = np.asarray([row["mean_tke"] for row in selected])
        std = np.asarray([row["std_tke"] for row in selected])
        plot_mean = np.maximum(mean, np.finfo(float).tiny)
        axes[0].plot(x, plot_mean, "o-", color=colors[coordinate], label=coordinate)
        axes[0].fill_between(x, np.maximum(mean - std, np.finfo(float).tiny), mean + std, color=colors[coordinate], alpha=0.14)
    axes[0].axvline(reference_n_bv, color="0.6", linestyle=":", linewidth=1.0)
    axes[0].set_yscale("log")
    axes[0].set(xlabel=r"Resting-state $N$ (s$^{-1}$)", ylabel=r"Mean spurious TKE (m$^2$ s$^{-2}$)")
    axes[0].legend(frameon=False)

    relative = [row for row in summary if row["coordinate"] == "NEUVE vs SLEVE" and row["relative_metric_reliable"]]
    x = np.asarray([row["resting_n_bv_s-1"] for row in relative])
    reduction = np.asarray([row["mean_reduction_percent"] for row in relative])
    axes[1].plot(x, reduction, "o-", color="tab:blue")
    axes[1].axhline(0.0, color="0.4", linewidth=1.0)
    axes[1].set(xlabel=r"Resting-state $N$ (s$^{-1}$)", ylabel="NEUVE reduction vs SLEVE (%)")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.ticklabel_format(axis="x", style="sci", scilimits=(-2, -2))
    figure.tight_layout()
    figure.savefig(output_dir / "stratification_sweep.png", dpi=250)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--neuve-weights", required=True)
    parser.add_argument("--sleve-config", required=True)
    parser.add_argument(
        "--n-values", default="0.01,0.0125,0.015,0.0175,0.02", help="Comma-separated physical resting-state buoyancy frequencies"
    )
    parser.add_argument("--reference-n-bv", type=float, default=REFERENCE_N_BV)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--relative-floor", type=float, default=1.0e-12)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-n-bv", type=float)
    parser.add_argument("--worker-result")
    args = parser.parse_args()
    if args.worker_result:
        _run_worker(args)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _collect(args, _parse_values(args.n_values))
    with open(args.output_dir / "stratification_sweep.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = _summarize(rows, args.relative_floor, args.reference_n_bv)
    with open(args.output_dir / "stratification_sweep.json", "w") as stream:
        json.dump(summary, stream, indent=2)
    _plot(summary, args.output_dir, args.reference_n_bv)

    print("\nStratification robustness summary")
    for row in summary:
        if row["coordinate"] != "NEUVE vs SLEVE":
            continue
        reduction = row["mean_reduction_percent"]
        message = f"{reduction:.2f}%" if reduction is not None else "not reported (near-zero SLEVE error)"
        print(f"N={row['resting_n_bv_s-1']:.5f} s^-1: NEUVE reduction={message}, wins={row['neuve_wins']}/{row['samples']}")
    print(f"Saved sweep to {args.output_dir}")


if __name__ == "__main__":
    main()
