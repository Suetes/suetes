#!/usr/bin/env python3
"""Quantitative long-horizon, cross-core refinement audit.

The default ``--run`` configuration integrates the rising bubble on 100, 50,
and 25 m grids to 1000 s.  Existing plot-ready artifacts can instead be
supplied with repeated ``--artifact`` options, ordered coarse to fine.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import xarray as xr

from suetes.shared.artifacts import ArtifactLayout, artifact_from_bundle


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = REPO_ROOT / "benchmarks" / "physical" / "rising_bubble_3d.py"
CORES = ("sisl", "split-explicit")


def _positive_moments(
    field: np.ndarray, z: np.ndarray
) -> tuple[float, float, float]:
    positive = np.maximum(field, 0.0)
    total = float(positive.sum())
    if total <= np.finfo(float).tiny:
        return float("nan"), float("nan"), float("nan")
    centroid = float((positive * z[None, :]).sum() / total)
    return centroid, float(np.max(field)), total


def _relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    reference = np.linalg.norm(0.5 * (left + right))
    return float(
        np.linalg.norm(left - right) / max(reference, np.finfo(float).tiny)
    )


def _restrict(field: np.ndarray, ratio: int) -> np.ndarray:
    if ratio == 1:
        return field
    nx, nz = field.shape
    if nx % ratio or nz % ratio:
        raise ValueError(
            f"Fine field shape {field.shape} is not divisible by ratio {ratio}"
        )
    return field.reshape(nx // ratio, ratio, nz // ratio, ratio).mean((1, 3))


def _open_artifacts(paths: list[Path]) -> list[xr.Dataset]:
    datasets = [xr.load_dataset(artifact_from_bundle(path)) for path in paths]
    datasets.sort(key=lambda ds: float(ds.attrs["dx_m"]), reverse=True)
    if len(datasets) < 3:
        raise ValueError("At least three refinement artifacts are required")
    expected_times = datasets[0].time.values
    for dataset in datasets:
        if set(dataset.core.values.astype(str)) != set(CORES):
            raise ValueError(f"Expected cores {CORES}, found {dataset.core.values}")
        if not np.allclose(dataset.time.values, expected_times):
            raise ValueError("All artifacts must contain identical snapshot times")
        if float(dataset.attrs.get("stabilization", 0.0)) != float(
            datasets[0].attrs.get("stabilization", 0.0)
        ):
            raise ValueError("All artifacts must use identical stabilization")
    return datasets


def _run_artifacts(args: argparse.Namespace) -> list[Path]:
    paths = []
    for dx in args.resolutions:
        dt = args.dt_per_metre * dx
        name = f"{args.name}-dx{dx:g}"
        command = [
            sys.executable, "-u", str(BENCHMARK),
            "--dx", str(dx), "--dt", str(dt),
            "--t-end", str(args.t_end),
            "--snapshot-interval", str(args.snapshot_interval),
            "--stabilization", str(args.stabilization),
            "--output-root", str(args.output_root),
            "--name", name, "--no-render",
        ]
        if args.reuse:
            command.append("--reuse")
        print("Running:", " ".join(command))
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        paths.append(
            args.output_root / "benchmarks" / "rising_bubble_3d"
            / name / "data" / "artifact.nc"
        )
    return paths


def analyze(
    datasets: list[xr.Dataset],
    *,
    early_time: float,
    min_early_order: float,
    max_centroid_cells: float,
    max_integral_relative_difference: float,
    max_mass_drift: float,
) -> tuple[list[dict], dict]:
    dx_values = [float(ds.attrs["dx_m"]) for ds in datasets]
    times = datasets[0].time.values.astype(float)
    positive_times = times[times > 0.0]
    if not len(positive_times):
        raise ValueError("Artifacts contain no positive snapshot time")
    early = float(positive_times[np.argmin(np.abs(positive_times - early_time))])

    rows: list[dict] = []
    cross_by_time: dict[float, list[float]] = {}
    for dataset, dx in zip(datasets, dx_values):
        per_time = []
        for time in times:
            sisl = dataset.theta_perturbation.sel(core="sisl", time=time).values
            split = dataset.theta_perturbation.sel(
                core="split-explicit", time=time
            ).values
            metric = _relative_l2(sisl, split)
            per_time.append(metric)
            rows.append({
                "metric": "cross_core_relative_l2", "core": "cross-core",
                "coarse_dx_m": dx, "fine_dx_m": dx,
                "time_s": float(time), "value": metric,
            })
        cross_by_time[dx] = per_time

    self_final: dict[str, list[float]] = {core: [] for core in CORES}
    for coarse, fine in zip(datasets[:-1], datasets[1:]):
        coarse_dx = float(coarse.attrs["dx_m"])
        fine_dx = float(fine.attrs["dx_m"])
        ratio_float = coarse_dx / fine_dx
        ratio = round(ratio_float)
        if not np.isclose(ratio_float, ratio):
            raise ValueError("Adjacent grid spacings must have an integer ratio")
        for core in CORES:
            for time in times:
                coarse_field = coarse.theta_perturbation.sel(
                    core=core, time=time
                ).values
                fine_field = fine.theta_perturbation.sel(
                    core=core, time=time
                ).values
                metric = _relative_l2(coarse_field, _restrict(fine_field, ratio))
                rows.append({
                    "metric": "self_relative_l2", "core": core,
                    "coarse_dx_m": coarse_dx, "fine_dx_m": fine_dx,
                    "time_s": float(time), "value": metric,
                })
                if np.isclose(time, times[-1]):
                    self_final[core].append(metric)

    early_index = int(np.flatnonzero(np.isclose(times, early))[0])
    early_cross = [cross_by_time[dx][early_index] for dx in dx_values]
    final_cross = [cross_by_time[dx][-1] for dx in dx_values]
    early_orders = [
        math.log(left / right) / math.log(dx_values[i] / dx_values[i + 1])
        for i, (left, right) in enumerate(zip(early_cross[:-1], early_cross[1:]))
    ]

    finest = datasets[-1]
    final_fields = {
        core: finest.theta_perturbation.sel(core=core, time=times[-1]).values
        for core in CORES
    }
    moments = {
        core: _positive_moments(field, finest.z.values)
        for core, field in final_fields.items()
    }
    centroid_difference = abs(moments["sisl"][0] - moments["split-explicit"][0])
    peak_relative_difference = abs(
        moments["sisl"][1] - moments["split-explicit"][1]
    ) / max(abs(moments["split-explicit"][1]), np.finfo(float).tiny)
    integral_relative_difference = abs(
        moments["sisl"][2] - moments["split-explicit"][2]
    ) / max(abs(moments["split-explicit"][2]), np.finfo(float).tiny)
    mass_drifts = {
        core: float(finest.mass_drift.sel(core=core)) for core in CORES
    }

    gates = {
        "cross_error_decreases_at_final_time": all(
            right < left for left, right in zip(final_cross[:-1], final_cross[1:])
        ),
        "early_cross_order": min(early_orders) >= min_early_order,
        "self_error_decreases_at_final_time": all(
            all(right < left for left, right in zip(values[:-1], values[1:]))
            for values in self_final.values()
        ),
        "fine_centroid_agreement": centroid_difference
        <= max_centroid_cells * dx_values[-1],
        "fine_positive_anomaly_integral_agreement":
        integral_relative_difference <= max_integral_relative_difference,
        "mass_conservation": max(mass_drifts.values()) <= max_mass_drift,
    }
    summary = {
        "passed": all(gates.values()),
        "gates": gates,
        "dx_m": dx_values,
        "early_time_s": early,
        "final_time_s": float(times[-1]),
        "early_cross_relative_l2": early_cross,
        "early_cross_orders": early_orders,
        "final_cross_relative_l2": final_cross,
        "final_self_relative_l2": self_final,
        "finest_centroid_difference_m": centroid_difference,
        "finest_positive_anomaly_integral_relative_difference":
        integral_relative_difference,
        # A gridpoint peak is phase sensitive at long horizons. Report it for
        # diagnosis, but use the integral above as the quantitative gate.
        "finest_peak_relative_difference": peak_relative_difference,
        "finest_mass_drift": mass_drifts,
        "thresholds": {
            "min_early_order": min_early_order,
            "max_centroid_cells": max_centroid_cells,
            "max_integral_relative_difference":
            max_integral_relative_difference,
            "max_mass_drift": max_mass_drift,
        },
    }
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run", action="store_true", help="Generate the refinement artifacts"
    )
    source.add_argument(
        "--artifact", type=Path, action="append",
        help="Artifact file or bundle; repeat at least three times",
    )
    parser.add_argument("--resolutions", type=float, nargs="+", default=[100, 50, 25])
    parser.add_argument("--dt-per-metre", type=float, default=0.02)
    parser.add_argument("--t-end", type=float, default=1000.0)
    parser.add_argument("--snapshot-interval", type=float, default=200.0)
    parser.add_argument("--stabilization", type=float, default=0.0)
    parser.add_argument("--early-time", type=float, default=400.0)
    parser.add_argument("--min-early-order", type=float, default=1.25)
    parser.add_argument("--max-centroid-cells", type=float, default=2.0)
    parser.add_argument(
        "--max-integral-relative-difference", type=float, default=0.02
    )
    parser.add_argument("--max-mass-drift", type=float, default=1.0e-5)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--name", default="long-horizon")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_paths = _run_artifacts(args) if args.run else args.artifact
    assert artifact_paths is not None
    datasets = _open_artifacts(artifact_paths)
    rows, summary = analyze(
        datasets,
        early_time=args.early_time,
        min_early_order=args.min_early_order,
        max_centroid_cells=args.max_centroid_cells,
        max_integral_relative_difference=args.max_integral_relative_difference,
        max_mass_drift=args.max_mass_drift,
    )
    layout = ArtifactLayout(
        kind="verification", case="long_horizon_core_audit",
        execution=args.name, output_root=args.output_root,
    ).create()
    csv_path = layout.data / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    json_path = layout.data / "summary.json"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Metrics: {csv_path}")
    print(f"Summary: {json_path}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
