#!/usr/bin/env python3
"""Plot dynamical-core performance results from one or more device CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from suetes.shared.experiment import ExperimentLayout

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "font.family": "sans-serif",
    }
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output"


def select_inputs(requested: list[Path] | None, data_dir: Path) -> list[Path]:
    if requested:
        return [path.resolve() for path in requested]
    candidates = sorted(data_dir.glob("benchmark_bubble_scaling_*.csv"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("No scaling CSV found; run rising_bubble_3d_scaling/run.py first")
    return candidates


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"core", "N", "dt_multiplier", "dt", "ms_per_step"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")
    if "kernel_sypd" not in data:
        if "sypd" not in data:
            raise ValueError("CSV contains neither kernel_sypd nor sypd")
        data["kernel_sypd"] = data["sypd"]
    if "vram_peak_increment_mib" not in data:
        if "vram_mb" not in data:
            raise ValueError("CSV contains no usable VRAM column")
        print("Warning: legacy CSV; vram_mb is not a true isolated peak metric")
        data["vram_peak_increment_mib"] = data["vram_mb"]
    if "wall_time_iqr_s" not in data:
        data["wall_time_iqr_s"] = 0.0
    if "num_steps" not in data:
        data["num_steps"] = 1
    data["ms_per_step_iqr"] = data["wall_time_iqr_s"] / data["num_steps"] * 1000.0
    if "device" not in data:
        data["device"] = path.stem.removeprefix("benchmark_bubble_scaling_")
    data["_source_mtime"] = path.stat().st_mtime
    return data.sort_values(["core", "dt_multiplier", "N"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, nargs="+", help="One or more device-specific CSVs (default: all device CSVs)"
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--name", default="default", help="Execution label")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    layout = ExperimentLayout(
        kind="benchmarks", case="rising_bubble_3d_scaling", execution=args.name, output_root=args.output_root
    )
    input_paths = select_inputs(args.input, layout.data)
    frames = []
    for input_path in input_paths:
        frames.append(load_data(input_path))
        print(f"Loaded benchmark data from {input_path}")
    data = pd.concat(frames, ignore_index=True)

    # If the output directory contains repeated runs for the same device,
    # retain the newest measurement for each plotted configuration.
    identity = ["device", "core", "dt_multiplier", "N"]
    duplicate_count = int(data.duplicated(identity, keep=False).sum())
    if duplicate_count:
        data = data.sort_values("_source_mtime").drop_duplicates(identity, keep="last")
        print("Warning: repeated device/configuration measurements found; using the newest values")

    curves = [
        ("split-explicit", 1.0, r"Split-Explicit ($\Delta t=\Delta t_{\mathrm{CFL}}$)", "o", "#e41a1c"),
        ("sisl", 1.0, r"SISL ($\Delta t=\Delta t_{\mathrm{CFL}}$)", "d", "#4daf4a"),
        ("sisl", 10.0, r"SISL ($\Delta t=10\Delta t_{\mathrm{CFL}}$)", "s", "#377eb8"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.05))
    marker_size = 4
    line_width = 1.2
    grid_style = {"ls": "--", "lw": 0.6, "alpha": 0.4, "which": "both"}
    devices = list(dict.fromkeys(data["device"].astype(str)))
    line_styles = ["-", "--", "-.", ":"]

    for device_index, device in enumerate(devices):
        line_style = line_styles[device_index % len(line_styles)]
        for core, multiplier, label, marker, color in curves:
            subset = data[
                (data["device"].astype(str) == device)
                & (data["core"] == core)
                & np.isclose(data["dt_multiplier"], multiplier)
            ].sort_values("N")
            if subset.empty:
                continue
            plot_label = label if len(devices) == 1 else f"{label} — {device}"
            axes[0].semilogy(
                subset["N"],
                subset["kernel_sypd"],
                marker=marker,
                linestyle=line_style,
                color=color,
                lw=line_width,
                ms=marker_size,
                label=plot_label,
            )
            axes[1].errorbar(
                subset["N"],
                subset["ms_per_step"],
                yerr=0.5 * subset["ms_per_step_iqr"],
                marker=marker,
                linestyle=line_style,
                color=color,
                lw=line_width,
                ms=marker_size,
                capsize=2,
                label=plot_label,
            )
            axes[2].loglog(
                subset["N"],
                subset["vram_peak_increment_mib"],
                marker=marker,
                linestyle=line_style,
                color=color,
                lw=line_width,
                ms=marker_size,
                label=plot_label,
            )

    # With fixed ns, an outer split-explicit step represents the same
    # algorithmic workload at every resolution. Since the domain contains N^3
    # cells, this is a meaningful ideal compute-scaling guide again. Anchor it
    # at the largest split-explicit measurement to emphasize the scaling slope
    # rather than an absolute performance prediction.
    split_reference = data[
        (data["device"].astype(str) == devices[0])
        & (data["core"] == "split-explicit")
        & np.isclose(data["dt_multiplier"], 1.0)
    ].sort_values("N")
    if len(split_reference) > 1:
        n_reference = split_reference["N"].to_numpy(dtype=float)
        latency_reference = float(split_reference["ms_per_step"].iloc[-1]) * (n_reference / n_reference[-1]) ** 3
        axes[1].plot(n_reference, latency_reference, "k--", lw=1.0, alpha=0.75, label=r"Ideal $\mathcal{O}(N^3)$")

    for axis, label in zip(axes, ("a)", "b)", "c)")):
        axis.text(-0.05, 1.01, label, transform=axis.transAxes, fontsize=10, ha="right", va="bottom")

    axes[0].set_ylabel("Kernel throughput (SYPD)")

    axes[1].set_ylabel("Median execution time (ms/step)")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")

    axes[2].set_ylabel("Peak allocation above baseline (MiB)")

    ticks = sorted(data["N"].unique())
    for axis in axes:
        axis.set_xlabel(r"Grid dimension ($N\times N\times N$)")
        axis.set_xticks(ticks)
        axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        axis.minorticks_off()
        axis.grid(True, **grid_style)

    handles, labels = axes[1].get_legend_handles_labels()
    curve_entries = [(handle, label) for handle, label in zip(handles, labels) if not label.startswith("Ideal")]
    reference_entries = [(handle, label) for handle, label in zip(handles, labels) if label.startswith("Ideal")]
    handles, labels = zip(*(curve_entries + reference_entries))
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        fontsize=9,
        frameon=False,
    )

    output_dir = args.output_dir or layout.figures
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "suetes_scaling_metrics.png"
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Figure saved to {output_path}")


if __name__ == "__main__":
    main()
