#!/usr/bin/env python3
"""Plot dynamical-core performance results as separate label-free panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from suetes.shared.experiment import ExperimentLayout, figure_dir_for

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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "figures"
CSV_PATTERN = "benchmark_bubble_scaling_*.csv"

# Every panel shares one canvas and one set of margins, so the axes land on
# identical pixels and a plain north append in make_fig.sh keeps them level.
# Do not -trim the outputs; that padding is what makes the row line up.
PANEL_W_IN = 6.0
PANEL_H_IN = 5.5
MARGINS = dict(left=0.20, right=0.97, bottom=0.15, top=0.96)

MARKER_SIZE = 4
LINE_WIDTH = 1.2
GRID_STYLE = {"ls": "--", "lw": 0.6, "alpha": 0.4, "which": "both"}

# Configuration is carried by colour and marker shape, device by linestyle.
# The ideal guide takes a long-dash grey that no device style reuses.
DEVICE_STYLES = ["-", "--", "-.", ":"]
IDEAL_STYLE = dict(color="0.35", ls=(0, (7, 3)), lw=1.0)

CURVES = (
    ("split-explicit", 1.0, r"Split-Explicit ($\Delta t=\Delta t_{\mathrm{CFL}}$)", "o", "#e41a1c"),
    ("sisl", 1.0, r"SISL ($\Delta t=\Delta t_{\mathrm{CFL}}$)", "d", "#4daf4a"),
    ("sisl", 10.0, r"SISL ($\Delta t=10\Delta t_{\mathrm{CFL}}$)", "s", "#377eb8"),
)

# stem, column, ylabel, xscale, yscale
PANELS = (
    ("suetes_scaling_throughput", "kernel_sypd", "Kernel throughput (SYPD)", "linear", "log"),
    ("suetes_scaling_latency", "ms_per_step", "Median execution time (ms/step)", "log", "log"),
    ("suetes_scaling_vram", "vram_peak_increment_mib", "Peak allocation above baseline (MiB)", "log", "log"),
)

ERRORBAR_PANEL = "suetes_scaling_latency"
IDEAL_PANEL = "suetes_scaling_latency"
XLABEL = r"Grid dimension ($N\times N\times N$)"


def discover_inputs(name: str, output_root: Path) -> list[Path]:
    """Return the CSVs beside this script, else those in the standard bundle."""
    local = sorted(SCRIPT_DIR.glob(CSV_PATTERN), key=lambda path: path.stat().st_mtime)
    if local:
        return local
    layout = ExperimentLayout(
        kind="benchmarks",
        case="rising_bubble_3d_scaling",
        execution=name,
        output_root=output_root,
    )
    candidates = sorted(layout.data.glob(CSV_PATTERN), key=lambda path: path.stat().st_mtime)
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


def _new_panel():
    fig = plt.figure(figsize=(PANEL_W_IN, PANEL_H_IN))
    axis = fig.add_subplot(111)
    fig.subplots_adjust(**MARGINS)
    return fig, axis


def _write_legend(devices: list[str], has_ideal: bool, output_dir: Path) -> Path:
    handles = [
        Line2D([], [], color=colour, marker=marker, ls="-", lw=LINE_WIDTH, ms=MARKER_SIZE)
        for _core, _mult, _label, marker, colour in CURVES
    ]
    labels = [label for _core, _mult, label, _marker, _colour in CURVES]

    # Device only earns its own legend block when more than one is plotted;
    # otherwise every curve already shares a single solid linestyle.
    if len(devices) > 1:
        for index, device in enumerate(devices):
            handles.append(Line2D([], [], color="black", ls=DEVICE_STYLES[index % len(DEVICE_STYLES)], lw=LINE_WIDTH))
            labels.append(str(device))

    if has_ideal:
        handles.append(Line2D([], [], **IDEAL_STYLE))
        labels.append(r"Ideal $\mathcal{O}(N^3)$")

    fig = plt.figure(figsize=(PANEL_W_IN * 3, 1.5))
    legend = fig.legend(handles, labels, loc="center", ncol=min(4, len(labels)), frameon=False)
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    output = output_dir / "suetes_scaling_legend.png"
    fig.savefig(output, bbox_inches=bbox.expanded(1.05, 1.3), facecolor="white")
    plt.close(fig)
    return output


def render(
    inputs: list[Path] | None = None,
    output_dir: Path | None = None,
    *,
    name: str = "default",
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> list[Path]:
    if inputs:
        input_paths = [path.resolve() for path in inputs]
        for path in input_paths:
            if not path.exists():
                raise FileNotFoundError(f"No such CSV: {path}")
    else:
        input_paths = discover_inputs(name, output_root)

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

    devices = list(dict.fromkeys(data["device"].astype(str)))
    ticks = sorted(data["N"].unique())

    # With fixed ns, an outer split-explicit step represents the same
    # algorithmic workload at every resolution. Since the domain contains N^3
    # cells, this is a meaningful ideal compute-scaling guide. Anchor it at the
    # largest split-explicit measurement to emphasize the scaling slope rather
    # than an absolute performance prediction.
    split_reference = data[
        (data["device"].astype(str) == devices[0])
        & (data["core"] == "split-explicit")
        & np.isclose(data["dt_multiplier"], 1.0)
    ].sort_values("N")
    ideal = None
    if len(split_reference) > 1:
        n_reference = split_reference["N"].to_numpy(dtype=float)
        ideal = (
            n_reference,
            float(split_reference["ms_per_step"].iloc[-1]) * (n_reference / n_reference[-1]) ** 3,
        )

    # Explicit CSVs keep their bundle figure directory; a bare run is sent to
    # the script-local figure directory by main().
    output_dir = output_dir or figure_dir_for(input_paths[0])
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for stem, column, ylabel, xscale, yscale in PANELS:
        fig, axis = _new_panel()

        for device_index, device in enumerate(devices):
            line_style = DEVICE_STYLES[device_index % len(DEVICE_STYLES)]
            for core, multiplier, _label, marker, colour in CURVES:
                subset = data[
                    (data["device"].astype(str) == device)
                    & (data["core"] == core)
                    & np.isclose(data["dt_multiplier"], multiplier)
                ].sort_values("N")
                if subset.empty:
                    continue
                if stem == ERRORBAR_PANEL:
                    axis.errorbar(
                        subset["N"],
                        subset[column],
                        yerr=0.5 * subset["ms_per_step_iqr"],
                        marker=marker,
                        linestyle=line_style,
                        color=colour,
                        lw=LINE_WIDTH,
                        ms=MARKER_SIZE,
                        capsize=2,
                    )
                else:
                    axis.plot(
                        subset["N"],
                        subset[column],
                        marker=marker,
                        linestyle=line_style,
                        color=colour,
                        lw=LINE_WIDTH,
                        ms=MARKER_SIZE,
                    )

        if stem == IDEAL_PANEL and ideal is not None:
            axis.plot(*ideal, **IDEAL_STYLE)

        axis.set_xscale(xscale)
        axis.set_yscale(yscale)
        axis.set_xlabel(XLABEL)
        axis.set_ylabel(ylabel)
        axis.set_xticks(ticks)
        axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        axis.minorticks_off()
        axis.grid(True, **GRID_STYLE)

        output_path = output_dir / f"{stem}_main.png"
        fig.savefig(output_path, facecolor="white")
        plt.close(fig)
        outputs.append(output_path)

    outputs.append(_write_legend(devices, ideal is not None, output_dir))
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="*",
        help="One or more device-specific CSVs (default: every device CSV beside this script or in the bundle)",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--name", default="default", help="Execution label")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None and not args.inputs:
        output_dir = DEFAULT_OUTPUT_DIR
    written = render(args.inputs, output_dir, name=args.name, output_root=args.output_root)
    for output_path in written:
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
