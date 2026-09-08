#!/usr/bin/env python3
"""Render reverse-mode scaling diagnostics as separate label-free panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from suetes.shared.experiment import ExperimentLayout, artifact_from_bundle, figure_dir_for

plt.rcParams.update(
    {
        "font.size": 16,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "font.family": "sans-serif",
    }
)

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_NAME = "single_gpu_domain_scaling.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "figures"

# Every panel shares one canvas and one set of margins, so the axes land on
# identical pixels and a plain north append in make_fig.sh keeps them level.
# Do not -trim the outputs; that padding is what makes the rows line up.
PANEL_W_IN = 6.0
PANEL_H_IN = 6.0
MARGINS = dict(left=0.20, right=0.97, bottom=0.13, top=0.95)

# Mode is carried by colour and marker shape. Pass type is carried by
# linestyle and marker fill. Neither channel is reused by the other, so the
# forward curves stay separable and the figure survives grayscale printing.
MARKERS = ["o", "s", "^", "D"]
GRAD_KW = dict(ls="-", linewidth=2.0, ms=8)
FWD_KW = dict(ls=":", linewidth=2.0, ms=8, markerfacecolor="none", markeredgewidth=1.5, alpha=0.8)

PANELS = (
    ("single_gpu_time", "value_and_grad_time_ms", "Time (ms)"),
    ("single_gpu_ratio", "adjoint_to_forward_ratio", "Ratio"),
    ("single_gpu_peak", "value_and_grad_peak_mib", "MiB"),
    ("single_gpu_peak_diff", "peak_difference_mib", "MiB"),
)

FORWARD_OVERLAY = {
    "single_gpu_time": "forward_time_ms",
    "single_gpu_peak": "forward_peak_mib",
}

REQUIRED = {
    "mode",
    "grid_cells",
    "forward_time_ms",
    "estimated_adjoint_time_ms",
    "value_and_grad_time_ms",
    "adjoint_to_forward_ratio",
    "forward_peak_mib",
    "value_and_grad_peak_mib",
    "peak_difference_mib",
}


def _default_source() -> Path:
    """Return the CSV beside the script, else the conventional bundle CSV."""
    beside_script = SCRIPT_DIR / CSV_NAME
    if beside_script.exists():
        return beside_script
    layout = ExperimentLayout(kind="benchmarks", case="core_ad_scaling")
    if layout.data.is_dir():
        return artifact_from_bundle(layout.root, CSV_NAME)
    return layout.data / CSV_NAME


def _new_panel():
    fig = plt.figure(figsize=(PANEL_W_IN, PANEL_H_IN))
    axis = fig.add_subplot(111)
    fig.subplots_adjust(**MARGINS)
    return fig, axis


def _write_legend(groups, output_dir: Path) -> Path:
    handles = [
        Line2D([], [], color=f"C{i}", marker=MARKERS[i % len(MARKERS)], **GRAD_KW)
        for i in range(len(groups))
    ]
    labels = [str(mode) for mode, _ in groups]

    handles += [
        Line2D([], [], color="black", marker="", ls="-", linewidth=2.0),
        Line2D([], [], color="black", marker="", ls=":", linewidth=2.0),
    ]
    labels += ["Value-and-gradient", "Forward only"]

    fig = plt.figure(figsize=(PANEL_W_IN * 2, 1.5))
    legend = fig.legend(handles, labels, loc="center", ncol=3, frameon=False)
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    output = output_dir / "single_gpu_legend.png"
    fig.savefig(output, bbox_inches=bbox.expanded(1.05, 1.3), facecolor="white")
    plt.close(fig)
    return output


def render(source: Path | None = None, output_dir: Path | None = None) -> list[Path]:
    # A bare call writes its panels beside the script; an explicit source
    # keeps the conventional figure directory of its bundle.
    fallback_output_dir = DEFAULT_OUTPUT_DIR if source is None else None
    source = Path(source) if source is not None else _default_source()
    csv_path = source / "data" / CSV_NAME if source.is_dir() else source
    if not csv_path.exists():
        raise FileNotFoundError(f"No benchmark CSV at {csv_path}")

    output_dir = output_dir or fallback_output_dir or figure_dir_for(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(csv_path)
    missing = REQUIRED.difference(data.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    if data.empty:
        raise ValueError(f"No rows in {csv_path}")

    # Materialise once so mode order, colour and marker stay locked across
    # all four panels and the legend.
    groups = [(mode, group.sort_values("grid_cells")) for mode, group in data.groupby("mode", sort=False)]

    written: list[Path] = []
    for stem, column, ylabel in PANELS:
        fig, axis = _new_panel()
        forward_column = FORWARD_OVERLAY.get(stem)

        for idx, (mode, group) in enumerate(groups):
            x = group["grid_cells"]
            colour = f"C{idx}"
            marker = MARKERS[idx % len(MARKERS)]
            axis.plot(x, group[column], color=colour, marker=marker, **GRAD_KW)
            if forward_column:
                axis.plot(x, group[forward_column], color=colour, marker=marker, **FWD_KW)

        axis.set(xlabel="Grid cells", ylabel=ylabel)
        axis.grid(True, ls="--", alpha=0.4)

        output = output_dir / f"{stem}_main.png"
        fig.savefig(output, facecolor="white")
        plt.close(fig)
        written.append(output)

    written.append(_write_legend(groups, output_dir))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=None,
        help="Benchmark CSV or the bundle holding it (default: CSV beside this script, else the output bundle)",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    for path in render(args.source, args.output_dir):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
