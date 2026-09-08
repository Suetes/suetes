#!/usr/bin/env python3
"""Render dual-core CPU--GPU scaling results as separate label-free panels.

One PNG per workload, both cores on each, plus a shared legend strip:

    panel_a_forward.png             forward, both cores
    panel_b_value_and_gradient.png  value + gradient, both cores
    legend_cpu_gpu.png              shared four-entry legend strip

Colour carries the platform (CPU blue, GPU red); marker and dash pattern carry
the core (split-explicit solid circles, SISL dashed squares). Panels and legend
are built from the same two style tables, so they cannot drift apart.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from suetes.shared.experiment import figure_dir_for

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
        "axes.linewidth": 0.9,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
CSV_NAME = "cpu_gpu_short_scaling.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "figures"
# run.py writes a flat directory for this case: no <execution>/data level.
BENCHMARK_OUTPUT = REPO_ROOT / "output" / "benchmarks" / "cpu_gpu_short_scaling"

# Both panels share one canvas and one set of margins, so the axes land on
# identical pixels and a plain north append in make_fig.sh keeps them level.
# Do not -trim the outputs; that padding is what makes the row line up.
FIG_SIZE = (3.4, 3.0)
MARGINS = dict(left=0.21, right=0.97, bottom=0.155, top=0.90)
DPI = 300
TIME_COLUMN = "median_s"  # switch to "min_s" for best-of runs

LEGEND_SIZE = (7.0, 0.60)  # inches, wide enough for four entries on one row
# Panel axis labels are 11 pt at the same dpi and nothing is rescaled during
# assembly, so this is a direct comparison. Raise it to make the legend louder.
LEGEND_FONTSIZE = 13
LEGEND_MARKERSIZE = 6.0
LEGEND_LINEWIDTH = 2.0
LEGEND_NCOL = 4  # drop to 2 for a two-row block under the figure

PLATFORM_STYLE = {
    "cpu": dict(color="#1f77b4", label="CPU"),
    "gpu": dict(color="#e8000b", label="GPU"),
}

CORE_STYLE = {
    "split-explicit": dict(marker="o", linestyle="-", label="split-explicit"),
    "sisl": dict(marker="s", linestyle="--", label="SISL"),
}

# Draw order inside a panel, and left-to-right order in the legend.
SERIES = [(platform, core) for platform in PLATFORM_STYLE for core in CORE_STYLE]

# (workload, output stem)
PANELS = (
    ("forward", "panel_a_forward"),
    ("value_and_gradient", "panel_b_value_and_gradient"),
)

LEGEND_STEM = "legend_cpu_gpu"

REQUIRED = {"platform", "core", "workload", "grid_cells", TIME_COLUMN}


def _default_source() -> Path:
    """Return the CSV beside the script, else the one run.py writes."""
    beside_script = SCRIPT_DIR / CSV_NAME
    if beside_script.exists():
        return beside_script
    return BENCHMARK_OUTPUT / CSV_NAME


def _line_kwargs(platform: str, core: str) -> dict:
    """Single source of truth for curve appearance, used by panels and legend."""
    return dict(
        color=PLATFORM_STYLE[platform]["color"],
        marker=CORE_STYLE[core]["marker"],
        linestyle=CORE_STYLE[core]["linestyle"],
    )


def _series_label(platform: str, core: str) -> str:
    return f"{PLATFORM_STYLE[platform]['label']}, {CORE_STYLE[core]['label']}"


def _write_panel(data: pd.DataFrame, workload: str, output_path: Path) -> Path:
    fig = plt.figure(figsize=FIG_SIZE)
    axis = fig.add_subplot(111)
    fig.subplots_adjust(**MARGINS)

    for platform, core in SERIES:
        selected = data[
            (data["workload"] == workload) & (data["platform"] == platform) & (data["core"] == core)
        ].sort_values("grid_cells")
        if selected.empty:
            print(f"Warning: no rows for {workload}/{platform}/{core}")
            continue
        axis.plot(
            selected["grid_cells"],
            selected["time_ms"],
            linewidth=1.6,
            markersize=4.5,
            **_line_kwargs(platform, core),
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"Grid cells $N^3$")
    axis.set_ylabel("Execution time (ms)")
    axis.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.45)
    axis.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.30)
    axis.set_axisbelow(True)
    axis.tick_params(which="both", direction="out")

    # No bbox_inches here on purpose. Tight bounding boxes vary with tick
    # label width and would knock the panels out of alignment in the row.
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)
    return output_path


def _write_legend(output_dir: Path) -> Path:
    handles = [
        Line2D(
            [],
            [],
            markersize=LEGEND_MARKERSIZE,
            linewidth=LEGEND_LINEWIDTH,
            label=_series_label(platform, core),
            **_line_kwargs(platform, core),
        )
        for platform, core in SERIES
    ]
    fig = plt.figure(figsize=LEGEND_SIZE)
    fig.legend(
        handles=handles,
        loc="center",
        ncol=LEGEND_NCOL,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=2.6,
        columnspacing=2.0,
    )
    output = output_dir / f"{LEGEND_STEM}.png"
    fig.savefig(output, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return output


def render(source: Path | None = None, output_dir: Path | None = None) -> list[Path]:
    source = Path(source) if source is not None else _default_source()
    csv_path = source / CSV_NAME if source.is_dir() else source
    if not csv_path.exists():
        raise FileNotFoundError(f"No benchmark CSV at {csv_path}; run run.py first")

    # An explicit source keeps the flat benchmark layout; a bare run writes its
    # panels into the figure directory beside the script.
    output_dir = output_dir or figure_dir_for(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(csv_path)
    missing = REQUIRED.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")
    if data.empty:
        raise ValueError(f"No rows in {csv_path}")
    data["time_ms"] = data[TIME_COLUMN] * 1000.0
    for column in ("core", "workload", "platform"):
        data[column] = data[column].str.strip().str.lower()

    written = [_write_panel(data, workload, output_dir / f"{stem}.png") for workload, stem in PANELS]
    written.append(_write_legend(output_dir))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir
    if output_dir is None and args.source is None:
        output_dir = DEFAULT_OUTPUT_DIR
    for path in render(args.source, output_dir):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
