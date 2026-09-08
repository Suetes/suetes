#!/usr/bin/env python3
"""Render self- and cross-convergence from a long-horizon audit bundle."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np

from suetes.shared.experiment import (
    ExperimentLayout,
    artifact_from_bundle,
    figure_dir_for,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
# The audit writes its bundle under the execution name long_horizon_core_audit.py
# uses by default, so a bare run of this renderer finds it without arguments.
DEFAULT_BUNDLE = ExperimentLayout(
    kind="verification",
    case="long_horizon_core_audit",
    execution="long-horizon",
    output_root=REPO_ROOT / "output",
).root

# Sized for a two-across figure_A8 row. Revert to 16 / 18 / 16 if this reads heavy
# against the rest of the paper.
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 20,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'font.family': 'sans-serif'
})

# The two A8 panels share one canvas size. Margins get measured per panel and then
# equalized, so labels do not clip and a plain `-gravity north +append` in
# make_fig.sh leaves the axes level. Do not call tight_layout or
# bbox_inches="tight" on either A8 panel; that gives each its own margins. The
# evolution figure stands alone, so it fits itself freely.
PANEL_SIZE = (6.0, 4.8)

# A8 carries one legend across both panels, so no colour+marker pair may mean two
# things. The cores keep the two hues they use elsewhere in the paper; the time
# slices are held clear of those hues and take their own markers. The old version
# drew times from the default colour cycle, which put a blue circle line in panel
# (a) and another in panel (b).
CORE_STYLES = {
    "sisl": dict(color="#0072B2", marker="o"),
    "split-explicit": dict(color="#D55E00", marker="s"),
}
CORE_LABELS = {"sisl": "SISL", "split-explicit": "Split-Explicit"}
TIME_STYLES = [
    dict(color="#009E73", marker="^"),
    dict(color="#CC79A7", marker="v"),
    dict(color="#332288", marker="D"),
]
# Its own figure, but still kept off the core hues so orange does not read as
# Split-Explicit to anyone flipping between the two figures.
DX_STYLES = [
    dict(color="#E69F00", marker="P"),
    dict(color="#56B4E9", marker="X"),
    dict(color="#999999", marker="*"),
    dict(color="#882255", marker="h"),
    dict(color="#44AA99", marker="8"),
]
REFERENCE_KWARGS = dict(color="black", ls="--", alpha=0.7)


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return [
            {
                **row,
                **{
                    key: float(row[key])
                    for key in ("coarse_dx_m", "fine_dx_m", "time_s", "value")
                },
            }
            for row in csv.DictReader(stream)
        ]


def _value_at(rows: list[dict], time: float, dx: float) -> float:
    return next(
        row["value"] for row in rows
        if np.isclose(row["time_s"], time) and np.isclose(row["coarse_dx_m"], dx)
    )


def _three_ticks(values) -> list[float]:
    return sorted(set([values[0], values[len(values) // 2], values[-1]]))


def _finish(axis, ticks=None) -> None:
    axis.grid(True, which="both", ls="--", alpha=0.4)
    axis.xaxis.set_minor_formatter(ticker.NullFormatter())
    axis.tick_params(axis="x", which="both", rotation=0)
    if ticks is not None:
        axis.set_xticks(ticks)
        axis.xaxis.set_major_formatter(ticker.ScalarFormatter())


def equalize_margins(figures) -> dict[str, float]:
    """Fit each panel, then give them all the tightest margins that suit every one."""
    for fig in figures:
        fig.tight_layout()
    boxes = [fig.axes[0].get_position() for fig in figures]
    margins = dict(
        left=max(box.x0 for box in boxes),
        right=min(box.x1 for box in boxes),
        bottom=max(box.y0 for box in boxes),
        top=min(box.y1 for box in boxes),
    )
    for fig in figures:
        fig.subplots_adjust(**margins)
    return margins


def build_cross_refinement(cross, dx_values, selected_times):
    fig, axis = plt.subplots(figsize=PANEL_SIZE)
    anchor = None
    for style, time in zip(TIME_STYLES, selected_times):
        values = [_value_at(cross, time, dx) for dx in dx_values]
        anchor = anchor if anchor is not None else values[0]
        axis.loglog(dx_values, values, ls="-", lw=2, **style)
    reference = np.asarray(dx_values, dtype=float) ** 2
    reference *= anchor / reference[0]
    axis.loglog(dx_values, reference, lw=2, **REFERENCE_KWARGS)
    axis.set_xlabel(r"Grid spacing $\Delta x$ [m]")
    axis.set_ylabel(r"Relative $L_2$ difference")
    axis.invert_xaxis()
    _finish(axis, _three_ticks(dx_values))
    return fig


def build_self_convergence(self_rows, cores, final_time, dx_self):
    fig, axis = plt.subplots(figsize=PANEL_SIZE)
    for core in cores:
        selected = sorted(
            (
                row for row in self_rows
                if row["core"] == core and np.isclose(row["time_s"], final_time)
            ),
            key=lambda row: row["coarse_dx_m"],
            reverse=True,
        )
        axis.loglog(
            [row["coarse_dx_m"] for row in selected],
            [row["value"] for row in selected],
            ls="-", lw=2, **CORE_STYLES[core],
        )
    axis.set_xlabel(r"Coarse-grid spacing $\Delta x$ [m]")
    axis.set_ylabel(r"Relative $L_2$ difference")
    axis.invert_xaxis()
    _finish(axis, _three_ticks(dx_self))
    return fig


def render_audit_legend(selected_times, cores, output: Path) -> Path:
    handles = [
        Line2D([], [], ls="-", lw=2, label=f"{time:g} s", **style)
        for style, time in zip(TIME_STYLES, selected_times)
    ]
    handles.append(Line2D([], [], lw=2, label="Order 2", **REFERENCE_KWARGS))
    handles.extend(
        Line2D([], [], ls="-", lw=2, label=CORE_LABELS[core], **CORE_STYLES[core])
        for core in cores
    )
    fig = plt.figure(figsize=(PANEL_SIZE[0] * 2, 0.7))
    fig.legend(handles=handles, loc="center", ncol=len(handles), frameon=False)
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return output


def build_error_evolution(cross, dx_values):
    fig, axis = plt.subplots(figsize=PANEL_SIZE)
    for style, dx in zip(DX_STYLES, dx_values):
        selected = sorted(
            (row for row in cross if np.isclose(row["coarse_dx_m"], dx)),
            key=lambda row: row["time_s"],
        )
        axis.plot(
            [row["time_s"] for row in selected],
            [row["value"] for row in selected],
            ls="-", lw=2, **style,
        )
    axis.set_xlabel("Simulation time [s]")
    axis.set_ylabel(r"Relative $L_2$ difference")
    axis.grid(True, ls="--", alpha=0.4)
    axis.xaxis.set_major_locator(ticker.MaxNLocator(5))
    axis.tick_params(axis="x", which="both", rotation=0)
    fig.tight_layout()
    return fig


def render_evolution_legend(dx_values, output: Path) -> Path:
    handles = [
        Line2D([], [], ls="-", lw=2, label=rf"$\Delta x={dx:g}$ m", **style)
        for style, dx in zip(DX_STYLES, dx_values)
    ]
    fig = plt.figure(figsize=(PANEL_SIZE[0], 0.7))
    fig.legend(handles=handles, loc="center", ncol=len(handles), frameon=False)
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return output


def render(metrics_path: Path, output_dir: Path | None = None) -> list[Path]:
    rows = _read(metrics_path)
    output_dir = output_dir or figure_dir_for(metrics_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    cross = [row for row in rows if row["metric"] == "cross_core_relative_l2"]
    self_rows = [row for row in rows if row["metric"] == "self_relative_l2"]
    times = sorted({row["time_s"] for row in cross if row["time_s"] > 0})
    dx_values = sorted({row["coarse_dx_m"] for row in cross}, reverse=True)
    dx_self = sorted({row["coarse_dx_m"] for row in self_rows}, reverse=True)
    selected_times = sorted(set((times[0], times[len(times) // 2], times[-1])))
    cores = [
        core for core in CORE_STYLES
        if any(row["core"] == core for row in self_rows)
    ]
    final_time = times[-1]

    if len(selected_times) > len(TIME_STYLES):
        raise ValueError(
            f"{len(selected_times)} time slices but only {len(TIME_STYLES)} distinct "
            "styles; extend TIME_STYLES or the A8 legend becomes ambiguous"
        )
    if len(dx_values) > len(DX_STYLES):
        raise ValueError(
            f"{len(dx_values)} grid spacings but only {len(DX_STYLES)} distinct "
            "styles; extend DX_STYLES"
        )

    # figure_A8: two panels appended side by side, so their margins must match.
    audit_panels = [
        (
            build_cross_refinement(cross, dx_values, selected_times),
            output_dir / "long_horizon_cross_refinement_main.png",
        ),
        (
            build_self_convergence(self_rows, cores, final_time, dx_self),
            output_dir / "long_horizon_self_convergence_main.png",
        ),
    ]
    equalize_margins([fig for fig, _ in audit_panels])

    outputs = []
    for fig, path in audit_panels:
        fig.savefig(path, dpi=300)
        plt.close(fig)
        outputs.append(path)
    outputs.append(
        render_audit_legend(
            selected_times, cores, output_dir / "long_horizon_audit_legend.png"
        )
    )

    # Its own figure, single panel, single legend.
    evolution = build_error_evolution(cross, dx_values)
    evolution_path = output_dir / "cross_core_error_evolution_main.png"
    evolution.savefig(evolution_path, dpi=300)
    plt.close(evolution)
    outputs.append(evolution_path)
    outputs.append(
        render_evolution_legend(
            dx_values, output_dir / "cross_core_error_evolution_legend.png"
        )
    )

    for path in outputs:
        print(f"Saved {path}")
    return outputs


def _default_source() -> Path:
    """A metrics CSV kept beside this script wins over the standard bundle."""
    local = SCRIPT_DIR / "metrics.csv"
    return local if local.is_file() else DEFAULT_BUNDLE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", nargs="?", type=Path, default=None,
        help="Metrics CSV, data directory, or bundle",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    source = args.source
    output_dir = args.output_dir
    if source is None:
        source = _default_source()
        # Only the argument-free run redirects the figures; an explicit source
        # keeps writing into its own bundle.
        if output_dir is None:
            output_dir = SCRIPT_DIR / "figures"
    metrics = artifact_from_bundle(source, pattern="metrics.csv")
    render(metrics, output_dir)


if __name__ == "__main__":
    main()
