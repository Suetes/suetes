#!/usr/bin/env python3
"""Render convergence plots from a smooth-mode verification summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from suetes.shared.artifacts import artifact_from_bundle, figure_dir_for

plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'font.family': 'sans-serif'
})


LABELS = {
    "q_test": r"Tracer $q$",
    "th_v": r"$\theta_v$",
    "w": r"$w$",
    "pi": r"$\pi$",
    "u": r"$u$",
}
CORE_STYLE = {
    "sisl": {"label": "SISL", "marker": "o", "color": "#0072B2"},
    "split-explicit": {
        "label": "Split-Explicit", "marker": "s", "color": "#D55E00"
    },
}


def render(summary_path: Path, output_dir: Path | None = None) -> list[Path]:
    with summary_path.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    output_dir = output_dir or figure_dir_for(summary_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    for case, core_results in summary["cases"].items():
        fields = list(next(iter(core_results.values()))["self_errors"])
        fig, axes = plt.subplots(
            1, len(fields), figsize=(6.2 * len(fields), 4.8), squeeze=False
        )
        axes = axes[0]
        reference_order = 1 if case == "tracer" else 2
        for axis, field in zip(axes, fields):
            first_values = None
            first_x = None
            for core, result in core_results.items():
                dt = np.asarray(result["dt_s"][:-1], dtype=float)
                errors = np.asarray(result["self_errors"][field], dtype=float)
                style = CORE_STYLE[core]
                axis.loglog(
                    dt, errors, linewidth=2, markersize=7,
                    label=style["label"], marker=style["marker"],
                    color=style["color"],
                )
                if first_values is None:
                    first_values, first_x = errors, dt
            reference = first_values[0] * (first_x / first_x[0]) ** reference_order
            axis.loglog(
                first_x, reference, "k--", alpha=0.7,
                label=f"Order {reference_order}",
            )
            axis.invert_xaxis()
            
            props = dict(boxstyle='square,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none')
            if len(fields) > 1:
                label_text = LABELS.get(field, field)
                axis.text(0.05, 0.95, label_text, transform=axis.transAxes, fontsize=16, verticalalignment='top', bbox=props)
                
            axis.set_xlabel(r"Outer timestep $\Delta t$ [s]")
            if axis == axes[0]:
                axis.set_ylabel(r"Relative $L_2$ difference")
            axis.grid(True, which="both", ls="--", alpha=0.4)
            import matplotlib.ticker as ticker
            axis.xaxis.set_minor_formatter(ticker.NullFormatter())
            ticks = sorted(list(set([dt[0], dt[len(dt)//2], dt[-1]])))
            axis.set_xticks(ticks)
            axis.xaxis.set_major_formatter(ticker.ScalarFormatter())
            axis.tick_params(axis="x", which="both", rotation=0)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="lower center", ncol=len(labels),
            frameon=False, bbox_to_anchor=(0.5, 0.0),
        )
        fig.tight_layout(rect=(0, 0.15, 1, 1))
        output = output_dir / f"{case}_temporal_self_convergence.png"
        fig.savefig(output, dpi=300)
        plt.close(fig)
        outputs.append(output)
        print(f"Saved {output}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Summary, data directory, or bundle")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    summary = artifact_from_bundle(args.source, pattern="summary.json")
    render(summary, args.output_dir)


if __name__ == "__main__":
    main()
