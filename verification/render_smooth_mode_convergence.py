#!/usr/bin/env python3
"""Render convergence plots from a smooth-mode verification summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from suetes.shared.artifacts import artifact_from_bundle, figure_dir_for


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
                axis.text(
                    0.04, 0.08 + 0.08 * list(core_results).index(core),
                    f"{style['label']}: finest order "
                    f"{result['orders'][field][-1]:.2f}",
                    transform=axis.transAxes, color=style["color"], fontsize=9,
                )
                if first_values is None:
                    first_values, first_x = errors, dt
            reference = first_values[0] * (first_x / first_x[0]) ** reference_order
            axis.loglog(
                first_x, reference, "k--", alpha=0.7,
                label=f"Order {reference_order}",
            )
            axis.invert_xaxis()
            axis.set_title(LABELS.get(field, field))
            axis.set_xlabel(r"Outer timestep $\Delta t$ (s)")
            axis.set_ylabel("Successive-refinement relative $L_2$ error")
            axis.grid(True, which="both", ls="--", alpha=0.4)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", ncol=len(labels),
            frameon=False, bbox_to_anchor=(0.5, 1.02),
        )
        fig.suptitle(
            f"{case.replace('_', ' ').title()} temporal self-convergence",
            y=1.08,
        )
        fig.tight_layout()
        output = output_dir / f"{case}_temporal_self_convergence.png"
        fig.savefig(output, dpi=300, bbox_inches="tight")
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
