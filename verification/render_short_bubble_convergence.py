#!/usr/bin/env python3
"""Render the 24 s rising-bubble self- and cross-convergence diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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

from convergence_plotting import save_four_panel_convergence
from suetes.shared.experiment import artifact_from_bundle, figure_dir_for


FIELDS = ("u", "w", "pi", "th_v")
SELF_PATTERN = re.compile(
    r"Res\s+(?P<coarse>[\d.]+)\s+->\s+(?P<fine>[\d.]+)m"
    r"\s+\|\s+err_u:\s+(?P<u>[\deE+.-]+)"
    r"\s+\|\s+err_w:\s+(?P<w>[\deE+.-]+)"
    r"\s+\|\s+err_pi:\s+(?P<pi>[\deE+.-]+)"
    r"\s+\|\s+err_th:\s+(?P<th_v>[\deE+.-]+)"
)
CROSS_PATTERN = re.compile(
    r"dx=\s*(?P<dx>[\d.]+)m\s+\|\s+u=(?P<u>[\deE+.-]+)"
    r"\s+\|\s+w=(?P<w>[\deE+.-]+)"
    r"\s+\|\s+pi=(?P<pi>[\deE+.-]+)"
    r"\s+\|\s+th_v=(?P<th_v>[\deE+.-]+)"
)


def parse_log(path: Path) -> tuple[np.ndarray, dict, dict, np.ndarray, dict]:
    text = path.read_text(encoding="utf-8")
    self_matches = list(SELF_PATTERN.finditer(text))
    if len(self_matches) != 6:
        raise ValueError(
            f"Expected three self-error rows per core in {path}, "
            f"found {len(self_matches)}"
        )
    halves = (self_matches[:3], self_matches[3:])
    dx = np.asarray([float(match["coarse"]) for match in halves[0]])
    errors = []
    for matches in halves:
        errors.append({
            field: [float(match[field]) for match in matches]
            for field in FIELDS
        })

    cross_matches = list(CROSS_PATTERN.finditer(text))
    if len(cross_matches) != 4:
        raise ValueError(
            f"Expected four cross-core rows in {path}, found {len(cross_matches)}"
        )
    cross_dx = np.asarray([float(match["dx"]) for match in cross_matches])
    cross = {
        field: [float(match[field]) for match in cross_matches]
        for field in FIELDS
    }
    return dx, errors[0], errors[1], cross_dx, cross


def parse_summary(path: Path) -> tuple[
    np.ndarray, dict, dict, np.ndarray, dict, float
]:
    with path.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    resolutions = np.asarray(summary["resolutions_m"], dtype=float)
    return (
        resolutions[:-1],
        summary["cores"]["sisl"]["self_errors"],
        summary["cores"]["split-explicit"]["self_errors"],
        resolutions,
        summary["cross_core"]["differences"],
        float(summary["t_end_s"]),
    )


def render(source: Path, output_dir: Path | None = None) -> list[Path]:
    if source.is_dir():
        source = artifact_from_bundle(source, pattern="summary.json")
    if source.suffix == ".json":
        dx, sisl, split, cross_dx, cross, t_end = parse_summary(source)
    else:
        dx, sisl, split, cross_dx, cross = parse_log(source)
        t_end = 24.0
    output_dir = output_dir or figure_dir_for(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    ylabels = {
        "u": r"$\|\Delta u\|_2$ [m s$^{-1}$]",
        "w": r"$\|\Delta w\|_2$ [m s$^{-1}$]",
        "pi": r"$\|\Delta\pi\|_2$ [1]",
        "th_v": r"$\|\Delta\theta_v\|_2$ [K]",
    }
    time_label = f"{t_end:g}"
    self_path = output_dir / f"bubble_t{time_label}_self_convergence.png"
    save_four_panel_convergence(
        dx,
        [("SISL", sisl), ("Split-Explicit", split)],
        ylabels,
        r"Coarse-grid spacing $\Delta x$ [m]",
        rf"Rising bubble self-convergence at $t={time_label}$ s",
        self_path,
        reference_order=2,
    )
    cross_path = output_dir / f"bubble_t{time_label}_cross_core_convergence.png"
    save_four_panel_convergence(
        cross_dx,
        [("SISL minus Split-Explicit", cross)],
        ylabels,
        r"Grid spacing $\Delta x$ [m]",
        rf"Rising bubble cross-core convergence at $t={time_label}$ s",
        cross_path,
        reference_order=2,
    )

    # A compact manuscript-friendly theta-only figure.
    fig, axis = plt.subplots(figsize=(6.7, 5.0))
    axis.loglog(dx, sisl["th_v"], "o-", lw=2, label="SISL")
    axis.loglog(dx, split["th_v"], "s-", lw=2, label="Split-Explicit")
    reference = max(sisl["th_v"][0], split["th_v"][0]) * (
        dx / dx[0]
    ) ** 2
    axis.loglog(dx, reference, "k--", alpha=0.7, label="Order 2")
    axis.invert_xaxis()
    axis.set_xlabel(r"Coarse-grid spacing $\Delta x$ [m]")
    axis.set_ylabel(r"$\|\Delta\theta_v\|_2$ [K]")
    axis.grid(True, which="both", ls="--", alpha=0.4)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=len(labels))
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    theta_path = output_dir / f"bubble_t{time_label}_theta_self_convergence.png"
    fig.savefig(theta_path, dpi=300)
    plt.close(fig)

    outputs = [self_path, cross_path, theta_path]
    for output in outputs:
        print(f"Saved {output}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", nargs="?", type=Path,
        default=Path(
            "output/verification/rising_bubble_core_convergence/t24"
        ),
        help="Standard bundle, summary JSON, or legacy bubble.log",
    )
    parser.add_argument(
        "--output-dir", type=Path,
    )
    args = parser.parse_args()
    source = args.source
    standard_summary = source / "data" / "summary.json"
    output_dir = args.output_dir
    if (
        source.name == "t24"
        and (not source.exists() or not standard_summary.exists())
    ):
        source = Path("output/verification/bubble.log")
        if output_dir is None:
            output_dir = Path(
                "output/verification/rising_bubble_core_convergence/"
                "t24/figures"
            )
    render(source, output_dir)


if __name__ == "__main__":
    main()
