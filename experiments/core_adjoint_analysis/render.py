#!/usr/bin/env python3
# Case renderer.
"""Render dual-core adjoint-gradient diagnostics from an artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
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

CORE_COLORS = {"split-explicit": "#D55E00", "sisl": "#0072B2"}
CORE_LABELS = {"split-explicit": "Split-Explicit", "sisl": "SISL"}

# Panels and the colorbar strip share one canvas height. Margins are measured
# per panel and then equalized, so nothing clips and a plain
# `-gravity north +append` in make_fig.sh still leaves the axes and bar level.
# Do not call tight_layout or bbox_inches="tight" on a single panel: that gives
# each one its own margins and breaks the alignment.
PANEL_SIZE = (6.0, 5.5)
CBAR_SIZE = (2.4, 5.5)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_FIGURE_DIR = SCRIPT_DIR / "figures"


def default_artifact() -> Path:
    """Prefer an artifact beside this script, else the conventional bundle."""
    local = SCRIPT_DIR / "artifact.nc"
    if local.is_file():
        return local
    layout = ExperimentLayout(
        kind="experiments",
        case="core_adjoint_analysis",
        output_root=REPO_ROOT / "output",
    )
    return artifact_from_bundle(layout.root)


def stem_for(core_name: str) -> str:
    return core_name.replace("-", "_")


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


def build_field(data, gradient, core, limit: float):
    fig, axis = plt.subplots(figsize=PANEL_SIZE)
    axis.pcolormesh(
        data["x"] / 1000.0,
        data["z"] / 1000.0,
        gradient.sel(core=core).values.T,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        shading="auto",
    )
    axis.set(xlabel="x (km)", ylabel="z (km)")
    return fig


def build_profile(data, gradient):
    fig, axis = plt.subplots(figsize=PANEL_SIZE)
    z_index = int(np.argmin(abs(data["z"].values - data.attrs["profile_height_m"])))
    for core in data["core"].values:
        core_name = str(core)
        axis.plot(
            data["x"] / 1000.0,
            gradient.sel(core=core).isel(z=z_index),
            color=CORE_COLORS.get(core_name),
        )
    axis.set(xlabel="x (km)", ylabel="Adjoint sensitivity")
    axis.grid(True, ls="--", alpha=0.4)
    return fig


def render_colorbar(limit: float, margins: dict[str, float], output: Path) -> Path:
    fig = plt.figure(figsize=CBAR_SIZE)
    cax = fig.add_axes(
        [0.06, margins["bottom"], 0.12, margins["top"] - margins["bottom"]]
    )
    mappable = ScalarMappable(norm=Normalize(vmin=-limit, vmax=limit), cmap="RdBu_r")
    fig.colorbar(mappable, cax=cax, label="Adjoint gradient")
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def render_legend(data, output: Path) -> Path:
    handles = [
        Line2D(
            [],
            [],
            color=CORE_COLORS.get(str(core)),
            label=CORE_LABELS.get(str(core), str(core)),
        )
        for core in data["core"].values
    ]
    fig = plt.figure(figsize=(PANEL_SIZE[0], 0.6))
    fig.legend(handles=handles, loc="center", ncol=len(handles), frameon=False)
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return output


def render(source: Path | None = None, output_dir: Path | None = None) -> list[Path]:
    artifact = artifact_from_bundle(source) if source is not None else default_artifact()
    # An explicit source keeps its figures beside its own bundle; a bare run
    # collects the panels next to this script.
    if output_dir is None:
        output_dir = figure_dir_for(artifact) if source is not None else DEFAULT_FIGURE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(artifact) as dataset:
        data = dataset.load()
    gradient = data["adjoint_gradient"]
    limit = float(abs(gradient).max())

    panels = [
        (
            build_field(data, gradient, core, limit),
            output_dir / f"adjoint_gradient_field_{stem_for(str(core))}_main.png",
        )
        for core in data["core"].values
    ]
    panels.append(
        (
            build_profile(data, gradient),
            output_dir / "adjoint_gradient_profile_main.png",
        )
    )
    margins = equalize_margins([fig for fig, _ in panels])

    outputs = []
    for fig, path in panels:
        fig.savefig(path, dpi=300)
        plt.close(fig)
        outputs.append(path)
    outputs.append(
        render_colorbar(limit, margins, output_dir / "adjoint_gradient_field_cbar.png")
    )
    outputs.append(
        render_legend(data, output_dir / "adjoint_gradient_field_legend.png")
    )
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    for path in render(args.source, args.output_dir):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
