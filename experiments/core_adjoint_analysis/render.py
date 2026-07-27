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

from suetes.shared.artifacts import figure_dir_for

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


def render(source: Path, output_dir: Path | None = None) -> Path:
    artifact = source / "data" / "artifact.nc" if source.is_dir() else source
    output_dir = output_dir or figure_dir_for(artifact)
    output_dir.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(artifact) as dataset:
        data = dataset.load()
    gradient = data["adjoint_gradient"]
    limit = float(abs(gradient).max())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    panel_titles = {"split-explicit": "(a) Split-Explicit", "sisl": "(b) SISL"}
    core_colors = {"split-explicit": "#D55E00", "sisl": "#0072B2"}
    for axis, core in zip(axes[:2], data["core"].values):
        core_name = str(core)
        image = axis.pcolormesh(
            data["x"] / 1000.0,
            data["z"] / 1000.0,
            gradient.sel(core=core).values.T,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            shading="auto",
        )
        props = dict(boxstyle="square,pad=0.3", facecolor="white", alpha=0.9, edgecolor="none")
        axis.text(
            0.05,
            0.95,
            panel_titles.get(core_name, core_name),
            transform=axis.transAxes,
            fontsize=16,
            verticalalignment="top",
            bbox=props,
        )
        axis.set(xlabel="x (km)", ylabel="z (km)")
    z_index = int(np.argmin(abs(data["z"].values - data.attrs["profile_height_m"])))
    for core in data["core"].values:
        core_name = str(core)
        axes[2].plot(
            data["x"] / 1000.0,
            gradient.sel(core=core).isel(z=z_index),
            color=core_colors.get(core_name),
            label="Split-Explicit" if core_name == "split-explicit" else "SISL",
        )
    axes[2].text(
        0.05,
        0.95,
        "(c) Sensitivity profile",
        transform=axes[2].transAxes,
        fontsize=16,
        verticalalignment="top",
        bbox=props,
    )
    axes[2].set(xlabel="x (km)", ylabel="Adjoint sensitivity")
    axes[2].grid(True, ls="--", alpha=0.4)

    # Attach legend to axes[2] with bbox_to_anchor so constrained_layout makes space for it.
    axes[2].legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    fig.colorbar(image, ax=axes[:2], label="Adjoint gradient")

    output = output_dir / "adjoint_gradient_field_comparison.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(f"Saved {render(args.source, args.output_dir)}")
