#!/usr/bin/env python3
# Case renderer.
"""Render the dual-core quantitative comparison artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr

from suetes.shared.experiment import figure_dir_for


def render(source: Path, output_dir: Path | None = None) -> Path:
    artifact = source / "data" / "artifact.nc" if source.is_dir() else source
    output_dir = output_dir or figure_dir_for(artifact)
    output_dir.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(artifact) as dataset:
        data = dataset.load()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for core in data["core"].values:
        label = str(core)
        axes[0].plot(data["time"], data["enstrophy"].sel(core=core), "o-", label=label)
        axes[1].plot(data["time"], data["maximum_vertical_velocity"].sel(core=core), "o-", label=label)
        axes[2].loglog(data["wavenumber"], data["vertical_velocity_spectrum"].sel(core=core), label=label)
    labels = (
        ("Enstrophy evolution", "Enstrophy"),
        ("Peak updraft", "Vertical velocity (m/s)"),
        ("Vertical-velocity spectrum", "Power"),
    )
    for axis, (title, ylabel) in zip(axes, labels):
        axis.set(title=title, ylabel=ylabel)
        axis.grid(True, which="both", ls="--", alpha=0.5)
        axis.legend()
    axes[0].set_xlabel("Time (s)")
    axes[1].set_xlabel("Time (s)")
    axes[2].set_xlabel("Horizontal wavenumber (rad/m)")
    output = output_dir / "dual_core_quantitative_comparison.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(f"Saved {render(args.source, args.output_dir)}")
