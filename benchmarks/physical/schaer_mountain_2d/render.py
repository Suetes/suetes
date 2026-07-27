#!/usr/bin/env python3
"""Render the 2-D Schär mountain benchmark artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from suetes.shared.artifacts import figure_dir_for


def render(source: Path, output_dir: Path | None = None) -> Path:
    artifact = source / "data" / "artifact.nc" if source.is_dir() else source
    output_dir = output_dir or figure_dir_for(artifact)
    output_dir.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(artifact) as dataset:
        data = dataset.load()
    x = np.broadcast_to(data["x"].values[:, None], data["physical_height"].shape)
    fig, axis = plt.subplots(figsize=(12, 6))
    image = axis.contourf(
        x / 1000.0,
        data["physical_height"] / 1000.0,
        data["vertical_velocity"],
        levels=np.linspace(-2.0, 2.0, 41),
        cmap="RdBu_r",
        extend="both",
    )
    axis.fill_between(data["x"] / 1000.0, 0.0, data["terrain_height"] / 1000.0, color="black")
    axis.set(xlabel="Distance (km)", ylabel="Altitude (km)", title="Schär mountain wave")
    fig.colorbar(image, ax=axis, label="Vertical velocity (m/s)")
    output = output_dir / "schaer_mountain_2d.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(f"Saved {render(args.artifact, args.output_dir)}")
