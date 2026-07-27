#!/usr/bin/env python3
"""Render stretched-SLEVE benchmark artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from suetes.shared.artifacts import figure_dir_for


def render(source: Path, output_dir: Path | None = None) -> list[Path]:
    if source.is_dir() and (source / "data").is_dir():
        artifacts = sorted((source / "data").glob("*.nc"))
    elif source.is_dir():
        artifacts = sorted(source.glob("*.nc"))
    else:
        artifacts = [source]
    if not artifacts:
        raise FileNotFoundError(f"No NetCDF artifacts found below {source}")
    output_dir = output_dir or figure_dir_for(artifacts[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for artifact in artifacts:
        with xr.open_dataset(artifact) as dataset:
            data = dataset.load()
        x = np.broadcast_to(data["x"].values[:, None], data["physical_height"].shape)
        variable = str(data.attrs["variable"])
        fig, axis = plt.subplots(figsize=(14, 6))
        image = axis.contourf(
            x / 1000.0, data["physical_height"] / 1000.0, data["field"], levels=50, cmap="RdBu_r", extend="both"
        )
        axis.fill_between(data["x"] / 1000.0, 0.0, data["terrain_height"] / 1000.0, color="black")
        axis.set(
            xlabel="Distance (km)",
            ylabel="Altitude (km)",
            title=f"{data.attrs['experiment']} ({data.attrs['coordinate']})",
            ylim=(0.0, 10.0),
        )
        fig.colorbar(image, ax=axis, label=variable)
        output = output_dir / f"{artifact.stem}.png"
        fig.savefig(output, dpi=150, bbox_inches="tight")
        plt.close(fig)
        outputs.append(output)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    for path in render(args.source, args.output_dir):
        print(f"Saved {path}")
