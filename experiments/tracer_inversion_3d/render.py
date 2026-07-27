#!/usr/bin/env python3
# Case renderer.
"""Render the 3-D tracer inversion artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from suetes.shared.experiment import figure_dir_for


def render(source: Path, output_dir: Path | None = None) -> list[Path]:
    artifact = source / "data" / "artifact.nc" if source.is_dir() else source
    output_dir = output_dir or figure_dir_for(artifact)
    output_dir.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(artifact) as dataset:
        data = dataset.load()
    snapshots = data["tracer_snapshot"]
    count = snapshots.sizes["time"]
    fig, axes = plt.subplots(count, 1, figsize=(12, 3.4 * count), squeeze=False)
    x = np.broadcast_to(data["x"].values[:, None], data["physical_height"].shape)
    for index, axis in enumerate(axes[:, 0]):
        image = axis.contourf(
            x / 1000.0, data["physical_height"] / 1000.0, snapshots.isel(time=index), levels=50, cmap="Blues"
        )
        axis.fill_between(data["x"] / 1000.0, 0.0, data["terrain_height"] / 1000.0, color="black")
        axis.set_title(f"t={float(data['time'][index]) / 60.0:g} min")
    fig.colorbar(image, ax=axes[:, 0], label="Tracer concentration")
    snapshot_path = output_dir / "tracer_snapshots.png"
    fig.savefig(snapshot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    parameters = data["optimization_parameters"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(parameters.sel(parameter="x"), parameters.sel(parameter="z"), "o-")
    axes[0].set(xlabel="Source x", ylabel="Source z", title="Optimization trajectory")
    axes[1].plot(data["target_sensor_profile"], label="target")
    axes[1].plot(data["recovered_sensor_profile"], "--", label="recovered")
    axes[1].set(title="Sensor profile", xlabel="Model level")
    axes[1].legend()
    inversion_path = output_dir / "tracer_source_inversion.png"
    fig.savefig(inversion_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [snapshot_path, inversion_path]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    for path in render(args.source, args.output_dir):
        print(f"Saved {path}")
