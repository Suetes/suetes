#!/usr/bin/env python3
# Case renderer.
"""Render ERA5-coupled adjoint diagnostics from reduced artifact data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr

from suetes.shared.experiment import figure_dir_for


def render(source: Path, output_dir: Path | None = None) -> list[Path]:
    artifact = source / "data" / "artifact.nc" if source.is_dir() else source
    output_dir = output_dir or figure_dir_for(artifact)
    output_dir.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(artifact) as dataset:
        data = dataset.load()
    outputs = []
    for variable, filename in (
        ("thermal_impact_percentage", "adjoint_impact_percentage.png"),
        ("wind_sensitivity_magnitude", "adjoint_kinetic_sensitivity.png"),
    ):
        fig, axis = plt.subplots(figsize=(8, 6))
        image = axis.pcolormesh(data["x"] / 1000.0, data["y"] / 1000.0, data[variable].values.T, shading="auto")
        axis.set(xlabel="x (km)", ylabel="y (km)", title=variable.replace("_", " "))
        fig.colorbar(image, ax=axis)
        output = output_dir / filename
        fig.savefig(output, dpi=200, bbox_inches="tight")
        plt.close(fig)
        outputs.append(output)
    for variable, filename in (
        ("thermal_sensitivity_cross_section", "adjoint_thermal_cross_section.png"),
        ("vertical_velocity_cross_section", "adjoint_vertical_velocity_cross_section.png"),
    ):
        fig, axis = plt.subplots(figsize=(10, 5))
        image = axis.pcolormesh(
            data["x"] / 1000.0, range(data[variable].shape[1]), data[variable].values.T, shading="auto", cmap="RdBu_r"
        )
        axis.set(xlabel="x (km)", ylabel="Model level", title=variable.replace("_", " "))
        fig.colorbar(image, ax=axis)
        output = output_dir / filename
        fig.savefig(output, dpi=200, bbox_inches="tight")
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
