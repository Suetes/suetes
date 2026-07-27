#!/usr/bin/env python3
# Case renderer.
"""Render synthetic or ERA5 4D-Var reconstruction artifacts."""

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
    if "theta_anomaly" in data:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for axis, state in zip(axes, data["state"].values):
            field = data["theta_anomaly"].sel(state=state)
            image = axis.contourf(data["x"] / 1000.0, data["z"], field.values.T, levels=21, cmap="RdBu_r")
            axis.set_title(str(state))
        fig.colorbar(image, ax=axes, label="Potential-temperature anomaly (K)")
        filename = "4dvar_reconstruction.png"
    else:
        variables = data["variable"].values
        states = data["state"].values
        fig, axes = plt.subplots(len(variables), len(states), figsize=(18, 12))
        for row, variable in enumerate(variables):
            for column, state in enumerate(states):
                image = axes[row, column].imshow(
                    data["surface_field"].sel(state=state, variable=variable).values.T, origin="lower"
                )
                axes[row, column].set_title(f"{state} ({variable})")
                fig.colorbar(image, ax=axes[row, column])
        filename = "multivariate_4dvar_recovery.png"
    output = output_dir / filename
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(f"Saved {render(args.source, args.output_dir)}")
