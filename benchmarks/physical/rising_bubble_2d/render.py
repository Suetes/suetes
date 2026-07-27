#!/usr/bin/env python3
"""Render the 2-D rising-bubble benchmark artifact."""

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
        field = dataset["theta_perturbation"].load()
    fig, axis = plt.subplots(figsize=(12, 5))
    image = axis.contourf(field["x"] / 1000.0, field["z"] / 1000.0, field.values.T, levels=20, cmap="RdBu_r")
    axis.set(xlabel="x (km)", ylabel="z (km)", title="SISL rising bubble")
    fig.colorbar(image, ax=axis, label="Temperature perturbation (K)")
    output = output_dir / "rising_bubble_2d.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(f"Saved {render(args.artifact, args.output_dir)}")
