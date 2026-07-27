#!/usr/bin/env python3
# Case renderer.
"""Render Schär optimal-perturbation diagnostics from an artifact."""

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
    outputs = []
    for variable, filename in (
        ("initial_perturbation", "schaer_perturbation_validation.png"),
        ("vertical_velocity", "schaer_forward_validation.png"),
    ):
        field = data[variable]
        vertical = field.dims[-1]
        fig, axes = plt.subplots(2, 2, figsize=(20, 12), squeeze=False)
        limit = float(abs(field).max())
        for axis, configuration in zip(axes.flat, data["configuration"].values):
            image = axis.contourf(
                data["x"] / 1000.0,
                np.arange(field.sizes[vertical]),
                field.sel(configuration=configuration).values.T,
                levels=np.linspace(-limit, limit, 41),
                cmap="RdBu_r",
                extend="both",
            )
            axis.set_title(str(configuration))
        fig.colorbar(image, ax=list(axes.flat))
        output = output_dir / filename
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
