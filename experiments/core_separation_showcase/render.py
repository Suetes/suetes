#!/usr/bin/env python3
# Case renderer.
"""Render the core-separation publication diagnostics from an artifact."""

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
    artifact = source / "data" / "artifact.nc" if source.is_dir() else source
    output_dir = output_dir or figure_dir_for(artifact)
    output_dir.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(artifact) as dataset:
        data = dataset.load()
    outputs = []
    limit = float(abs(data["vorticity"]).max())
    for configuration in data["configuration"].values:
        fig, axis = plt.subplots(figsize=(10, 5))
        image = axis.contourf(
            data["x"] / 1000.0,
            data["z"] / 1000.0,
            data["vorticity"].sel(configuration=configuration).values.T,
            levels=np.linspace(-limit, limit, 40),
            cmap="RdBu_r",
            extend="both",
        )
        axis.set(xlabel="Horizontal distance (km)", ylabel="Altitude (km)", title=str(configuration))
        fig.colorbar(image, ax=axis, label="Vorticity (s$^{-1}$)")
        output = output_dir / f"vorticity_{configuration}.png"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)
        outputs.append(output)

    for variable, filename, ylabel in (
        ("enstrophy", "enstrophy_series.png", "Enstrophy"),
        ("kinetic_energy_spectrum", "energy_spectrum.png", "Power density"),
    ):
        fig, axis = plt.subplots(figsize=(9, 5.5))
        coordinate = "time" if variable == "enstrophy" else "wavenumber"
        for configuration in data["configuration"].values:
            values = data[variable].sel(configuration=configuration)
            if coordinate == "wavenumber":
                axis.loglog(data[coordinate], values, label=str(configuration))
            else:
                axis.plot(data[coordinate], values, "o-", label=str(configuration))
        axis.set(xlabel=coordinate, ylabel=ylabel)
        axis.grid(True, which="both", ls="--", alpha=0.5)
        axis.legend()
        output = output_dir / filename
        fig.savefig(output, dpi=300, bbox_inches="tight")
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
