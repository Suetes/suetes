"""Utilities for durable, self-describing, plot-ready experiment artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import xarray as xr


SCHEMA = "suetes-plot-data-v1"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def save_plot_dataset(
    dataset: xr.Dataset,
    path: str | Path,
    *,
    experiment: str,
    metadata: dict | None = None,
) -> Path:
    """Write a compressed NetCDF artifact plus a human-readable JSON sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata or {})
    provenance = {
        "artifact_schema": SCHEMA,
        "experiment": experiment,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        **metadata,
    }
    dataset = dataset.copy()
    dataset.attrs.update({
        key: value if isinstance(value, (str, int, float, np.number)) else json.dumps(value)
        for key, value in provenance.items()
    })
    encoding = {
        name: {"zlib": True, "complevel": 4, "shuffle": True}
        for name, variable in dataset.data_vars.items()
        if variable.ndim and np.issubdtype(variable.dtype, np.number)
    }
    dataset.to_netcdf(path, engine="netcdf4", encoding=encoding)
    with path.with_suffix(".json").open("w") as stream:
        json.dump(provenance, stream, indent=2)
    return path
