"""Utilities for durable, self-describing, plot-ready experiment artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import argparse
from typing import Tuple
import subprocess

import numpy as np
import xarray as xr


SCHEMA = "suetes-plot-data-v1"
ARTIFACT_KINDS = frozenset({"benchmarks", "experiments", "runs", "verification"})


class ExperimentLayout:
    """Paths for one self-contained model execution.

    The bundle root is ``<output_root>/<kind>/<case>/<execution>``. Numerical
    artifacts and rendered figures are deliberately separated inside the
    bundle while remaining easy to archive together.
    """

    def __init__(
        self,
        *,
        kind: str,
        case: str,
        execution: str = "default",
        output_root: str | Path = "output",
    ) -> None:
        if kind not in ARTIFACT_KINDS:
            choices = ", ".join(sorted(ARTIFACT_KINDS))
            raise ValueError(f"Unknown artifact kind {kind!r}; expected one of {choices}")
        for label, value in (("case", case), ("execution", execution)):
            path = Path(value)
            if not value or path.is_absolute() or len(path.parts) != 1 or value in {".", ".."}:
                raise ValueError(f"{label} must be a single non-empty path component")
        self.kind = kind
        self.case = case
        self.execution = execution
        self.output_root = Path(output_root)

    @property
    def root(self) -> Path:
        return self.output_root / self.kind / self.case / self.execution

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    def create(self) -> "ExperimentLayout":
        self.data.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)
        return self


def resolve_data_dir(
    *,
    kind: str,
    case: str,
    execution: str = "default",
    output_root: str | Path = "output",
    output_dir: str | Path | None = None,
) -> Path:
    """Return an explicit legacy directory or create a standard data directory."""
    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return ExperimentLayout(
        kind=kind,
        case=case,
        execution=execution,
        output_root=output_root,
    ).create().data


def figure_dir_for(artifact: str | Path) -> Path:
    """Return the conventional figure directory for an artifact path."""
    artifact = Path(artifact)
    if artifact.parent.name == "data":
        return artifact.parent.parent / "figures"
    return artifact.parent


def artifact_from_bundle(source: str | Path, pattern: str = "artifact.nc") -> Path:
    """Resolve one artifact from a bundle, data directory, or explicit file."""
    source = Path(source)
    if source.is_file():
        return source
    data_dir = source / "data" if (source / "data").is_dir() else source
    matches = sorted(data_dir.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {pattern!r} artifact in {data_dir}, "
            f"found {len(matches)}"
        )
    return matches[0]


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
    def netcdf_attribute(value):
        # NetCDF4 attributes do not support booleans. Store them as 0/1 while
        # preserving ordinary numeric and string attributes without conversion.
        if isinstance(value, (bool, np.bool_)):
            return int(value)
        if isinstance(value, (str, int, float, np.number)):
            return value
        return json.dumps(value)

    dataset.attrs = {
        key: netcdf_attribute(value)
        for key, value in dataset.attrs.items()
    }
    dataset.attrs.update({
        key: netcdf_attribute(value)
        for key, value in provenance.items()
    })
    encoding = {
        name: {"zlib": True, "complevel": 4, "shuffle": True}
        for name, variable in dataset.data_vars.items()
        if variable.ndim and np.issubdtype(variable.dtype, np.number)
    }
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        dataset.to_netcdf(
            temporary_path, engine="netcdf4", encoding=encoding
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    with path.with_suffix(".json").open("w") as stream:
        json.dump(provenance, stream, indent=2)
    return path


def add_experiment_args(
    parser: argparse.ArgumentParser,
    *,
    default_output_root: str | Path = Path("output"),
) -> None:
    """Add standard output-directory arguments to an argument parser."""
    parser.add_argument(
        "--output-root", type=Path, default=Path(default_output_root),
        help="Root directory for the standard experiment bundle",
    )
    parser.add_argument(
        "--name", default="default",
        help="Execution name for the standard experiment bundle",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Explicit legacy directory to bypass the standard bundle layout",
    )


def setup_experiment_directories(
    args: argparse.Namespace, *, kind: str, case: str
) -> Tuple[Path, Path]:
    """Return (data_dir, figure_dir) based on standard CLI arguments."""
    if getattr(args, "output_dir", None) is not None:
        path = Path(args.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path, path
    
    layout = ExperimentLayout(
        kind=kind,
        case=case,
        execution=getattr(args, "name", "default"),
        output_root=getattr(args, "output_root", Path("output")),
    ).create()
    return layout.data, layout.figures
