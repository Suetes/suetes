"""Backward-compatible imports for the renamed experiment utilities.

New code should import from :mod:`suetes.shared.experiment`.
"""

from suetes.shared.experiment import (
    ARTIFACT_KINDS,
    SCHEMA,
    ExperimentLayout,
    add_experiment_args,
    artifact_from_bundle,
    figure_dir_for,
    resolve_data_dir,
    save_plot_dataset,
    setup_experiment_directories,
)

ArtifactLayout = ExperimentLayout

__all__ = [
    "ARTIFACT_KINDS",
    "SCHEMA",
    "ArtifactLayout",
    "ExperimentLayout",
    "add_experiment_args",
    "artifact_from_bundle",
    "figure_dir_for",
    "resolve_data_dir",
    "save_plot_dataset",
    "setup_experiment_directories",
]
