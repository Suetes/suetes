import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from suetes.shared.experiment import (
    ExperimentLayout,
    artifact_from_bundle,
    figure_dir_for,
    resolve_data_dir,
    save_plot_dataset,
)


def test_artifact_layout_creates_bundle_directories(tmp_path: Path) -> None:
    layout = ExperimentLayout(
        kind="benchmarks",
        case="rising_bubble_3d",
        execution="smoke",
        output_root=tmp_path,
    ).create()

    assert layout.root == tmp_path / "benchmarks" / "rising_bubble_3d" / "smoke"
    assert layout.data.is_dir()
    assert layout.figures.is_dir()


@pytest.mark.parametrize("value", ["", ".", "..", "nested/name"])
def test_artifact_layout_rejects_unsafe_components(value: str) -> None:
    with pytest.raises(ValueError):
        ExperimentLayout(kind="benchmarks", case=value)


def test_resolve_data_dir_uses_bundle_or_explicit_legacy_path(
    tmp_path: Path,
) -> None:
    standard = resolve_data_dir(
        kind="experiments",
        case="example",
        execution="paper",
        output_root=tmp_path,
    )
    assert standard == tmp_path / "experiments" / "example" / "paper" / "data"
    assert figure_dir_for(standard / "artifact.nc") == (
        tmp_path / "experiments" / "example" / "paper" / "figures"
    )

    explicit = resolve_data_dir(
        kind="experiments",
        case="ignored",
        output_dir=tmp_path / "legacy",
    )
    assert explicit == tmp_path / "legacy"
    assert figure_dir_for(explicit / "artifact.nc") == explicit


def test_artifact_from_bundle_resolves_data_file(tmp_path: Path) -> None:
    layout = ExperimentLayout(
        kind="experiments", case="example", output_root=tmp_path
    ).create()
    artifact = layout.data / "artifact.nc"
    artifact.touch()

    assert artifact_from_bundle(layout.root) == artifact
    assert artifact_from_bundle(layout.data) == artifact
    assert artifact_from_bundle(artifact) == artifact


def test_save_plot_dataset_accepts_boolean_attributes(tmp_path: Path) -> None:
    artifact = save_plot_dataset(
        xr.Dataset(
            {"value": ("x", [1.0, 2.0])},
            attrs={"periodic_x": True},
        ),
        tmp_path / "artifact.nc",
        experiment="boolean_attribute_test",
    )

    with xr.open_dataset(artifact) as dataset:
        assert dataset.attrs["periodic_x"] == 1
        np.testing.assert_allclose(dataset["value"], [1.0, 2.0])
    assert not (tmp_path / ".artifact.nc.tmp").exists()


def test_rising_bubble_renderer_consumes_artifact(tmp_path: Path) -> None:
    layout = ExperimentLayout(
        kind="benchmarks",
        case="rising_bubble_3d",
        execution="render-test",
        output_root=tmp_path,
    ).create()
    x = np.linspace(-1000.0, 1000.0, 5)
    z = np.linspace(0.0, 2000.0, 4)
    base = np.exp(
        -(
            (x[None, :, None] / 700.0) ** 2
            + ((z[None, None, :] - 1000.0) / 700.0) ** 2
        )
    )
    theta = np.stack([base, 0.98 * base], axis=0)
    dataset = xr.Dataset(
        data_vars={
            "theta_perturbation": (("core", "time", "x", "z"), theta),
            "mass_drift": ("core", [1.0e-10, 2.0e-10]),
            "symmetry_error": ("core", [1.0e-12, 2.0e-12]),
        },
        coords={
            "core": ["sisl", "split-explicit"],
            "time": [0.0],
            "x": x,
            "z": z,
        },
    )
    artifact = save_plot_dataset(
        dataset,
        layout.data / "artifact.nc",
        experiment="rising_bubble_dual_core_3d",
    )

    renderer_path = (
        Path(__file__).parents[1]
        / "benchmarks"
        / "physical"
        / "rising_bubble_3d"
        / "render.py"
    )
    spec = importlib.util.spec_from_file_location("rising_bubble_renderer", renderer_path)
    assert spec is not None and spec.loader is not None
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)

    outputs = renderer.render(layout.root)

    assert outputs == [
        layout.figures / "rising_bubble_sisl_main.png",
        layout.figures / "rising_bubble_split_explicit_main.png",
        layout.figures / "rising_bubble_difference_main.png",
        layout.figures / "evolution.png",
    ]
    assert all(path.is_file() for path in outputs)
