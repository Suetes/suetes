#!/usr/bin/env python3
"""Render precipitation sensitivities from a saved squall-line artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np
import xarray as xr


def symmetric_limit(field):
    finite = np.abs(np.asarray(field)[np.isfinite(field)])
    if not finite.size:
        return 1.0
    return max(float(np.percentile(finite, 99.0)), 1.0e-20)


def storm_limits(coordinate, occupied, padding, domain_limit=None):
    indices = np.flatnonzero(occupied)
    if not indices.size:
        return float(coordinate[0]), float(coordinate[-1])
    lower = max(float(coordinate[indices[0]]) - padding, float(coordinate[0]))
    upper = min(float(coordinate[indices[-1]]) + padding, float(coordinate[-1]))
    if domain_limit is not None:
        lower = max(lower, domain_limit[0])
        upper = min(upper, domain_limit[1])
    return lower, upper


def scientific_colorbar(fig, image, axis, label):
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    bar = fig.colorbar(image, ax=axis, pad=0.02, fraction=0.047, format=formatter)
    bar.set_label(label)
    bar.update_ticks()
    return bar


def render(
    artifact: Path, output_dir: Path, theta_perturbation=0.1, vapor_perturbation=1.0e-4, boundary_layer_top=3000.0
):
    with xr.open_dataset(artifact) as source:
        dataset = source.load()
    output_dir.mkdir(parents=True, exist_ok=True)
    x = dataset.x.values / 1000.0
    y = dataset.y.values / 1000.0
    z = dataset.z.values / 1000.0

    rain = dataset.window_accumulated_rain.values
    target = dataset.target_mask.values.astype(bool)
    boundary_layer = dataset.z.values <= boundary_layer_top
    theta_response = theta_perturbation * np.sum(dataset.theta_sensitivity.values[:, :, boundary_layer], axis=2)
    vapor_response = vapor_perturbation * np.sum(dataset.water_vapor_sensitivity.values[:, :, boundary_layer], axis=2)
    section_y = float(dataset.attrs.get("section_y_m", dataset.y.values[len(y) // 2])) / 1000.0
    x_limits = storm_limits(x, np.any(target, axis=1), 15.0)
    y_limits = storm_limits(y, np.any(target, axis=0), 15.0)
    panels = (
        (rain, "Blues", "Accumulated rain (mm)", None),
        (
            theta_response,
            "RdBu_r",
            rf"$\Delta J$ for lowest-{boundary_layer_top / 1000:g}-km "
            rf"${theta_perturbation:g}$ K "
            r"(mm h$^{-1}$)",
            symmetric_limit(theta_response),
        ),
        (
            vapor_response,
            "RdBu_r",
            rf"$\Delta J$ for lowest-{boundary_layer_top / 1000:g}-km "
            rf"${1000.0 * vapor_perturbation:g}$ g kg$^{{-1}}$ "
            r"(mm h$^{-1}$)",
            symmetric_limit(vapor_response),
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), constrained_layout=True)
    for axis, (field, cmap, label, limit) in zip(axes, panels):
        if limit is None:
            norm = colors.Normalize(vmin=0.0, vmax=max(float(np.max(field)), 1e-6))
        else:
            norm = colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        image = axis.pcolormesh(x, y, field.T, shading="auto", cmap=cmap, norm=norm)
        scientific_colorbar(fig, image, axis, label)
        axis.contour(x, y, target.T.astype(float), levels=[0.5], colors="k", linewidths=0.65)
        axis.axhline(section_y, color="0.35", linewidth=0.65, linestyle="--")
        axis.set_xlabel("$x$ (km)")
        axis.set_xlim(x_limits)
        axis.set_ylim(y_limits)
        axis.set_aspect("equal")
    axes[0].set_ylabel("$y$ (km)")
    axes[0].set_title("(a) Precipitation window")
    axes[1].set_title(r"(b) Sensitivity to $\theta_d$")
    axes[2].set_title(r"(c) Sensitivity to $q_v$")
    map_path = output_dir / "squall_precipitation_sensitivity_maps.png"
    fig.savefig(map_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    theta_xz = theta_perturbation * dataset.theta_sensitivity_xz.values
    vapor_xz = vapor_perturbation * dataset.water_vapor_sensitivity_xz.values
    cloud = dataset.spinup_cloud_water_xz.values
    vertical_velocity = dataset.spinup_vertical_velocity_xz.values
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.6), constrained_layout=True)
    for axis, field, label, title in (
        (
            axes[0],
            theta_xz,
            rf"$\Delta J$ for local ${theta_perturbation:g}$ K "
            r"(mm h$^{-1}$)",
            r"(a) $\theta_d$ sensitivity",
        ),
        (
            axes[1],
            vapor_xz,
            rf"$\Delta J$ for local "
            rf"${1000.0 * vapor_perturbation:g}$ g kg$^{{-1}}$ "
            r"(mm h$^{-1}$)",
            r"(b) $q_v$ sensitivity",
        ),
    ):
        limit = symmetric_limit(field)
        image = axis.pcolormesh(
            x, z, field.T, shading="auto", cmap="RdBu_r", norm=colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        )
        scientific_colorbar(fig, image, axis, label)
        cloud_level = 1.0e-4
        if float(np.nanmax(cloud)) >= cloud_level:
            axis.contour(x, z, cloud.T, levels=[cloud_level], colors="#e69f00", linewidths=0.8)
        updraft_level = 5.0
        if float(np.nanmax(vertical_velocity)) >= updraft_level:
            axis.contour(x, z, vertical_velocity.T, levels=[updraft_level], colors="k", linewidths=0.7)
        axis.set_ylabel("$z$ (km)")
        axis.set_xlim(x_limits)
        axis.set_ylim(0.0, min(15.0, float(z[-1])))
        axis.set_title(title)
    axes[-1].set_xlabel("$x$ (km)")
    axes[0].legend(
        handles=[
            Line2D([0], [0], color="#e69f00", lw=1.0, label=r"$q_c=0.1$ g kg$^{-1}$"),
            Line2D([0], [0], color="k", lw=1.0, label=r"$w=5$ m s$^{-1}$"),
        ],
        loc="upper right",
        frameon=False,
        fontsize=8,
    )
    section_path = output_dir / "squall_precipitation_sensitivity_sections.png"
    fig.savefig(section_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    taylor_path = output_dir / "squall_precipitation_taylor_tests.png"
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), constrained_layout=True)
    for axis, amplitude, actual, linear, label, unit in (
        (
            axes[0],
            dataset.theta_taylor_amplitude.values,
            dataset.theta_taylor_actual_change.values,
            dataset.theta_taylor_linear_change.values,
            r"$\theta_d$",
            "K",
        ),
        (
            axes[1],
            dataset.q_taylor_amplitude.values * 1000.0,
            dataset.water_vapor_taylor_actual_change.values,
            dataset.water_vapor_taylor_linear_change.values,
            r"$q_v$",
            r"g kg$^{-1}$",
        ),
    ):
        axis.plot(amplitude, actual, "o-", label="Centered nonlinear")
        axis.plot(amplitude, linear, "s--", label="Adjoint")
        axis.set_xscale("log")
        axis.set_xlabel(f"Maximum perturbation ({unit})")
        axis.set_ylabel(r"$\Delta J$ (mm h$^{-1}$)")
        axis.set_title(label)
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.savefig(taylor_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {map_path}")
    print(f"Saved {section_path}")
    print(f"Saved {taylor_path}")
    return [map_path, section_path, taylor_path]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifact", nargs="?", type=Path, default=Path("output/squall_line_3d_sensitivity/sensitivity.nc")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/squall_line_3d_sensitivity"))
    parser.add_argument("--theta-perturbation", type=float, default=0.1)
    parser.add_argument(
        "--vapor-perturbation", type=float, default=1.0e-4, help="display perturbation in kg kg-1 (default: 0.1 g kg-1)"
    )
    parser.add_argument(
        "--boundary-layer-top", type=float, default=3000.0, help="top of the layer included in column-response maps (m)"
    )
    args = parser.parse_args()
    if not args.artifact.is_file():
        raise FileNotFoundError(args.artifact)
    render(args.artifact, args.output_dir, args.theta_perturbation, args.vapor_perturbation, args.boundary_layer_top)


if __name__ == "__main__":
    main()
