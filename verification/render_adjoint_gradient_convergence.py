"""Render Taylor-remainder and temporal adjoint-convergence diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from suetes.shared.artifacts import ArtifactLayout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--name", default="default")
    args = parser.parse_args()
    layout = ArtifactLayout(
        kind="verification",
        case="adjoint_gradient_convergence",
        execution=args.name,
        output_root=args.output_root,
    ).create()
    summary = json.loads((layout.data / "summary.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    colors = {"split-explicit": "#D55E00", "sisl": "#0072B2"}
    for core, result in summary["results"].items():
        eps = np.asarray(result["epsilons"])
        axes[0].loglog(
            eps, result["first_order_remainder"], "o-",
            color=colors[core],
            label="Split-Explicit" if core == "split-explicit" else "SISL",
        )
    eps = np.asarray(next(iter(summary["results"].values()))["epsilons"])
    reference = np.asarray(
        next(iter(summary["results"].values()))["first_order_remainder"]
    )
    axes[0].loglog(eps, reference[0] * (eps / eps[0]) ** 2, "k--", label="Order 2")
    axes[0].set_xlabel(r"Perturbation amplitude $\varepsilon$")
    axes[0].set_ylabel(r"First-order Taylor remainder $R_1$")
    axes[0].set_title("Discrete-adjoint Taylor test")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend()

    for core, result in summary["results"].items():
        dt = np.asarray(result["time_steps"][:-1])
        axes[1].loglog(
            dt, result["successive_gradient_differences"], "o-",
            color=colors[core],
            label=("Split-Explicit" if core == "split-explicit" else "SISL"),
        )
    dt = np.asarray(next(iter(summary["results"].values()))["time_steps"][:-1])
    errors = np.asarray(
        next(iter(summary["results"].values()))["successive_gradient_differences"]
    )
    axes[1].loglog(dt, errors[0] * (dt / dt[0]) ** 2, "k--", label="Order 2")
    axes[1].invert_xaxis()
    axes[1].set_xlabel(r"Large time step $\Delta t$ (s)")
    axes[1].set_ylabel(r"Gradient $L_2$ difference")
    axes[1].set_title("Temporal adjoint convergence")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    output = layout.figures / "adjoint_gradient_convergence.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
