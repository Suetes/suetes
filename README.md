# Suetes

**Suetes** is a high-performance, differentiable, non-hydrostatic atmospheric fluid dynamics solver written in **JAX**.

It implements the fully compressible Euler equations on an **Arakawa C-grid** using **Terrain-Following Coordinates**. The codebase supports both a 2D vertical slice model and a full 3D regional model, featuring a dual-dynamical core architecture: a **Semi-Implicit Semi-Lagrangian (SISL)** solver for large-timestep coarse runs, and a strictly Eulerian **Split-Explicit Runge-Kutta** solver for convection-permitting high-resolution runs.

Because it is built entirely in JAX, the dynamical cores are end-to-end differentiable and fully fused by XLA, making it ideal for machine learning integration (e.g., neural closures, hybrid modeling, data assimilation, and learned coordinate transformations).

---

##  Features

###  Dynamics & Numerics
* **Dual Dynamical Cores:** Support for two runtime-selectable dynamical cores:
    * **Semi-Implicit Semi-Lagrangian (SISL):** 2nd-order time integration using midpoint trajectory and tendency extrapolation, bypassing explicit acoustic and advective CFL limits for highly efficient coarse simulations ($\ge 10$~km).
    * **Eulerian Split-Explicit Runge-Kutta:** strictly Eulerian split-explicit scheme handling small-scale nonlinearities exceptionally well, optimal for high-resolution convection-permitting simulations ($\le 3$~km).
* **Spatial Discretization:** Arakawa C-Grid for optimal dispersion properties.
* **Advection:**
    * Optimized 3D Tricubic and 2D Bicubic Semi-Lagrangian advection.
    * **Flux-Form Semi-Lagrangian (FFSL)** scheme for exact mass conservation of tracers and density.
    * **Quasi-Monotone Limiters** to prevent unphysical undershoots in tracer transport.
* **Implicit Solver:** JAX-native GMRES solver for the 2D/3D Helmholtz acoustic problem.
* **Simulation Driver:** High-level `Simulation` class for managed integration loops, JIT compilation, and chunked execution.
* **Stabilization:** Configurable Davies sponge lateral boundaries, Rayleigh damping at the model top, and divergence damping for acoustic modes.

###  Differentiable Science & ML Integration
* **Reverse-Mode AD:** Full support for backpropagating through the entire 3D dynamical core, including the implicit GMRES solver and Semi-Lagrangian trajectories.
* **Adjoint Modeling:** Calculate sensitivities of downstream states (e.g., kinetic energy) to initial conditions or boundary forcing.
* **Inverse Problems:** Optimize terrain profiles or physical parameters to match target observations.
* **Neural Closures:** Integrate Flax-based neural networks directly into the physics suite for learned subgrid-scale (SGS) parameterizations.
* **Checkpointing:** Native support for `jax.checkpoint` (Rematerialization) to handle long-horizon adjoint sensitivity runs within memory constraints.

---

## Scientific Lineage & Theoretical Foundations

Suetes is built upon decades of research in Semi-Implicit Semi-Lagrangian (SISL) atmospheric modeling. While modernized for GPU execution via JAX, its architectural DNA traces back to several seminal models and papers:

* **Core Architecture (CRCM & MC2):** The fully elastic, non-hydrostatic Euler equations solved in perturbation form on a limited-area domain directly mirror the Canadian Regional Climate Model (CRCM) (Caya and Laprise, 1999) and the Mesoscale Compressible Community (MC2) model (Tanguay et al., 1990).
* **SISL Formulation:** The implicit stabilization of acoustic and gravity waves draws heavily from the principles outlined by Wood et al. (2013), adapted here for a shallow-atmosphere regional framework.
* **Mass Conservation (FFSL):** To solve the classic mass-leakage problem of standard SL advection, Suetes uses a split Flux-Form Semi-Lagrangian scheme inspired by the Cell-Integrated Semi-Lagrangian (CISL) method (Kaas, 2008).
* **Vertical Coordinates:** The model utilizes the SLEVE (Smooth LEvel VErtical) coordinate (Schär et al., 2002) to minimize truncation errors over steep topography.
* **Helmholtz Preconditioning:** The vertical tridiagonal preconditioner applied before the GMRES solve is heavily inspired by the non-hydrostatic formulation of the Canadian GEM model (Yeh et al., 2002).

---

## Quickstart

### 1. Forward Simulation (ERA5 Driven)
Run a real-world regional simulation over Wreckhouse using ERA5 boundary conditions. This requires a three-step configuration-driven workflow:
* **Preprocess boundary conditions** (download ERA5 and terrain raw data, compile Zarr boundary stores):
```bash
python runs/preprocess.py --config configs/wreckhouse25_config.yaml
```
* **Execute the forward run** (integrated via JIT compiled core and save results):
```bash
python runs/run_simulation.py --config configs/wreckhouse25_config.yaml
```
* **Render output fields** (generate figures beside the run artifact):
```bash
python runs/render.py --config configs/wreckhouse25_config.yaml
```

### 2. Adjoint Sensitivity Analysis
Calculate the sensitivity of 3D wave energy with respect to the initial wind perturbation at $t=0$. This demonstrates the core's ability to propagate gradients backward through the solver.
```bash
python experiments/simulation_adjoint/run.py
python experiments/schaer_optimal_perturbation/run.py
```

### 3. Inverse Topography Optimization
Optimize a 3D terrain profile to maximize the generation of gravity waves downstream, illustrating how the core can be used for "Atmospheric Engineering" and inverse modeling.
```bash
python experiments/gravity_wave_optimal_topography/run.py
```

## Repository organization

- `runs/`: operational ERA5 coupling, simulation, and rendering workflows.
- `experiments/`: paper-specific science, inverse problems, and sensitivities.
- `benchmarks/`: reusable physical cases and performance measurements.
- `verification/`: numerical consistency and convergence programs.
- `tests/`: automated pytest regression tests.

## Artifact and rendering contract

Benchmarks, experiments, operational runs, and numerical verification use the
same execution-bundle structure:

```text
output/<kind>/<case>/<execution>/
├── data/
└── figures/
```

Model and optimization entry points support data-only execution, normally via
`--no-render`. Standalone renderers consume the saved artifact and write to the
sibling `figures/` directory, so plots can be regenerated without rerunning an
expensive model. Some older entry points still render by default for command
compatibility; the case READMEs document those flags.
