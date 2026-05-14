# Suetes

**Suetes** is a high-performance, differentiable, non-hydrostatic atmospheric fluid dynamics solver written in **JAX**.

It implements the fully compressible Euler equations on an **Arakawa C-grid** using **Terrain-Following Coordinates**. The codebase supports both a 2D vertical slice model and a full 3D regional model, enabling highly stable, large-timestep simulations via a **Semi-Implicit Semi-Lagrangian (SISL)** solver.

Because it is built entirely in JAX, the dynamical core is end-to-end differentiable and fully fused by XLA, making it ideal for machine learning integration (e.g., neural closures, hybrid modeling, data assimilation, and learned coordinate transformations).

---

##  Features

###  Dynamics & Numerics
* **Time Integration:** Semi-Implicit Semi-Lagrangian (SISL) allowing large timesteps by overcoming explicit acoustic and advective CFL limits.
* **Spatial Discretization:** Arakawa C-Grid for optimal dispersion properties.
* **Advection:** * Optimized 3D Tricubic and 2D Bicubic Semi-Lagrangian advection.
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
Run a real-world regional simulation over the Alps using ERA5 boundary conditions. This script will download data, initialize the 3D domain, and generate visualizations.
```bash
python run_simulation.py
```

### 2. Adjoint Sensitivity Analysis
Calculate the sensitivity of 3D wave energy with respect to the initial wind perturbation at $t=0$. This demonstrates the core's ability to propagate gradients backward through the solver.
```bash
python run_simulation_adjoint.py
python schaer_mountain_optimal_perturbation_3d.py
```

### 3. Inverse Topography Optimization
Optimize a 3D terrain profile to maximize the generation of gravity waves downstream, illustrating how the core can be used for "Atmospheric Engineering" and inverse modeling.
```bash
python inverse_topography_optimal_3d.py
```

### 4. Learned Neural Diffusion
Train a Flax neural network to predict an anisotropic Eddy Viscosity field ($\nu_h, \nu_v$) that preserves a target $k^{-5/3}$ spectral cascade, replacing traditional Smagorinsky closures with learned ones (does not work yet).
```bash
python learn_neural_diffusion.py
```