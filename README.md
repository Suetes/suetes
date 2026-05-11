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

The easiest way to see Suetes in action is to run the provided 3D simulation script. This script processes ERA5 data, initializes a regional domain, and runs a 6-hour simulation with full visualization output.

```bash
python run_simulation.py