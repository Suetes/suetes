# AtmosJax

**AtmosJax** is a high-performance, differentiable, non-hydrostatic atmospheric fluid dynamics solver written in **JAX**.

It implements the fully compressible Euler equations (Equation Set 1: Exner-Theta form) on an **Arakawa C-grid** using **Terrain-Following Coordinates**. This codebase is designed to reproduce the standard mesoscale benchmarks from [Giraldo & Restelli (2008)](https://doi.org/10.1016/j.jcp.2007.12.009) while enabling automatic differentiation for machine learning applications (e.g., end-to-end training with NeuralGCM, Data Assimilation).

## Features

* **Core:** 2D Compressible Euler Equations (Perturbation form).
* **Grid:** Staggered Arakawa C-Grid with Gal-Chen & Somerville terrain transformation.
* **Numerics:**
    * **Spatial:** 4th-Order Centered Differences (Periodic) and 2nd-Order Centered Differences (Wall-Bounded).
    * **Time:** Strong Stability Preserving Runge-Kutta (SSPRK3) and Classic RK4.
    * **Stabilization:** 4th-Order Hyper-Diffusion ($\nabla^4$) and Physical Laplacian Viscosity ($\nabla^2$).
    * **Advection:** Full 2D non-linear advection with correct metric terms for terrain.
    * **BCs:** Sponge Layers (Rayleigh damping) and Rigid/Free-Slip walls.
* **Backend:** JAX (Supports CPU, NVIDIA GPU, and Apple Silicon/Metal).

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/abihlo/atmos_jax.git
    cd atmos_jax
    ```

2.  Create a virtual environment (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  Install the package in editable mode:
    ```bash
    pip install -e .
    ```

### Apple Silicon (M1/M2/M3) Setup
To enable GPU acceleration on Mac, you must install the Metal plugin. Note that `numpy < 2` is often required for compatibility with current `jax-metal` releases.

```bash
pip install "numpy<2"
pip install jax-metal
