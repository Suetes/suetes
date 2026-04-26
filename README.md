# Suetes

**Suetes** is a high-performance, differentiable, non-hydrostatic atmospheric fluid dynamics solver written in **JAX**.

It implements the fully compressible Euler equations on an **Arakawa C-grid** using **Terrain-Following Coordinates**. The codebase supports both a 2D vertical slice model and a full 3D regional model, enabling highly stable, large-timestep simulations via a **Semi-Implicit Semi-Lagrangian (SISL)** solver.

Because it is built entirely in JAX, the dynamical core is end-to-end differentiable and fully fused by XLA, making it ideal for machine learning integration (e.g., neural closures, hybrid modeling, data assimilation).

## Features

* **Dynamics Architecture:**
    * **`slice2d`**: 2D Vertical Slice non-hydrostatic Euler equations.
    * **`regional3d`**: Full 3D Regional non-hydrostatic Euler equations with Oblique Stereographic projection.
    * **`shared`**: Shared numerical utilities, coordinate transforms, and model drivers.
* **Numerics:**
    * **Time Integration:** Semi-Implicit Semi-Lagrangian (SISL) allowing large timesteps (overcoming explicit acoustic and advective CFL limits).
    * **Spatial Discretization:** Arakawa C-Grid.
    * **Advection:** Semi-Lagrangian advection with optimized 3D Tricubic (tensor-product) and 2D Bicubic interpolation. Eulerian TVD (Van Leer) interpolation for fluxes.
    * **Implicit Solver:** JAX-native GMRES solver for the 2D/3D Helmholtz acoustic problem.
    * **Stabilization:** 4th-Order Hyper-Diffusion ($\nabla^4$) and acoustic off-centering.
* **Vertical Coordinates:**
    * Standard Gal-Chen & Somerville.
    * Hybrid Sigma-Z.
    * SLEVE (Smooth LEvel VErtical) for complex topography.
    * Integral Neural Coordinate for ML-driven grid transformations.
* **Boundaries:**
    * Davies Sponge relaxation zones for 3D lateral boundaries (e.g., blending with ERA5).
    * Top/Bottom rigid walls, periodic conditions, and Rayleigh damping sponge layers.
* **Backend:** JAX (Supports CPU, NVIDIA GPU, and Apple Silicon/Metal).

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/abihlo/suetes.git
    cd suetes
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
```
