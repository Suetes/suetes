# Suetes Documentation

Welcome to the **Suetes** documentation. 

**Suetes** is a high-performance, differentiable, non-hydrostatic atmospheric fluid dynamics solver written in **JAX**.

## Key Features

*   **Differentiable Dynamical Core**: Built entirely in JAX, allowing for end-to-end differentiability.
*   **High Performance**: Fully fused by XLA for CPU, GPU, and TPU.
*   **Advanced Numerics**: Semi-Implicit Semi-Lagrangian (SISL) time integration and Arakawa C-grid spatial discretization.
*   **Terrain-Following Coordinates**: Supports SLEVE, Hybrid Sigma-Z, and Gal-Chen coordinates.
*   **ERA5 & GEBCO Integration**: Automated preprocessing pipeline for real-world data.

## Contents

*   [Geometry](geometry.md): Detailed information about the 3D regional grid construction and map projections.
*   [Operators](operators.md): Detailed information about the finite difference stencils and interpolators.
*   [Euler](euler.md): Detailed information about the dynamical core.
*   [Steppers](steppers.md): Detailed information about the time integration and advection.
*   [Diffusion](diffusion.md): Detailed information about the spatial filters and turbulence models.
*   [Boundaries](boundaries.md): Detailed information about the relaxation zones and external forcing.

