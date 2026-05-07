# Suetes

**Suetes** is a high-performance, differentiable, non-hydrostatic atmospheric fluid dynamics solver written in **JAX**.

It implements the fully compressible Euler equations on an **Arakawa C-grid** using **Terrain-Following Coordinates**. The codebase supports both a 2D vertical slice model and a full 3D regional model, enabling highly stable, large-timestep simulations via a **Semi-Implicit Semi-Lagrangian (SISL)** solver.

Because it is built entirely in JAX, the dynamical core is end-to-end differentiable and fully fused by XLA, making it ideal for machine learning integration (e.g., neural closures, hybrid modeling, data assimilation, and learned coordinate transformations).

---

## 🚀 Features

### 🧩 Dynamics & Numerics
*   **Time Integration:** Semi-Implicit Semi-Lagrangian (SISL) allowing large timesteps by overcoming explicit acoustic and advective CFL limits.
*   **Spatial Discretization:** Arakawa C-Grid for optimal dispersion properties.
*   **Advection:** 
    *   Optimized 3D Tricubic and 2D Bicubic Semi-Lagrangian advection.
    *   **Flux-Form Semi-Lagrangian (FFSL)** scheme for exact mass conservation of tracers and density.
    *   **Quasi-Monotone Limiters** to prevent unphysical undershoots in tracer transport.
*   **Implicit Solver:** JAX-native GMRES solver for the 2D/3D Helmholtz acoustic problem.
*   **Simulation Driver:** High-level `Simulation` class for managed integration loops, JIT compilation, and chunked execution.
*   **Stabilization:** 4th-Order Hyper-Diffusion ($\nabla^4$), Rayleigh damping sponge layers, and acoustic off-centering.

### 🏔️ Terrain & Coordinates
*   **Vertical Coordinates:**
    *   Standard Gal-Chen & Somerville (Linear decay).
    *   Hybrid Sigma-Z (Exponential decay).
    *   **SLEVE** (Smooth LEvel VErtical) for stable simulations over complex, steep topography.
    *   **Integral Neural Coordinate:** A differentiable, monotonicity-guaranteed ML-driven grid transformation.
*   **Projection:** Oblique Stereographic projection for regional modeling.

### 🔄 Data & Preprocessing
*   **ERA5 Integration:** Automated pipeline to ingest ERA5 pressure and single-level data.
*   **ERA5 Downloader:** Built-in `ERA5Manager` using `cdsapi` for automated regional data fetching and caching.
*   **Topography:** High-resolution **GEBCO** bathymetry/topography blending with ERA5.
*   **Boundary Processing:** 
    *   Automatic regridding and wind rotation to the model's grid basis.
    *   Thermodynamic balancing and hydrostatic reconstruction for clean cold-starts.
    *   **Davies Sponge** relaxation zones for seamless coupling with external forcing.

### 🌡️ Physics
*   **Bulk Aerodynamic PBL:** Surface drag parameterization for land and ocean.
*   **Rayleigh Damping:** Top-of-atmosphere sponge layer to prevent wave reflection.

### 📊 Visualization & Diagnostics
Comprehensive `Visualizer` suite for model analysis:
*   **Cross-sections:** Vertical slices of potential temperature and topography.
*   **Geographic Maps:** Interactive maps with Cartopy integration.
*   **Spectral Analysis:** Spatial power spectrum plots (Energy vs. Wavenumber) with theoretical $k^{-5/3}$ references.
*   **Dashboards:** Multi-panel diagnostic views of W, Divergence, and Potential Temperature.
*   **Anomalies:** Explicit difference mapping between Suetes and ERA5 target states.

---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/abihlo/suetes.git
    cd suetes
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the package:**
    ```bash
    pip install -e .
    ```

### Apple Silicon (M-Series) Support
To enable GPU acceleration on Mac, install the Metal plugin:
```bash
pip install "numpy<2" jax-metal
```

---

## 🚦 Getting Started

The easiest way to see Suetes in action is to run the provided 3D simulation script. This script processes ERA5 data, initializes a regional domain, and runs a 6-hour simulation with full visualization output.

```bash
python run_simulation.py
```

Results will be saved in `suetes/plots/`, including:
*   `dashboard_6h.png`: A 4-panel overview of the final model state.
*   `energy_spectrum_6h.png`: Analysis of the resolved turbulence scales.
*   `isentropes_6h.png`: Wave breaking diagnostics.

---

## 🧪 Diagnostic Suite

Suetes includes an extensive suite of tests in `test_dynamical_core_3d.py` and `test_era5_coupling.py` that validate:
*   **Exact Mass Conservation** of the FFSL scheme.
*   **Hydrostatic Balance** over extremely steep terrain.
*   **Metric Gradient Consistency** (no spurious winds from grid transformations).
*   **Solver Convergence** and stability.

---

## 📁 Project Structure

```text
suetes/
├── suetes/              # Core package
│   ├── data/            # Sample NetCDF data (ERA5/GEBCO)
│   ├── preprocessing/   # ERA5 (downloader/processor) & Topography pipeline
│   ├── regional3d/      # 3D Regional Model
│   ├── shared/          # Shared utilities, driver, and Coordinate Transforms
│   ├── slice2d/         # 2D Vertical Slice Model
│   └── vis/             # Visualization and Dashboarding suite
├── run_simulation.py    # Main entry point for 3D regional runs
├── test_suite_2d.py     # 2D standardized benchmarks
└── test_dynamical_core_3d.py # 3D stability and diagnostic tests
```
