# Module reference

This source-oriented index describes module responsibilities. The generated
pages expose current signatures and source docstrings directly from the code:

- [Regional three-dimensional API](api/regional3d.md)
- [Two-dimensional slice API](api/slice2d.md)
- [Physics API](api/physics.md)
- [Shared infrastructure API](api/shared.md)
- [Preprocessing API](api/preprocessing.md)
- [Visualization API](api/visualization.md)

Scientific rationale remains in the narrative chapters so API contracts and
scientific explanation can evolve at their appropriate levels.

## Regional three-dimensional model

| Module | Principal interface | Responsibility |
|---|---|---|
| `suetes.regional3d.geometry` | `ObliqueStereographic`, `RegionalGrid3D` | Projection, coordinates, height, metrics |
| `suetes.regional3d.operators` | `CGridOperator3D` | Stagger-aware interpolation and differences |
| `suetes.regional3d.euler` | `Euler3D` | Background construction and spatial tendencies |
| `suetes.regional3d.steppers` | `build_dynamical_core`, `SISLStepper3D`, `SplitExplicitStepper3D` | Advection and integration |
| `suetes.regional3d.boundaries` | `ExternalForcing`, `DaviesBoundary` | Time interpolation and lateral relaxation |
| `suetes.regional3d.diffusion` | `HyperFilter`, `DivergenceDamper` | Numerical dissipation |

## Two-dimensional slice model

| Module | Responsibility |
|---|---|
| `suetes.slice2d.geometry` | Terrain-following slice geometry |
| `suetes.slice2d.operators` | Slice interpolation and differences |
| `suetes.slice2d.euler` | Slice-model spatial tendencies |
| `suetes.slice2d.steppers` | Slice advection and integration |

## Shared infrastructure

| Module | Responsibility |
|---|---|
| `suetes.shared.config` | YAML schema, effective parameters, grids, caches |
| `suetes.shared.driver` | Chunked and differentiable integration |
| `suetes.shared.transforms` | Analytical, SLEVE, stretched, learned mappings |
| `suetes.shared.experiment` | Execution paths and provenance |
| `suetes.shared.output` | Model NetCDF output |
| `suetes.shared.latlon_output` | Geographic remapped output |
| `suetes.shared.optimization` | JAX/Optax optimization support |

## Physics, preprocessing, and visualization

`suetes.physics.base.PhysicsSuite` orchestrates tendency schemes and
instantaneous update schemes. Neighboring modules implement microphysics,
surface processes, turbulence, gravity-wave effects, forcing, and learned
closures.

Preprocessing covers ERA5 download, dataset assembly, topography blending,
model-state conversion, and boundary stores. See
[Data and preprocessing](era2suetes.md).

`suetes.vis` provides reusable maps, sections, diagnostics, and comparisons.
Case-specific figure composition remains in each case's renderer.
