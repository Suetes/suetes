# Suêtes documentation

Suêtes is a differentiable, fully compressible, nonhydrostatic limited-area
atmospheric model implemented in JAX. It provides two three-dimensional
dynamical cores over the same terrain-following Arakawa C grid:

- a semi-implicit semi-Lagrangian (SISL) core with a matrix-free GMRES solve;
- a split-explicit Eulerian RK3 core with acoustic substepping.

The repository supports idealized verification, ERA5-driven regional
simulations, adjoint sensitivity analysis, inverse problems, and learned
vertical coordinates. It is research software rather than an operational
forecasting system.

## Where to begin

- New users should start with [Installation and quick start](getting-started.md).
- Model developers should read [Architecture and data flow](architecture.md)
  and [Model state and grid](model-state.md).
- Readers interested in the mathematics should continue to
  [Scientific formulation](scientific-formulation.md) and the two core guides.
- Differentiation users should read [Differentiability and adjoints](differentiability.md)
  before selecting controls or objectives.
- Paper readers can use [Reproducing the manuscript experiments](paper_suite.md)
  and [Verification and evidence](verification.md).

## Scope of this documentation

The documentation distinguishes three kinds of statements:

1. **Implemented formulation** describes behavior visible in the current code.
2. **Verification evidence** describes checks that are executable in this repository.
3. **Scientific context** explains why a method is used without claiming that a
   test establishes more than it measures.

This distinction is important for a differentiable model: a forward simulation
can be stable while a selected gradient is inaccurate, and a valid local Taylor
test does not by itself establish long-horizon forecast skill.

## Repository map

| Location | Responsibility |
|---|---|
| `suetes/regional3d/` | Three-dimensional grid, operators, dynamics, boundaries, diffusion, and steppers |
| `suetes/slice2d/` | Two-dimensional vertical-slice model |
| `suetes/physics/` | Physical parameterizations and their orchestration |
| `suetes/preprocessing/` | ERA5 download, remapping, topography, and boundary stores |
| `suetes/shared/` | Configuration, drivers, transforms, artifacts, output, and optimization |
| `benchmarks/` | Reusable physical and performance reference cases |
| `verification/` | Consistency, convergence, and adjoint checks |
| `experiments/` | Scientific questions, inversions, sensitivities, and learned methods |
| `runs/` | Configuration-driven real-data workflows |

The [module reference](api-reference.md) provides a source-oriented index.
