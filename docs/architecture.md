# Architecture and data flow

Suêtes separates reusable model machinery from the programs that pose scientific
questions. This lets benchmark, verification, experiment, and regional-run
entry points share the same dynamical components while retaining different
success criteria.

## Execution flow

```text
YAML or experiment arguments
          |
          v
 grid + vertical transform + operators
          |
          v
 background state + Euler tendencies + physics
          |
          v
 SISL stepper or split-explicit stepper
          |
          v
 Simulation / experiment integration
          |
          v
 NetCDF data artifact + JSON provenance
          |
          v
 standalone renderer -> figures
```

For ERA5-driven runs, preprocessing additionally constructs the terrain-following
grid, converts ERA5 fields to the model state, and writes time-dependent boundary
stores. Simulation reconstructs the same grid from saved static topography so it
does not repeat the external download and remapping stage.

## Core boundaries

`regional3d.geometry` owns coordinates and metrics. `regional3d.operators`
owns stagger-aware differences and averages. `regional3d.euler` evaluates
spatial tendencies but does not own time integration. `regional3d.steppers`
combines tendencies with advection, implicit or acoustic solves, physics,
forcing, and boundary conditions.

The two-dimensional slice follows the same division in `slice2d`. It is an
independent reduced model, not simply a three-dimensional run with one grid
cell in the transverse direction.

## Program categories

| Category | Question answered | Expected output |
|---|---|---|
| Tests | Did a deterministic behavior regress? | Pass/fail assertion |
| Verification | Is a numerical or adjoint property convergent or consistent? | Quantitative errors and orders |
| Physical benchmarks | Does the model reproduce a recognized reference problem? | Fields and comparisons |
| Performance benchmarks | What are runtime, memory, or scaling characteristics? | Resource measurements |
| Experiments | What scientific or differentiable-modeling question is being studied? | Case-specific diagnostics |
| Runs | How is an externally forced regional simulation executed? | Forecast artifacts and figures |

## Artifacts and provenance

```text
output/<kind>/<case>/<execution>/
├── data/
│   ├── artifact.nc
│   ├── artifact.json
│   └── case-specific tables or checkpoints
└── figures/
```

NetCDF artifacts contain reduced fields and diagnostics needed by the renderer,
not necessarily a restart-complete trajectory. JSON sidecars record provenance.
The paper-suite runner additionally records its manifest, checksum, console
logs, return codes, and timestamps.
