# Suêtes

Suêtes is a differentiable, fully compressible, nonhydrostatic limited-area
atmospheric model written in [JAX](https://github.com/jax-ml/jax). It is
designed for convection-permitting simulation, numerical verification,
adjoint sensitivity analysis, inverse problems, and learned components of the
discretization.

The model uses terrain-following coordinates and an Arakawa C grid. Two
three-dimensional dynamical cores expose the same model state and operators:

- a split-explicit Eulerian core with acoustic substepping; and
- a semi-implicit semi-Lagrangian (SISL) core with a matrix-free GMRES solve.

Both paths support reverse-mode differentiation. The SISL pressure solve uses
a custom linear-solve adjoint, while long explicit trajectories can use
gradient checkpointing to trade computation for memory.

> [!NOTE]
> Suêtes is research software. The repository contains validation cases and
> real-data demonstrations, but it is not an operational forecasting system.

See the [scope and limitations](docs/limitations.md) before interpreting model
or gradient results.

## What is included

- Dry, fully compressible nonhydrostatic dynamics in two-dimensional slices
  and three-dimensional regional domains.
- Gal--Chen, SLEVE, stretched SLEVE, and neural NEUVE vertical coordinates.
- Semi-Lagrangian and flux-form tracer transport with monotonicity controls.
- Kessler warm-rain microphysics and optional Smagorinsky--Lilly turbulence.
- Davies lateral relaxation, Rayleigh upper damping, numerical diffusion, and
  divergence damping.
- ERA5 preprocessing and time-dependent limited-area boundary forcing.
- JAX-native optimization, adjoint sensitivities, Taylor tests, and inverse
  modeling examples.
- Self-describing NetCDF plot artifacts, allowing figures to be regenerated
  without repeating expensive simulations.

Representative cases include rising thermals, Schär mountain waves,
Weisman--Klemp squall lines, tracer-source and terrain inversion, learned
vertical coordinates, and an adjoint-directed Wreckhouse wind scenario.

## Installation

Clone the repository and create an isolated environment:

```bash
git clone <repository-url> suetes
cd suetes
python3 -m venv .venv
.venv/bin/python3 -m pip install --upgrade pip
.venv/bin/python3 -m pip install -e .
```

JAX installation is platform dependent. For GPU execution, install the JAX
wheel appropriate for the installed CUDA version by following the
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html),
then install Suêtes with `pip install -e .`.

Confirm the installation with:

```bash
.venv/bin/python3 -m pytest -q
```

Large three-dimensional examples require an NVIDIA GPU with substantial
memory. Smaller verification and plotting programs can be run independently.

## Quick start

### Physical benchmark

Run and render the two-core rising-bubble comparison:

```bash
.venv/bin/python3 benchmarks/physical/rising_bubble_3d/run.py \
  --dx 50 --dt 1 --t-end 600 \
  --output-root output --name quickstart

.venv/bin/python3 benchmarks/physical/rising_bubble_3d/render.py \
  output/benchmarks/rising_bubble_3d/quickstart
```

### Inverse problem

Optimize eight terrain-basis amplitudes to alter a downstream gravity-wave
diagnostic:

```bash
.venv/bin/python3 experiments/gravity_wave_optimal_topography/run.py \
  split-explicit --output-root output --name quickstart

.venv/bin/python3 experiments/gravity_wave_optimal_topography/render.py \
  output/experiments/gravity_wave_optimal_topography/quickstart
```

### ERA5-driven regional run

The configuration-driven workflow separates preprocessing, integration, and
rendering:

```bash
.venv/bin/python3 runs/preprocess.py \
  --config configs/wreckhouse25_config.yaml

.venv/bin/python3 runs/run_simulation.py \
  --config configs/wreckhouse25_config.yaml

.venv/bin/python3 runs/render.py \
  --config configs/wreckhouse25_config.yaml
```

ERA5 downloads require a configured CDS API account. Static and atmospheric
input data are stored below `inputs/`, not in the output tree.

## Reproducing the manuscript experiments

[`paper_suite.toml`](paper_suite.toml) is the authoritative manifest for the
forward benchmarks, inverse problems, sensitivities, learned-coordinate
experiment, and appendix verification studies used by the manuscript. The
runner executes cases serially and stores all new results below
`output/paper/`.

Inspect the suite:

```bash
.venv/bin/python3 scripts/run_paper_suite.py list
.venv/bin/python3 scripts/run_paper_suite.py status
.venv/bin/python3 scripts/run_paper_suite.py all --dry-run
```

Run the main-table and appendix cases:

```bash
.venv/bin/python3 scripts/run_paper_suite.py all --group table
.venv/bin/python3 scripts/run_paper_suite.py all --group appendix
```

Run or render an individual case:

```bash
.venv/bin/python3 scripts/run_paper_suite.py run squall_forward
.venv/bin/python3 scripts/run_paper_suite.py render squall_forward
```

Completed stages are skipped unless `--force` is supplied. Each stage records
its command, console log, return code, timestamps, and Git commit. The output
root also receives the exact manifest and its checksum. See
[`docs/paper_suite.md`](docs/paper_suite.md) for details.

## Output and rendering contract

Benchmarks, experiments, runs, and verification studies use execution bundles:

```text
output/<kind>/<case>/<execution>/
├── data/
│   ├── artifact.nc
│   ├── artifact.json
│   └── checkpoints, histories, or diagnostic tables
└── figures/
```

The NetCDF artifacts contain the reduced fields and diagnostics required by
the corresponding figures. JSON sidecars record provenance. They are
plot-ready products rather than complete restart states.

Where supported, `--no-render` performs only the expensive computation.
Standalone `render.py` programs accept either an execution bundle or its
primary artifact. This separation makes plotting reproducible and permits
collaborators to regenerate figures without the model or a GPU.

## Repository layout

```text
suetes/
├── suetes/          # model implementation
│   ├── regional3d/  # three-dimensional dynamics and operators
│   ├── slice2d/     # two-dimensional slice model
│   ├── physics/     # microphysics, turbulence, and other physics
│   ├── preprocessing/
│   └── shared/      # configuration, drivers, transforms, and artifacts
├── benchmarks/      # reusable physical and performance benchmarks
├── experiments/     # inversions, sensitivities, and learned methods
├── verification/    # convergence and adjoint-consistency studies
├── runs/            # ERA5-driven regional workflows
├── configs/         # regional-run configurations
├── tests/           # automated regression tests
├── scripts/         # repository-level orchestration
└── paper_suite.toml # manuscript experiment manifest
```

The distinction is intentional:

- `benchmarks/` contains reusable physical cases and performance measurements;
- `verification/` tests numerical consistency and convergence;
- `experiments/` answers scientific or differentiable-modeling questions; and
- `runs/` contains externally forced regional workflows.

## Differentiable workflows

Any scalar functional constructed from a simulated trajectory can, subject to
the differentiability of the selected parameterizations, be differentiated
with respect to model inputs. Existing examples demonstrate gradients with
respect to:

- initial potential temperature and water vapor;
- tracer-source location and amplitude;
- terrain-basis coefficients;
- upstream thermal perturbations; and
- neural vertical-coordinate parameters.

Taylor remainder tests are included in the verification and sensitivity
programs to compare the adjoint directional derivative with centered nonlinear
perturbations. These checks are important whenever a new differentiated
objective, model component, or control variable is introduced.

## Scientific background

The limited-area, fully elastic formulation follows the lineage of Canadian
nonhydrostatic regional models including MC2, CRCM, and GEM. The coordinate and
transport options build on Gal--Chen terrain-following coordinates, SLEVE, and
conservative semi-Lagrangian methods. The repository’s experiments extend
these methods with automatic differentiation and PDE-supervised neural
coordinate discovery.

For exact equations, numerical choices, experiment configurations, and
literature citations, consult the manuscript and the case-specific source and
documentation in `benchmarks/`, `experiments/`, and `verification/`.

## Authors

Suêtes is authored by Alex Bihlo, Elsa Cardoso-Bihlo, Alejandro Di Luca,
Seth Taylor, and Tim Whittaker.

## License

Suêtes is licensed under the [Apache License 2.0](LICENSE). You may use,
modify, and distribute the software, including for commercial purposes, under
the conditions of that license. Please cite the associated scientific work
when using Suêtes in research; citation information will be added when the
manuscript record is available.
