# Scientific experiments

This directory contains paper-specific scientific experiments, training runs,
inverse problems, sensitivity analyses, and publication diagnostics. Scripts
here answer scientific questions rather than defining general numerical
correctness or reusable performance baselines.

Operational ERA5 preprocessing, forecast execution, and rendering entry points
belong in `runs/`.

## Case layout

Every experiment owns a directory. A typical case contains:

```text
experiments/<case>/
├── run.py
├── render.py
└── README.md          # optional case-specific scientific notes
```

Multi-stage cases use descriptive entry-point names instead. For example,
`neuve_coordinates/` contains `train.py`, `tune_sleve.py`, `evaluate.py`, and
their renderers. Shared implementation belongs in `experiments/_shared/`.

Current cases:

- `core_adjoint_analysis/`
- `core_quantitative_comparison/`
- `core_separation_showcase/`
- `differentiable_closure/`
- `era5_4dvar/`
- `gravity_wave_optimal_topography/`
- `neuve_coordinates/`
- `schaer_optimal_perturbation/`
- `simulation_adjoint/`
- `synthetic_4dvar/`
- `tracer_inversion_3d/`
- `tracer_inversion_3d_complex/`

## Output layout

Each execution owns a self-contained bundle:

```text
output/experiments/<experiment-name>/<execution-name>/
├── data/
│   ├── artifact.nc
│   ├── artifact.json
│   └── checkpoints, tables, or optimization histories
└── figures/
```

Entry points use `--name` to distinguish executions and `--output-root` to
redirect the bundle tree. `--output-dir` remains as a flat-directory
compatibility override. Where a simulation historically rendered inline,
`--no-render` or the existing opt-in plotting flag allows data production to
run independently. Renderer scripts accept either the execution bundle or its
primary artifact and never rerun the model.

## NEUVE coordinate-discovery workflow

The publication PGF experiment has four manifest-driven entry points:

1. `neuve_coordinates/create_dataset.py` freezes the training or testing
   terrain ensemble.
2. `neuve_coordinates/train_density.py` trains the geometry-initialized,
   terrain-conditioned density coordinate end to end on the training ensemble.
3. `neuve_coordinates/tune_sleve.py` tunes SLEVE on that exact training
   manifest.
4. `neuve_coordinates/evaluate.py` evaluates frozen NEUVE, tuned SLEVE, and
   Gal-Chen on an arbitrary held-out manifest.

The older `neuve_coordinates/train.py` remains available for the exploratory
tracer-reversibility and mountain-flux objectives. It is not used for the
publication PGF result.

All entry points support `--target pgf_rest`, `--target
tracer_reversibility`, and `--target mountain_flux`.  The first minimizes
kinetic energy generated from a resting stratified atmosphere.  The second
transports tracer mass with a prescribed divergence-free, terrain-tangent
flow, reverses that flow, and minimizes the normalized tracer return error.
The mountain-wave target runs the split-explicit compressible core in a
stratified background flow and minimizes the vertical non-uniformity of
resolved orographic momentum flux between the terrain and upper sponge.  The
last two objectives require no analytical trajectory or high-resolution
reference.

For the mountain-wave experiment, `--terrain-family multiscale3d` combines
resolved broad massifs and narrow peaks in every sample.  Its objective compares
the post-spin-up, time-mean momentum flux in the lower troposphere with that
transmitted into the upper troposphere.  This exposes the compromise imposed by
a single global terrain-decay scale while allowing a terrain-conditioned
coordinate to respond locally.

All three commands should use the same `--name` and `--output-root`. Data files
such as `neuve_coordinate.npz`, `training_dataset.json`, `best_sleve.json`,
and CSV/JSON diagnostics are kept in the bundle's `data/` directory. The SLEVE
script defaults to the directory containing its dataset, and evaluation can
infer the NEUVE and SLEVE filenames from its resolved data directory.

SLEVE tuning first uses inexpensive grid geometry and bisection to locate the
largest feasible exponent at each decay scale.  For the resting-PGF target it
then evaluates those boundary candidates, where the improvement is monotonic.
Tracer reversibility and mountain-wave flux are not assumed monotonic in
terrain decay, so those targets also evaluate feasible interior exponents at
every scale.  Evaluation reports
paired terrain-by-terrain reductions and a deterministic 95\% bootstrap
confidence interval (10,000 resamples by default).  For an ablation that
separates the benefit of the architecture from the benefit of optimization,
tracer evaluation can also report the initialization with
`--include-untrained`; it is omitted from publication plots by default.  The
evaluator saves common-axis coordinate and metric galleries for the first
three test terrains; use `--plot-samples` to change that count.

Use `--terrain-family ridge --seeds 999` for fixed-domain curve fitting, or
provide multiple comma-separated seeds (normally with `random3d`) for dataset
training.  A held-out test manifest must use seeds not present in the training
manifest. `_shared/neuve_coordinate.py` is a shared implementation module
rather than an experiment entry point.

The default NEUVE optimization is a reproducible 90-update run: 30 discovery
updates at $5\times10^{-3}$ followed automatically by a reset Adam optimizer
and 60 refinement updates at $3\times10^{-3}$.  No pre-existing weights are
required.  `--initial-weights` remains available only for deliberate restarts.

The obsolete acoustic-energy training and evaluation entry points have been
removed.

## Plot-ready artifacts

Long-running experiments write a self-describing NetCDF file alongside their
figures.  The files contain the reduced fields and diagnostic time series used
by the plots, plus a JSON provenance sidecar with the Git commit and experiment
metadata.  They intentionally omit full restart states.

NEUVE evaluation writes `coordinate_evaluation.nc`; regenerate its galleries
without running the model with:

```bash
python experiments/neuve_coordinates/render_evaluation.py \
  output/experiments/neuve_coordinates/EXPERIMENT/data/coordinate_evaluation.nc
```

The optimal-topography experiment writes
`gravity_wave_optimal_topography_CORE.nc`; regenerate its figures with
`experiments/gravity_wave_optimal_topography/render.py`. The Wreckhouse run
writes `wreckhouse_worst_case_feb2025_plot_data.nc`, consumed by
`runs/plot_wreckhouse_worst_case.py`.
