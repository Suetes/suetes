# Scientific experiments

This directory contains paper-specific scientific experiments, training runs,
inverse problems, sensitivity analyses, and publication diagnostics. Scripts
here answer scientific questions rather than defining general numerical
correctness or reusable performance baselines.

Operational ERA5 preprocessing, forecast execution, and rendering entry points
belong in `runs/`.

## NEUVE coordinate-discovery workflow

The coordinate experiments have three manifest-driven entry points:

1. `train_neuve_coordinate.py` trains NEUVE on one or more terrain samples and writes
   `training_dataset.json`, a frozen checkpoint, history, and training plot.
2. `tune_sleve_coordinate.py` tunes SLEVE on that exact training manifest.
3. `evaluate_neuve_coordinates.py` evaluates frozen NEUVE, tuned SLEVE, and
   Gal-Chen on an arbitrary testing manifest containing one or more samples.

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

All three commands should use the same `--output-dir`.  Files are kept flat in
that experiment directory: `neuve_coordinate.npz`, `training_dataset.json`,
`best_sleve.json`, CSV/JSON diagnostics, and PNG figures.  The SLEVE script
defaults to the directory containing its dataset, and evaluation can infer the
NEUVE and SLEVE filenames from `--output-dir`.

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
manifest.  `neuve_coordinate_common.py` is a shared implementation module rather than
an experiment entry point.

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
python experiments/plot_neuve_coordinate_evaluation.py \
  output/EXPERIMENT/coordinate_evaluation.nc
```

The optimal-topography experiment writes
`gravity_wave_optimal_topography_CORE.nc`; regenerate its figures with
`experiments/plot_gravity_wave_optimal_topography.py`.  The Wreckhouse run
writes `wreckhouse_worst_case_feb2025_plot_data.nc`, consumed by
`runs/plot_wreckhouse_worst_case.py`.
