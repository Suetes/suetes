# Scientific experiments

This directory contains paper-specific scientific experiments, training runs,
inverse problems, sensitivity analyses, and publication diagnostics. Scripts
here answer scientific questions rather than defining general numerical
correctness or reusable performance baselines.

Operational ERA5 preprocessing, forecast execution, and rendering entry points
belong in `runs/`.

## NEUVE resting-PGF workflow

The publication PGF experiment has three manifest-driven entry points:

1. `tune_neuve_pgf.py` tunes NEUVE on one or more terrain samples and writes
   `training_dataset.json`, a frozen checkpoint, history, and training plot.
2. `tune_sleve_pgf.py` grid-searches SLEVE on that exact training manifest.
3. `evaluate_pgf_coordinates.py` evaluates frozen NEUVE, tuned SLEVE, and
   Gal-Chen on an arbitrary testing manifest containing one or more samples.

All three commands should use the same `--output-dir`.  Files are kept flat in
that experiment directory: `neuve_pgf.npz`, `training_dataset.json`,
`best_sleve.json`, CSV/JSON diagnostics, and PNG figures.  The SLEVE script
defaults to the directory containing its dataset, and evaluation can infer the
NEUVE and SLEVE filenames from `--output-dir`.

SLEVE tuning first uses inexpensive grid geometry and bisection to locate the
largest feasible exponent at each of ten decay scales, then executes the
dynamical model only for those ten boundary candidates.  This exploits the
observed monotonic improvement toward the layer-thickness constraint without a
costly rectangular grid search.  Evaluation reports paired terrain-by-terrain reductions and a
deterministic 95\% bootstrap confidence interval (10,000 resamples by default).
By default the evaluator also saves common-axis coordinate and TKE galleries
for the first three test terrains; use `--plot-samples` to change that count.

Use `--terrain-family ridge --seeds 999` for fixed-domain curve fitting, or
provide multiple comma-separated seeds (normally with `random3d`) for dataset
training.  A held-out test manifest must use seeds not present in the training
manifest.  `neuve_pgf_common.py` is a shared implementation module rather than
an experiment entry point.

The default NEUVE optimization is a reproducible 90-update run: 30 discovery
updates at $5\times10^{-3}$ followed automatically by a reset Adam optimizer
and 60 refinement updates at $3\times10^{-3}$.  No pre-existing weights are
required.  `--initial-weights` remains available only for deliberate restarts.

The older `train_neuve.py` and `evaluate_neuve.py` remain for the separate
acoustic-coordinate experiments; they are not part of the publication PGF
workflow.
