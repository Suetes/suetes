# NEUVE diagnostics

These scripts record the optimization and adjoint investigations that led to
the publication workflow. They are useful for regression diagnosis, but they
are not stages of `paper_suite.toml`:

- `scalar_gradient_audit.py` checks the coordinate gradient against centered
  finite differences.
- `aggressiveness_scan.py` maps the PGF objective across a scalar family of
  increasingly terrain-decaying coordinates.
- `reachability_pipeline.py` separates scalar, global-profile, and
  terrain-conditioned optimization stages.

The supported publication workflow is:

1. `../create_dataset.py`
2. `../tune_sleve.py`
3. `../train_density.py`
4. `../evaluate.py`
5. `../render_training.py` and `../render_evaluation.py`

Run the complete workflow through `scripts/run_paper_suite.py`; these
diagnostics should only be run deliberately when investigating optimization
or adjoint behavior.
