# Numerical verification

This directory contains quantitative checks of numerical consistency and
convergence. These programs are intentionally separate from pytest regression
tests and from physical/performance benchmarks.

Active verification programs:

- `equation_tendency_convergence.py`: spatial consistency of the discrete
  Euler tendencies against manufactured analytical tendencies.
- `temporal_core_convergence.py`: pure temporal self-convergence of SISL and
  split-explicit integration.
- `rising_bubble_core_convergence.py`: combined space-time self-convergence
  and cross-core convergence using the rising thermal bubble.
- `sisl_component_convergence.py`: interpolation, trajectory, implicit-solver,
  operator, and advection convergence.
- `gmres_adjoint_convergence.py`: adjoint sensitivity to the GMRES solve.
- `verify_dynamical_cores.py`: isolated-process orchestration of the active
  verification ladder.

The `legacy/` directory retains superseded diagnostics for provenance. In
particular, the short integrated MMS programs are not active verification
gates because temporal and solver-error floors obscure their spatial rates.

Run the quick verification ladder from the repository root with:

```bash
python verification/verify_dynamical_cores.py --profile quick
```

