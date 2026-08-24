# Verification and evidence

Suêtes separates fast regression tests, numerical verification, physical
benchmarks, and scientific experiments. They provide complementary evidence.

## Regression tests

`pytest` collects only `tests/`. Tests cover formula regressions, periodic
boundaries, both cores, selected physics, ERA5 coupling, experiment layout,
paper-suite behavior, and the simulation driver.

```bash
python -m pytest -q
```

## Numerical verification ladder

| Program | Property measured |
|---|---|
| `equation_tendency_convergence.py` | Spatial consistency against manufactured tendencies |
| `temporal_core_convergence.py` | Fixed-space temporal self-convergence |
| `rising_bubble_core_convergence.py` | Space-time and cross-core convergence |
| `smooth_mode_core_convergence.py` | Tracer, gravity, and acoustic mode refinement |
| `sisl_component_convergence.py` | SISL interpolation, trajectory, solver, and advection components |
| `gmres_adjoint_convergence.py` | Adjoint sensitivity to linear-solver convergence |
| `adjoint_gradient_convergence.py` | Taylor tests and cross-core gradient behavior |
| `long_horizon_core_audit.py` | Opt-in multi-grid long-horizon audit |

```bash
python verification/verify_dynamical_cores.py --profile quick
```

Long profiles are separate because their cost is unsuitable for ordinary
regression testing. See `verification/README.md` for commands and gates.

## Interpreting convergence

Self-convergence measures consistency between numerical resolutions when an
exact solution is unavailable. Cross-core convergence asks whether two methods
approach one another under refinement. Neither proves observational accuracy.
A measured order should be quoted with its norm, resolution range, timestep
relation, and horizon.

The conservative FFSL tracer test is gated separately: its documented temporal
behavior is approximately first order, while smooth acoustic and internal-
gravity modes approach second order in the tested regime.

## Reproducibility checklist

Retain:

- Git commit and working-tree status;
- command line and configuration;
- JAX version, backend, device, and precision;
- core, timestep, grid, solver tolerance, and iteration limits;
- enabled physics, dissipation, and boundary treatment;
- random seeds and optimization settings;
- primary data and JSON provenance;
- renderer and figures.

The [paper suite](paper_suite.md) automates much of this record.
