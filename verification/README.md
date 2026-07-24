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
- `long_horizon_core_audit.py`: opt-in 1000 s, three-grid bubble refinement
  audit with quantitative gates and CSV/JSON results.
- `smooth_mode_core_convergence.py`: fixed-grid timestep refinement for a
  translating passive tracer, standing internal-gravity mode, and standing
  acoustic mode.
- `sisl_component_convergence.py`: interpolation, trajectory, implicit-solver,
  operator, and advection convergence.
- `gmres_adjoint_convergence.py`: adjoint sensitivity to the GMRES solve.
- `verify_dynamical_cores.py`: isolated-process orchestration of the active
  verification ladder.

Superseded short integrated-MMS diagnostics were removed after the smooth-mode,
equation-tendency, and short-bubble tests replaced them with validated
asymptotic convergence checks.

Run the quick verification ladder from the repository root with:

```bash
python verification/verify_dynamical_cores.py --profile quick
```

The temporal and 24 s bubble stages now write data-only JSON artifacts:

```text
output/verification/temporal_core_convergence/default/data/summary.json
output/verification/rising_bubble_core_convergence/t24/data/summary.json
```

The long-horizon gate is deliberately excluded from normal pytest and the
quick/full profiles because it performs six expensive integrations:

```bash
python verification/verify_dynamical_cores.py --profile long
```

It checks that cross-core and per-core refinement errors decrease, requires
early cross-core convergence order of at least 1.25, and bounds finest-grid
centroid, positive-anomaly integral, and mass differences. The phase-sensitive
gridpoint peak is reported but is not a pass/fail gate. Results are written to
`output/verification/long_horizon_core_audit/long-horizon/data`.

To analyze existing artifacts without rerunning the model:

```bash
python verification/long_horizon_core_audit.py \
  --artifact output/benchmarks/rising_bubble_3d/audit-dx100 \
  --artifact output/benchmarks/rising_bubble_3d/smoke \
  --artifact output/benchmarks/rising_bubble_3d/audit-dx25 \
  --name existing-artifacts
```

Run the isolated smooth-mode suite with:

```bash
python verification/verify_dynamical_cores.py --profile smooth
```

The acoustic and internal-gravity modes require at least order 1.7 and
currently measure approximately second order for both cores. The conservative
FFSL tracer transport is gated separately at order 0.9: it currently measures
approximately first order under timestep refinement for both cores. This is
reported explicitly rather than being presented as second-order core evidence.

## Rendering convergence artifacts

Verification and rendering are separate. After producing a smooth-mode
artifact, render one self-convergence figure per case with:

```bash
python verification/render_smooth_mode_convergence.py \
  output/verification/smooth_mode_core_convergence/default
```

Render the long-horizon bubble self-convergence, cross-core refinement, and
cross-core error evolution with:

```bash
python verification/render_long_horizon_core_audit.py \
  output/verification/long_horizon_core_audit/long-horizon
```

Both renderers also accept an explicit `summary.json`/`metrics.csv`, and
`--output-dir` can override the standard sibling `figures/` directory.

The 24 s rising-bubble verification remains the preferred formal space-time
order figure because the solution is still smooth. Render its self-convergence
and cross-core convergence from the standard artifact with:

```bash
python verification/render_short_bubble_convergence.py
```

This writes explicitly time-labelled figures under
`output/verification/rising_bubble_core_convergence/t24/figures`.
For historical results generated before the JSON migration, the renderer also
accepts `output/verification/bubble.log`.

Render the fixed-grid temporal convergence artifact with:

```bash
python verification/render_temporal_core_convergence.py
```

The rising-bubble convergence horizon is configurable.  For the publication
$t=100$ s experiment, run and render:

```bash
python verification/rising_bubble_core_convergence.py --t-end 100 --name t100
python verification/render_short_bubble_convergence.py \
  output/verification/rising_bubble_core_convergence/t100
```

Use `--t-end 50 --name t50` for the shorter fallback study.

The GMRES adjoint conditioning study follows the same two-step pattern:

```bash
python verification/gmres_adjoint_convergence.py
python verification/render_gmres_adjoint_convergence.py
```

Verify the discrete adjoints of both cores with directional Taylor tests and
measure fixed-grid temporal self- and cross-core gradient convergence with:

```bash
python verification/adjoint_gradient_convergence.py
python verification/render_adjoint_gradient_convergence.py
```
