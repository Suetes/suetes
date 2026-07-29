# Dynamics-only Wreckhouse experiment

This experiment repeats the adjoint-directed Wreckhouse adverse-scenario
workflow without the Smagorinsky--Lilly SGS and McFarlane gravity-wave-drag
schemes used by the publication experiment. The original
`runs/wreckhouse_worst_case.py` and its paper outputs are not modified.

Run the experiment with:

```bash
MPLCONFIGDIR=/tmp/suetes-mpl \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python3 experiments/wreckhouse_dynamics_only/run.py \
  --output-dir output/wreckhouse_dynamics_only
```

Regenerate its dashboard without rerunning the model:

```bash
MPLCONFIGDIR=/tmp/suetes-mpl \
.venv/bin/python3 experiments/wreckhouse_dynamics_only/render.py \
  output/wreckhouse_dynamics_only/wreckhouse_dynamics_only_feb2025_plot_data.nc
```

The forward and adverse integrations use resolved dry dynamics with
`nu_h_factor=nu_div_factor=0.05`. The differentiable control model is also
dynamics-only, but uses `nu_h_factor=nu_div_factor=0.20` to stabilize its
one-hour reverse trajectory.
