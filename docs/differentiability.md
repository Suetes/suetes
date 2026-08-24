# Differentiability and adjoints

Suêtes uses JAX transformations to differentiate scalar objectives through model
integration. Differentiability is a property of a particular configured path,
not an unconditional property of every repository workflow.

## Differentiable integration

`Simulation.run_differentiable` expresses integration with `jax.lax.scan` and
keeps state in JAX pytrees. Chunks are rematerialized with `jax.checkpoint`,
trading extra forward computation for lower reverse-pass storage. Individual
steppers can also checkpoint costly tendency and advection work.

The SISL implicit solve uses a linear-solve adjoint rather than differentiating
through every Krylov iteration. Solver tolerance still matters: inaccurate
primal or transpose solves can contaminate gradients.

## Intentional gradient boundaries

SISL departure indices are wrapped in `jax.lax.stop_gradient`. Gradients pass
through interpolation at diagnosed departure points, but not through the
dependence of those points on the flow. This deliberate approximation matters
for trajectory-sensitive controls.

Limiters, clipping, conditional physics updates, file I/O, host callbacks, and
integer or Boolean choices may be nondifferentiable, piecewise differentiable,
or outside a traced function. Select controls and objectives so the intended
path remains in JAX.

## Demonstrated controls

- initial thermal and moisture perturbations;
- tracer-source location and amplitude;
- terrain-basis coefficients;
- upstream perturbations;
- neural vertical-coordinate parameters.

These are executable patterns, not a guarantee that every field or option is a
suitable control.

## Gradient validation

For control `x`, direction `v`, and scalar objective `J`, compare

\[
R_0(h)=|J(x+hv)-J(x)|
\]

with

\[
R_1(h)=|J(x+hv)-J(x)-h\nabla J(x)\cdot v|.
\]

For a smooth, correctly differentiated objective, `R_0` is first order and
`R_1` second order before roundoff or solver error dominates. Validate again
after changing the objective, control, timestep, solver tolerance, limiter,
precision, physics, or trajectory length.

A successful Taylor test remains local to one state, direction, configuration,
and horizon. Report those conditions, checkpointing, solver tolerances, and
the stopped departure-point dependence with sensitivity results. See
[Verification and evidence](verification.md).
