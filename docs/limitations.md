# Scope and limitations

Suêtes is research software for numerical experimentation with differentiable
regional atmospheric dynamics. It is not an operational forecast system and
must not be used as the sole basis for safety-critical or public-warning
decisions.

## Scientific scope

- Validation consists of the repository's regression, convergence, benchmark,
  and adjoint studies. It does not establish operational forecast skill.
- Physical parameterizations are selective and configuration-dependent. The
  presence of a scheme in the source tree does not mean it is enabled or
  validated in every experiment.
- Real-data workflows depend on ERA5 initial and boundary conditions and on the
  assumptions made during remapping and topography blending.
- Plot artifacts contain fields needed for published diagnostics and are not
  necessarily restart-complete model states.

## Differentiation scope

- A valid forward integration does not guarantee an accurate gradient.
- SISL departure indices are treated as fixed in reverse mode; gradients do not
  include the flow dependence of diagnosed departure locations.
- Limiters, clipping, conditional parameterizations, and finite solver
  tolerances can introduce nonsmoothness or gradient error.
- Taylor tests establish local consistency for a particular control, direction,
  state, configuration, and horizon.

## Computational scope

- Large three-dimensional simulations can require substantial GPU memory and
  compilation time.
- Results can depend on JAX version, backend, precision, device, and solver
  configuration. Preserve these with published outputs.
- CPU support is exercised by the quick start and regression suite, but this
  does not imply practical CPU performance for publication-scale cases.

## Interfaces and stability

The project is currently pre-1.0. Public interfaces, configuration fields, and
artifact schemas may still change. Tagged paper releases should be treated as
immutable reproducibility snapshots; development should continue on later
versions.
