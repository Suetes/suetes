# Dynamical core tendencies

`Euler3D` builds the hydrostatic background and evaluates the spatial right-hand
side for velocity, Exner pressure, virtual potential temperature, and registered
physics fields. It is not itself a complete integrator.

The tendency evaluator has a linear mode used inside the SISL GMRES operator
and an explicit mode that includes nonlinear terms, buoyancy, dissipation, and
physics. Keeping the implicit operator strictly linear is required both by the
Krylov solve and its transpose solve in reverse mode.

The background precomputation maps density and virtual potential temperature
onto velocity faces and caches layer thicknesses and compressibility factors.
This avoids rebuilding static quantities during repeated operator applications.

The physical state contains total `pi` and `th_v`, while the tendency routines
operate on perturbations relative to the hydrostatic background. See
[Scientific formulation](scientific-formulation.md) for equations and
[Model state and grid](model-state.md) for field locations.

See the generated [`Euler3D` API](api/regional3d.md#suetes.regional3d.euler.Euler3D)
for current signatures, parameters, and methods. Source:
`suetes/regional3d/euler.py`.
