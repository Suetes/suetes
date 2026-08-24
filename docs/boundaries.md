# Lateral boundaries

`ExternalForcing` interpolates time-indexed external states to the requested
model time. `DaviesBoundary` applies a relaxation profile across a configurable
lateral sponge zone, nudging the model toward that external state.

Relaxation is field- and stagger-aware because `u`, `v`, `w`, and mass-point
variables do not share horizontal shapes. The lower and upper vertical
boundaries are handled by the dynamical stepper; the Davies mechanism is a
lateral open-boundary treatment.

Physics tendencies may be multiplied by an interior mask so parameterizations
are inactive within the relaxation zone. This avoids simultaneous forcing by
physics and boundary nudging in the same cells.

Source: `suetes/regional3d/boundaries.py`. Boundary data construction is covered
under [Data and preprocessing](era2suetes.md).
