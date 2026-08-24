# Two-dimensional time integration

The slice model provides semi-implicit semi-Lagrangian and split-explicit
integration machinery for idealized cases. It contains trajectory interpolation,
implicit wave coupling, and explicit substepping specialized to two dimensions.

Use the two-dimensional verification and benchmark programs to establish the
behavior of this path; three-dimensional verification does not automatically
cover its independent implementation.

Source: `suetes/slice2d/steppers.py`.
