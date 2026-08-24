# Discrete operators

`CGridOperator3D` supplies finite differences and two-point averages between
named C-grid locations. Callers state `from_loc` and `to_loc`, making the
stagger transition part of the operation rather than an implicit shape guess.

Horizontal derivatives use logical spacing and are combined with projection
map factors by the dynamical core. Vertical logical differences are converted
to physical derivatives with stored terrain-following layer thicknesses.
Pressure gradients additionally subtract the terrain-slope contribution so a
horizontal derivative is taken at constant physical height.

The module also provides tensor-product interpolation used by semi-Lagrangian
transport. Boundary behavior is part of each operator and should be checked
when adding a new variable or stagger; interior order alone does not determine
the full-domain discretization.

See the generated [operator API](api/regional3d.md#suetes.regional3d.operators)
for current signatures. Source: `suetes/regional3d/operators.py`. Consistency
tests are described in [Verification and evidence](verification.md).
