# Model state and grid

## Grid staggering

The three-dimensional model uses an Arakawa C grid horizontally and Lorenz
staggering vertically.

| Location | Logical position | Principal variables |
|---|---|---|
| `m` | cell center `(i, j, k)` | Exner pressure, virtual potential temperature, density, moisture, tracers |
| `u` | x face `(i+1/2, j, k)` | x-directed velocity |
| `v` | y face `(i, j+1/2, k)` | y-directed velocity |
| `w` | vertical face `(i, j, k+1/2)` | physical and contravariant vertical velocity |

The operator API requires both a source and destination location. This makes
interpolation and differencing choices explicit.

## Primary state fields

The steppers expect a dictionary-like JAX pytree:

| Key | Meaning | Location |
|---|---|---|
| `u`, `v` | Horizontal grid-relative wind components | `u`, `v` |
| `w` | Physical vertical velocity | `w` |
| `eta_dot` | Contravariant vertical index velocity | `w` |
| `pi` | Total Exner pressure | `m` |
| `th_v` | Total virtual potential temperature | `m` |
| `rho` | Density used by conservative transport | `m` |

Registered moisture species and passive tracers add mass-point fields. The SISL
stepper also stores previous velocities, tendencies, and a first-step flag for
second-order estimates. Those history keys are integration state rather than
new atmospheric prognostic variables.

## Background and perturbation split

\[
\pi = \bar\pi + \pi', \qquad
\theta_v = \bar\theta_v + \theta_v'.
\]

The time-independent reference satisfies discrete hydrostatic balance. For an
idealized case, `Euler3D` constructs an analytical profile from a specified
Brunt–Väisälä frequency. For an ERA5-initialized case, it uses the horizontal
mean virtual-potential-temperature profile and anchors Exner pressure at the
domain top. Pressure is integrated downward using physical layer thicknesses
so the background pressure gradient cancels gravity in a resting column.

The public state stores total `pi` and `th_v`; each stepper forms perturbations
before calling the tendency operator.

## Map projection and metrics

The regional grid uses an oblique stereographic conformal projection centered
at the configured latitude and longitude. Its isotropic map factor is

\[
m(x,y)=1+\frac{x^2+y^2}{4R^2}.
\]

Latitude, longitude, Coriolis parameter, projection convergence, and map-factor
derivatives are evaluated at appropriate staggers. Geographic input winds are
rotated into the grid-relative x/y basis during preprocessing.

## Terrain-following height

Logical height `zeta` is mapped to physical height `z`. Implemented transform
families include Gal–Chen, hybrid sigma-z, simplified and full SLEVE variants,
stretched SLEVE variants, and the integral neural coordinate used by NEUVE.
All impose physical terrain at the lower boundary and the fixed model lid at
the upper boundary.

Grid construction rejects negative physical layer thicknesses outside traced
calculations. Users should still inspect minimum layer thickness and metric
distortion: an untangled grid may nevertheless be numerically poor.

See [Geometry and grid](geometry.md) and [Discrete operators](operators.md).
