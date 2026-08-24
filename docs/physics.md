# Physics and parameterizations

`PhysicsSuite` provides two extension points:

- tendency schemes return continuous rates that are accumulated with dynamics;
- update schemes apply sequential instantaneous adjustments after a step.

Registered tracer names tell the steppers which additional fields require
transport. Learned closures receive parameter pytrees explicitly, allowing
those parameters to serve as differentiable controls.

The physics directory contains warm-rain microphysics, surface processes,
turbulence, gravity-wave effects, forcing, and learned components. Availability
does not imply activation: each benchmark, experiment, or regional run chooses
which schemes to construct. Configuration switches for regional workflows are
listed in [Configuration reference](configuration.md).

Physical schemes can contain thresholds, clipping, or adjustment logic. Review
the selected scheme before assuming an objective is smoothly differentiable;
see [Differentiability and adjoints](differentiability.md).

Source: `suetes/physics/`.
