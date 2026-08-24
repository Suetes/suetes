# Scientific formulation

## Governing system

The three-dimensional core advances a dry, fully compressible, nonhydrostatic
Euler system expressed with velocity, Exner pressure, and virtual potential
temperature in terrain-following coordinates. Moisture and subgrid processes
enter through optional parameterizations and transported species.

\[
\pi = \left(\frac{p}{p_0}\right)^{R_d/c_p}.
\]

Using the hydrostatic background split described in
[Model state and grid](model-state.md), the implemented momentum terms can be
summarized schematically as

\[
\frac{D\mathbf{u}_h}{Dt}
=-c_p\theta_v\,\nabla_h\pi' + f\,\mathbf{k}\times\mathbf{u}_h
+\mathcal{M}+\mathcal{D}+\mathcal{P},
\]

\[
\frac{Dw}{Dt}
=-c_p\theta_{v,\mathrm{eff}}\frac{\partial\pi'}{\partial z}
+g\frac{\theta_v'}{\bar\theta_v}
+\mathcal{D}_w+\mathcal{P}_w.
\]

Here `M` represents projection and terrain-metric terms, `D` numerical
dissipation, and `P` optional physical tendencies. Horizontal derivatives
include the conformal map factor and terrain corrections.

The linearized Exner-pressure tendency uses reference density and virtual
potential temperature:

\[
\frac{\partial\pi'}{\partial t}
=-C_\pi\left[
m^2\left(
\partial_\xi\frac{\bar\rho\bar\theta_v u}{m}
+\partial_\eta\frac{\bar\rho\bar\theta_v v}{m}
\right)
+\partial_z(\bar\rho\bar\theta_v\dot\eta)
\right]+\mathcal{N}_\pi,
\]

\[
C_\pi=\frac{R_d}{c_{vd}}
\frac{\bar\pi}{\bar\rho\bar\theta_v}.
\]

`N_pi` collects nonlinear thermodynamic-flux contributions. Exact discrete
averages and differences, rather than this schematic continuous notation,
define the implemented model.

## Kinematic vertical velocity

Physical `w` and contravariant `eta_dot` are distinct over terrain. The latter
is the velocity through logical vertical surfaces and appears in vertical fluxes
and semi-Lagrangian departure calculations. Impermeability is imposed at the
terrain and rigid lid.

## Transport and time integration

The SISL path uses trajectory-based cubic interpolation for momentum and
thermodynamic perturbations. Density and registered tracers use dimensionally
split flux-form semi-Lagrangian transport; tracer mass is transported as
`rho * tracer` and divided by updated density afterward.

The split-explicit path computes slow advective and physical tendencies within
a three-stage RK cycle, while acoustic pressure and velocity perturbations are
advanced with smaller substeps. Vertical pressure–velocity coupling is solved
implicitly column by column.

## Dissipation and boundaries

- Fourth-order hyperdiffusion controls grid-scale variance.
- Divergence damping targets compressible divergence noise.
- An upper Rayleigh sponge reduces reflection from the rigid lid.
- Davies lateral relaxation blends the interior solution toward forcing data.

Physics tendencies can be masked in the Davies zone so they do not compete
with boundary relaxation.

## Parameterized physics

`PhysicsSuite` distinguishes continuous tendency schemes from instantaneous
state updates. Implemented modules include warm-rain microphysics, surface
processes, turbulence, gravity-wave effects, forcing, and learned closures.
A module is active only when the constructing program registers and enables it.

Continue to [Dynamical core](euler.md), [Time integration](steppers.md),
[Physics](physics.md), and [Lateral boundaries](boundaries.md).
