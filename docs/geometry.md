# Vertical coordinates and geometry

`suetes.regional3d.geometry` constructs the oblique-stereographic regional grid,
all Arakawa C-grid locations, geographic coordinates, map factors, Coriolis
parameters, terrain-following physical heights, and layer metrics.

For logical height `zeta`, Gal–Chen uses

\[
z=h+\zeta\frac{L_z-h}{L_z}.
\]

The simplified SLEVE transform uses

\[
z=\zeta+h\left[
\frac{\sinh((L_z-\zeta)/s)}{\sinh(L_z/s)}
\right]^n.
\]

Stretched SLEVE first maps uniform logical levels through

\[
\zeta_s=L_z\frac{\exp(\kappa\zeta/L_z)-1}{\exp(\kappa)-1}
\]

and evaluates the SLEVE decay at `zeta_s`. The general SLEVE implementation
splits terrain into large- and small-scale components with distinct decay
heights. The integral neural transform instead predicts a bounded positive
layer-density field, integrates it to a normalized cumulative coordinate `S`,
and uses `z = zeta + h(1-S)`.

The grid stores physical heights at mass and vertical-velocity levels, physical
layer thicknesses, terrain slopes, map-factor derivatives, and projection
quantities at every required stagger. See [Model state and grid](model-state.md)
for the location convention.

Source: `suetes/regional3d/geometry.py` and `suetes/shared/transforms.py`.
