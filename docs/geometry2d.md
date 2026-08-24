# Two-dimensional geometry

The vertical-slice grid stores horizontal mass and velocity locations together
with terrain-following mass and vertical-velocity heights. It uses the same
vertical-transform strategies as the regional model where applicable, but has
no transverse direction or regional map projection.

The slice model is useful for inexpensive idealized dynamics, convergence
studies, and inspection of terrain-coordinate effects. It is a distinct model;
results should not be described as a three-dimensional configuration.

Source: `suetes/slice2d/geometry.py`.
