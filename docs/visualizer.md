# Visualization

`suetes.vis` contains reusable horizontal maps, vertical cross-sections,
Hovmöller diagrams, spectra, and comparison utilities. The legacy
`suetes.vis.visualizer` module re-exports this split interface for compatibility.

Reusable plotting primitives belong in the package; case-specific panel layout,
labels, and scientific annotations belong in the corresponding `render.py`.
Renderers consume saved artifacts and should not rerun the simulation.

When publishing a figure, retain the primary artifact and renderer as well as
the image. The artifact carries quantitative values that cannot be recovered
reliably from a raster or PDF figure.
