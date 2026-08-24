# Topography blending

Regional topography combines a high-resolution GEBCO interior with terrain
consistent with ERA5 near the lateral boundary. A smooth transition prevents
the interior terrain from conflicting with the coarser external atmospheric
state where Davies relaxation is strongest.

The processor remaps and smooths terrain, then supplies a continuous function
used to construct physical grid heights. The sampled mass-point topography is
saved in the preprocessing store so simulation rebuilds exactly the same grid.

After changing smoothing, sponge depth, vertical mapping, or resolution, inspect
minimum physical layer thickness and terrain slopes.

Source: `suetes/preprocessing/topography.py`.
