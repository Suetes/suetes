# Data and preprocessing

The regional pipeline converts external geographic data into a state and
boundary sequence consistent with the model projection, staggering, vertical
coordinate, and thermodynamic variables.

## Stages

1. `era5downloader` requests pressure-level and single-level ERA5 fields over a
   buffered geographic domain.
2. `processor` combines and normalizes the downloaded datasets.
3. `topography` blends high-resolution interior terrain with ERA5 terrain near
   the lateral boundary and constructs a continuous terrain function.
4. `era2suetes` rotates winds, remaps horizontal fields, reconstructs the
   vertical thermodynamic state, and places fields on model staggers.
5. `bc_store` writes chunked time-dependent boundary data and static topography.

Preprocessing and simulation share configuration resolution and grid-building
helpers. The simulation can reconstruct its grid from saved topography without
reopening the original ERA5 and GEBCO sources.

Boundary stores are scientific inputs. Preserve their creation configuration
and source-data identity alongside a regional experiment.
