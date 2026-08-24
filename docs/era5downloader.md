# ERA5 downloader

The downloader requests pressure-level atmospheric variables and single-level
surface fields from the Copernicus Climate Data Store. The requested box
includes a configurable buffer beyond the projected model domain so remapping
and lateral boundary construction have adequate coverage.

A configured CDS API account and network access are required. Download identity
is geographic rather than tied to model grid resolution, permitting reuse by
multiple numerical grids over the same physical domain.

Source: `suetes/preprocessing/era5downloader.py`.
