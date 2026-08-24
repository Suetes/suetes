# Data processor

The processor assembles ERA5 single-level and pressure-level files into the
time sequence consumed by conversion. It handles dataset alignment and the
coarsening/smoothing choices supplied by configuration.

Inspect time coordinates, units, missing values, and geographic coverage before
using a newly downloaded dataset. A successful file read does not ensure that
the requested simulation interval or buffered boundary region is complete.

Source: `suetes/preprocessing/processor.py`.
