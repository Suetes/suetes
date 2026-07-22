"""
Incremental NetCDF output for suetes simulations.

Writes one time-slice per chunk to a NetCDF4 file with an unlimited ``time``
dimension, so the full 4D output never accumulates in host (or GPU) memory and
the result is directly readable with xarray / ncview. Staggered prognostics keep
their native dimensions (``x``/``xc``, ``y``/``yc``, ``z``/``zc``); 2D fields use
``(x, y)``. 2D ``lat``/``lon`` (on mass points) are written as coordinates for
geographic context.

Usage (matches the runner's per-hour snapshot cadence)::

    w = SimulationOutputWriter(path, grid, save_keys, attrs={...})
    w.write_snapshot(initial_subset, t_seconds=0.0)   # t = 0
    ...                                                # inside the run loop:
    w.write_snapshot(hourly_subset)                   # t auto-increments by 3600 s
    w.close()
    snaps = NetcdfSnapshots(path, save_keys)           # lazy, list-like, for plotting
    
    
TODO: This should be more flexible and broken down into different components. 
      The problem is more so that we want to free up the batches in GPU...
      this can be done with a subprocess that transforms the output into
      a host xarray and then xarray has the capability to save and so we 
      won't need this whole class. This works for now...
"""

import os

import numpy as np
import netCDF4


class SimulationOutputWriter:
    """Append-per-chunk NetCDF4 writer (constant memory in run length)."""

    def __init__(self, path, grid, keys, attrs=None, dt_hours=1.0, ref_date=None):
        self.path = path
        self.keys = list(keys)
        self._t = 0
        self._dt_seconds = float(dt_hours) * 3600.0
        nx, ny, nz = grid.nx, grid.ny, grid.nz
        self._sizes = (nx, ny, nz)

        ds = netCDF4.Dataset(path, "w", format="NETCDF4")
        ds.createDimension("time", None)             # unlimited
        ds.createDimension("x", nx);   ds.createDimension("xc", nx + 1)
        ds.createDimension("y", ny);   ds.createDimension("yc", ny + 1)
        ds.createDimension("z", nz);   ds.createDimension("zc", nz + 1)

        self._time_var = ds.createVariable("time", "f8", ("time",))
        self._time_var.units = (f"seconds since {ref_date} 00:00:00" if ref_date
                                else "seconds")
        self._time_var.long_name = "time"

        # 1D projection-plane axes (exact / lossless) + level index.
        def _coord1d(name, dim, data, units, long_name, std=None, axis=None, positive=None):
            v = ds.createVariable(name, "f8", (dim,)); v[:] = np.asarray(data)
            v.units = units; v.long_name = long_name
            if std:
                v.standard_name = std
            if axis:
                v.axis = axis
            if positive:
                v.positive = positive
        _coord1d("x", "x", grid.x_m, "m", "projection x (mass)", "projection_x_coordinate", "X")
        _coord1d("y", "y", grid.y_m, "m", "projection y (mass)", "projection_y_coordinate", "Y")
        _coord1d("xc", "xc", grid.x_c, "m", "projection x (u faces)", "projection_x_coordinate", "X")
        _coord1d("yc", "yc", grid.y_c, "m", "projection y (v faces)", "projection_y_coordinate", "Y")
        _coord1d("z", "z", np.arange(nz), "1", "model level (mass)", axis="Z", positive="up")
        _coord1d("zc", "zc", np.arange(nz + 1), "1", "model level (w)", axis="Z", positive="up")

        Xi, Yi = np.meshgrid(np.asarray(grid.x_m), np.asarray(grid.y_m), indexing="ij")
        lat, lon = grid.proj.get_lat_lon(Xi, Yi)        # (nx, ny) -> store as (y, x)
        latv = ds.createVariable("lat", "f4", ("y", "x")); latv[:] = np.asarray(lat).T
        lonv = ds.createVariable("lon", "f4", ("y", "x")); lonv[:] = np.asarray(lon).T
        latv.units = "degrees_north"; latv.standard_name = "latitude"; latv.long_name = "latitude"
        lonv.units = "degrees_east"; lonv.standard_name = "longitude"; lonv.long_name = "longitude"

        proj = grid.proj
        crs = ds.createVariable("crs", "i4")
        crs.grid_mapping_name = "stereographic"
        crs.latitude_of_projection_origin = float(np.degrees(np.asarray(proj.phi_c)))
        crs.longitude_of_projection_origin = float(np.degrees(np.asarray(proj.lam_c)))
        crs.scale_factor_at_projection_origin = 1.0
        crs.false_easting = 0.0
        crs.false_northing = 0.0
        crs.earth_radius = float(proj.R)
        ds.Conventions = "CF-1.7"

        if attrs:
            for k, v in attrs.items():
                setattr(ds, k, v)

        self._ds = ds
        self._vars = {}   # created lazily once shapes are known (first snapshot)

    def _dims_for_shape(self, shape):
        nx, ny, nz = self._sizes
        axis_maps = [{nx: "x", nx + 1: "xc"}, {ny: "y", ny + 1: "yc"},
                     {nz: "z", nz + 1: "zc"}]
        dims = []
        for ax, s in enumerate(shape):
            m = axis_maps[ax]
            if s not in m:
                raise ValueError(f"output field axis {ax} size {s} matches no grid "
                                 f"dimension {set(m)}")
            dims.append(m[s])
        return tuple(reversed(dims))

    def write_snapshot(self, state, t_seconds=None):
        """Write one time-slice. ``state`` is a dict of host arrays (x,y[,z])."""
        i = self._t
        for k in self.keys:
            if k not in state:
                continue
            arr = np.asarray(state[k])
            v = self._vars.get(k)
            if v is None:
                spatial = self._dims_for_shape(arr.shape)   # (z,y,x) / (z,y,xc) / (y,x)
                dims = ("time",) + spatial
                v = self._ds.createVariable(k, "f4", dims, zlib=True, complevel=1)
                if len(spatial) >= 2 and spatial[-2] == "y" and spatial[-1] == "x":
                    v.coordinates = "lat lon"
                if len(spatial) >= 2:                 # any horizontal field
                    v.grid_mapping = "crs"
                self._vars[k] = v
            # reverse spatial axes to match the (z,y,x) declaration: (x,y,z)->(z,y,x)
            v[i, ...] = np.transpose(arr, tuple(reversed(range(arr.ndim))))
        self._time_var[i] = (i * self._dt_seconds if t_seconds is None else t_seconds)
        self._t += 1
        self._ds.sync()    # flush this chunk to disk; do not hold in memory

    def close(self):
        self._ds.close()


_PROG_KEYS = ("u", "v", "w", "th_v", "pi", "q", "q_c", "q_r")
_PHYS_KEYS = ("sw_sfc", "rain_acc", "conv_acc", "precip", "precip_conv",
              "cloud_fraction", "total_cloud_cover", "rad_th_tend")
GROUPS = {"prog": _PROG_KEYS, "phys": _PHYS_KEYS}


class DailyGroupedWriter:
    """Routes hourly snapshots to per-day, per-group NetCDF files.

    Instead of one monolithic file, writes ``<run>_<group>_<YYYYMMDD>.nc`` -- a
    separate file for each simulated day and each variable group (prognostic vs
    physics). Each daily file holds that day's (up to 24) hourly steps with its
    own date as the time reference.

    ``make_writer(path, keys, ref_date)`` returns a per-file writer exposing
    ``write_snapshot(state, t_seconds)`` and ``close()`` (e.g. SimulationOutputWriter
    or LatLonOutputWriter).
    """

    def __init__(self, out_dir, run_name, make_writer, year, month, start_day,
                 groups=GROUPS, label=""):
        self.out_dir = out_dir
        self.run_name = run_name
        self.make_writer = make_writer
        self.groups = groups
        self.label = label
        self._y, self._m, self._d0 = int(year), int(month), int(start_day)
        self._writers = {}     # (group, day_idx) -> writer
        os.makedirs(out_dir, exist_ok=True)

    def _date(self, day_idx):
        d = (np.datetime64(f"{self._y:04d}-{self._m:02d}-{self._d0:02d}")
             + np.timedelta64(day_idx, "D"))
        return str(d)   # 'YYYY-MM-DD'

    def write(self, hour, state):
        day_idx, hour_in_day = divmod(int(hour), 24)
        # Close (flush + unlock) any earlier day's files so finished days are
        # immediately readable in ncview/xarray while the run continues.
        for key in [k for k in self._writers if k[1] < day_idx]:
            self._writers.pop(key).close()
        for gname, gkeys in self.groups.items():
            sub = {k: np.asarray(v) for k, v in state.items() if k in gkeys}
            if not sub:
                continue
            w = self._writers.get((gname, day_idx))
            if w is None:
                date = self._date(day_idx)
                fname = f"{self.run_name}_{gname}{self.label}_{date.replace('-', '')}.nc"
                w = self.make_writer(os.path.join(self.out_dir, fname),
                                     list(sub.keys()), date)
                self._writers[(gname, day_idx)] = w
            w.write_snapshot(sub, t_seconds=hour_in_day * 3600.0)

    def close(self):
        for w in self._writers.values():
            w.close()


class DailyNetcdfSnapshots:
    """Read a run's daily prog/phys output files back as a per-hour sequence.

    Globs ``<run>_prog_*.nc`` + ``<run>_phys_*.nc`` in ``rundir``, concatenates
    across days (by time) and merges the groups, exposing the same list-like
    interface as NetcdfSnapshots (``len`` / indexing) -- each item is a dict of
    native (x,y,z)/(x,y) arrays (transposed back from the stored (z,y,x)/(y,x)).
    """

    def __init__(self, rundir, run_name, keys=None):
        import glob
        import xarray as xr
        prog = sorted(glob.glob(os.path.join(rundir, f"{run_name}_prog_*.nc")))
        phys = sorted(glob.glob(os.path.join(rundir, f"{run_name}_phys_*.nc")))
        parts = []
        for grp in (prog, phys):
            if grp:
                parts.append(xr.open_mfdataset(grp, combine="by_coords", data_vars="minimal"))
        if not parts:
            raise FileNotFoundError(f"no daily prog/phys files for {run_name} in {rundir}")
        self._ds = xr.merge(parts)
        self.n = self._ds.sizes["time"]
        avail = [v for v in self._ds.data_vars if v not in ("lat", "lon", "crs")]
        self.keys = [k for k in (keys or avail) if k in avail]

    def __len__(self):
        return self.n

    def __getitem__(self, h):
        if h < 0:
            h += self.n
        out = {}
        for k in self.keys:
            a = np.asarray(self._ds[k].isel(time=h).values)   # (z,y,x) or (y,x)
            out[k] = np.transpose(a, tuple(reversed(range(a.ndim))))  # -> (x,y,z)/(x,y)
        return out

    def __iter__(self):
        for h in range(self.n):
            yield self[h]

    def close(self):
        self._ds.close()


class NetcdfSnapshots:
    """Lazy, list-like view over a written output file.

    Drop-in for the old in-RAM ``snapshots`` list used by the plotting code:
    supports ``len()``, integer indexing (incl. negative), and iteration, reading
    each time-slice from disk on demand (constant memory).
    """

    def __init__(self, path, keys):
        self._ds = netCDF4.Dataset(path, "r")
        self.n = self._ds.dimensions["time"].size
        self.keys = [k for k in keys if k in self._ds.variables]

    def __len__(self):
        return self.n

    def __getitem__(self, h):
        if h < 0:
            h += self.n
        out = {}
        for k in self.keys:
            a = np.asarray(self._ds.variables[k][h])
            out[k] = np.transpose(a, tuple(reversed(range(a.ndim))))
        return out

    def __iter__(self):
        for h in range(self.n):
            yield self[h]

    def close(self):
        self._ds.close()
