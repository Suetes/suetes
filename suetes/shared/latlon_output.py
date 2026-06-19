"""
Regular lat/lon NetCDF output for suetes (ERA5-style, ncview-friendly).

suetes runs on an oblique-stereographic *projected* grid, so true lat/lon are 2D
(`lat(y,x)`, `lon(y,x)`) over Cartesian x/y. ncview cannot use 2D curvilinear
coordinates as map axes (it will not reproject them), so a native file shows
index/projection axes, not a geographic map. To get a real lon/lat map per level
in ncview, the data must be on a **regular lat/lon grid with 1D
longitude/latitude/level coordinate variables** -- the structure ERA5 uses.

This module interpolates each model field from the projected grid onto a regular
lat/lon target grid and streams it (one time-slice per chunk) to a CF/ERA5-style
NetCDF: ``var(time, level, latitude, longitude)`` with 1D coordinate variables.

The regrid uses the closed-form inverse of the projection (lat/lon -> planar x/y
-> fractional model index) + bilinear ``map_coordinates`` -- precomputed once, so
each field/level is a cheap gather. Points outside the domain become NaN.
"""

import numpy as np
import netCDF4
from scipy.ndimage import map_coordinates


def latlon_to_xy(proj, lat_deg, lon_deg):
    """Inverse of ObliqueStereographic.get_lat_lon: (lat,lon) [deg] -> planar (x,y).

    Closed-form spherical oblique-stereographic inverse matching the forward in
    geometry.py (diameter form rho = 2R tan(c/2), scale 1 at the origin).
    """
    phi = np.radians(np.asarray(lat_deg, dtype=float))
    lam = np.radians(np.asarray(lon_deg, dtype=float))
    phi_c = float(proj.phi_c); lam_c = float(proj.lam_c); R = float(proj.R)
    dlam = lam - lam_c
    sinpc, cospc = np.sin(phi_c), np.cos(phi_c)
    k = 2.0 * R / (1.0 + sinpc * np.sin(phi) + cospc * np.cos(phi) * np.cos(dlam))
    x = k * np.cos(phi) * np.sin(dlam)
    y = k * (cospc * np.sin(phi) - sinpc * np.cos(phi) * np.cos(dlam))
    return x, y


class LatLonRegridder:
    """Precomputes the projected-grid -> regular-lat/lon-grid index mapping."""

    def __init__(self, grid, nlon=None, nlat=None, pad_frac=0.0):
        self.grid = grid
        x_m = np.asarray(grid.x_m, dtype=float)
        y_m = np.asarray(grid.y_m, dtype=float)
        self.x0, self.dx = x_m[0], (x_m[1] - x_m[0])
        self.y0, self.dy = y_m[0], (y_m[1] - y_m[0])

        # Domain lat/lon extent from the projected mass grid corners/edges.
        Xi, Yi = np.meshgrid(x_m, y_m, indexing="ij")
        lat2d, lon2d = (np.asarray(a) for a in grid.proj.get_lat_lon(Xi, Yi))
        lat_min, lat_max = float(lat2d.min()), float(lat2d.max())
        lon_min, lon_max = float(lon2d.min()), float(lon2d.max())
        dlat = (lat_max - lat_min) * pad_frac
        dlon = (lon_max - lon_min) * pad_frac
        nlon = int(nlon or grid.nx)
        nlat = int(nlat or grid.ny)
        # Ascending 1D axes (ncview/CF want monotonic coordinate variables).
        self.lons = np.linspace(lon_min - dlon, lon_max + dlon, nlon)
        self.lats = np.linspace(lat_min - dlat, lat_max + dlat, nlat)

        # For each target (lat,lon), the fractional model index (i along x, j along y).
        LON, LAT = np.meshgrid(self.lons, self.lats)        # (nlat, nlon)
        xt, yt = latlon_to_xy(grid.proj, LAT, LON)
        self._ifrac = (xt - self.x0) / self.dx              # (nlat, nlon)
        self._jfrac = (yt - self.y0) / self.dy
        # Mark targets that fall outside the projected domain -> NaN later.
        self._oob = ((self._ifrac < 0) | (self._ifrac > grid.nx - 1) |
                     (self._jfrac < 0) | (self._jfrac > grid.ny - 1))
        self._coords = np.stack([self._ifrac.ravel(), self._jfrac.ravel()], axis=0)

    def regrid(self, field_xy):
        """Bilinear interpolate a mass-point field (nx,ny) -> (nlat,nlon)."""
        out = map_coordinates(np.asarray(field_xy, dtype=float), self._coords,
                              order=1, mode="constant", cval=np.nan)
        out = out.reshape(self.lats.size, self.lons.size)
        out[self._oob] = np.nan
        return out.astype(np.float32)


def destagger_to_mass(name, arr, grid):
    """Average a staggered field onto mass points so it can share the lat/lon grid.

    u(nx+1,ny,nz)->(nx,ny,nz); v(nx,ny+1,nz)->(nx,ny,nz); w(nx,ny,nz+1)->(nx,ny,nz).
    Mass/2D fields pass through unchanged.
    """
    a = np.asarray(arr)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    if a.shape[:1] == (nx + 1,):
        return 0.5 * (a[:-1] + a[1:])
    if a.ndim >= 2 and a.shape[1] == ny + 1:
        return 0.5 * (a[:, :-1] + a[:, 1:])
    if a.ndim == 3 and a.shape[2] == nz + 1:
        return 0.5 * (a[:, :, :-1] + a[:, :, 1:])
    return a


class LatLonOutputWriter:
    """ERA5-style streaming writer on a regular lat/lon grid (ncview-ready).

    Layout: var(time, level, latitude, longitude) for 3D fields, var(time,
    latitude, longitude) for 2D. 1D coordinate variables latitude/longitude/level
    + a CF time axis. One time-slice written per ``write_snapshot`` call.
    """

    def __init__(self, path, grid, keys, ref_date=None, attrs=None, dt_hours=1.0,
                 nlon=None, nlat=None):
        self.keys = list(keys)
        self.grid = grid
        self._t = 0
        self._dt = float(dt_hours) * 3600.0
        self.rg = LatLonRegridder(grid, nlon=nlon, nlat=nlat)
        nz = grid.nz

        ds = netCDF4.Dataset(path, "w", format="NETCDF4")
        ds.Conventions = "CF-1.7"
        ds.createDimension("time", None)
        ds.createDimension("level", nz)
        ds.createDimension("latitude", self.rg.lats.size)
        ds.createDimension("longitude", self.rg.lons.size)

        tv = ds.createVariable("time", "f8", ("time",))
        tv.units = f"seconds since {ref_date} 00:00:00" if ref_date else "seconds"
        tv.long_name = "time"; tv.standard_name = "time"; tv.axis = "T"
        if ref_date:
            tv.calendar = "proleptic_gregorian"
        self._tv = tv

        lev = ds.createVariable("level", "i4", ("level",)); lev[:] = np.arange(nz)
        lev.long_name = "model level"; lev.axis = "Z"; lev.positive = "up"
        latv = ds.createVariable("latitude", "f8", ("latitude",)); latv[:] = self.rg.lats
        latv.units = "degrees_north"; latv.standard_name = "latitude"; latv.axis = "Y"
        lonv = ds.createVariable("longitude", "f8", ("longitude",)); lonv[:] = self.rg.lons
        lonv.units = "degrees_east"; lonv.standard_name = "longitude"; lonv.axis = "X"

        if attrs:
            for k, v in attrs.items():
                setattr(ds, k, v)
        self._ds = ds
        self._vars = {}

    def write_snapshot(self, state, t_seconds=None):
        i = self._t
        for k in self.keys:
            if k not in state:
                continue
            a = destagger_to_mass(k, state[k], self.grid)   # -> (nx,ny[,nz])
            is3d = (a.ndim == 3)
            v = self._vars.get(k)
            if v is None:
                dims = ("time", "level", "latitude", "longitude") if is3d else \
                       ("time", "latitude", "longitude")
                v = self._ds.createVariable(k, "f4", dims, zlib=True, complevel=1)
                self._vars[k] = v
            if is3d:
                nz = a.shape[2]
                out = np.empty((nz, self.rg.lats.size, self.rg.lons.size), np.float32)
                for kz in range(nz):
                    out[kz] = self.rg.regrid(a[:, :, kz])
                v[i, ...] = out
            else:
                v[i, ...] = self.rg.regrid(a)
        self._tv[i] = i * self._dt if t_seconds is None else t_seconds
        self._t += 1
        self._ds.sync()

    def close(self):
        self._ds.close()
