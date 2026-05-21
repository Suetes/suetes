"""
Dinosaur to Suetes Data Bridge.

Reads dinosaur global-model output (saved by ``run_dinosaur_jw.py`` as a
netCDF) and provides it to Suetes via the same interface as
``ERA5Processor``. This enables a perfect-model lateral-boundary-condition
test of the Suetes regional dynamical core: a dinosaur run on the sphere
drives a Suetes run on a Cartesian sub-domain, and we check that Suetes'
interior matches the corresponding region of the dinosaur reference.

Expected netCDF schema (the runner script produces this):

  Dimensions:
    time      (T)
    level     (L) on standard pressure levels (units: hPa)
    latitude  (Y)
    longitude (X)
  Pressure-level variables (T, L, Y, X):
    u, v   [m s-1]
    t      [K]
    q      [kg kg-1]    (zeros for the dry baroclinic instability test)
    z      [m**2 s-2]   geopotential
    w      [Pa s-1]     omega = dp/dt (zeros for the dry test; OK to keep zero)
  Single-level variables (T, Y, X):
    sp     [Pa]         surface pressure
    skt    [K]          skin temperature (lowest-level T is used for J-W)
    lsm    [0..1]       land-sea mask (zeros for ocean-everywhere idealized test)
  Coordinates:
    level     [hPa]
    latitude  [deg]
    longitude [deg]
    time      [hour]

The variable names are chosen to mirror ERA5's pressure-level naming (u, v,
t, q, z, w) and ERA5's single-level naming (sp, skt, lsm), so downstream
code reads identically from either source.
"""

import jax.numpy as jnp
import numpy as np
import xarray as xr


class DinosaurProcessor:
    """Drop-in replacement for ``ERA5Processor``.

    Matches the ``get_stitched_state(time_idx)`` interface exactly so the
    existing ``BoundaryProcessor`` consumes its output without modification.
    """

    def __init__(self, nc_path):
        """
        Args:
            nc_path (str): netCDF file produced by ``run_dinosaur_jw.py``.
        """
        self.ds = xr.open_dataset(nc_path)

    def get_stitched_state(self, time_idx=0):
        """
        Extract arrays, broadcast pressure, and stitch the single-level
        surface row to the bottom of the pressure-level fields.

        Args:
            time_idx (int): timestep index in the dinosaur output.

        Returns:
            dict: stitched state in the same layout as
            ``ERA5Processor.get_stitched_state``, with two extra fields
            ('land_sea_mask', 'skin_temperature') for the Phase 2 surface
            blending.
        """
        snap = self.ds.isel(time=time_idx)

        # Pressure-level fields, shape (level, lat, lon).
        u_pl = np.asarray(snap['u'].values)
        v_pl = np.asarray(snap['v'].values)
        t_pl = np.asarray(snap['t'].values)
        q_pl = np.asarray(snap['q'].values)
        z_pl = np.asarray(snap['z'].values)
        w_pl = np.asarray(snap['w'].values)

        # Broadcast the 1D pressure coordinate (hPa -> Pa) into 3D.
        p_1d = np.asarray(self.ds['level'].values) * 100.0
        p_pl = np.broadcast_to(p_1d[:, None, None], t_pl.shape)

        # Single-level "surface" fields, shape (lat, lon).
        sp_2d  = np.asarray(snap['sp'].values)
        skt_2d = np.asarray(snap['skt'].values)
        lsm_2d = np.asarray(self.ds['lsm'].values)

        # Surface row to stitch to the bottom of each pressure-level field.
        # The bridge interprets the last (highest-pressure) row as the
        # near-surface state. For a flat-earth J-W test, surface geopotential
        # is zero; surface wind is the lowest pressure-level wind; q and
        # omega are zero in dry initial conditions.
        z_surf = np.zeros_like(sp_2d)[None, :, :]
        p_surf = sp_2d[None, :, :]
        t_surf = skt_2d[None, :, :]
        u_surf = u_pl[-1:, :, :].copy()
        v_surf = v_pl[-1:, :, :].copy()
        q_surf = np.zeros_like(p_surf)
        w_surf = np.zeros_like(p_surf)

        stitched_state = {
            'geopotential':     jnp.array(np.concatenate([z_pl, z_surf], axis=0)),
            'p':                jnp.array(np.concatenate([p_pl, p_surf], axis=0)),
            'T':                jnp.array(np.concatenate([t_pl, t_surf], axis=0)),
            'q':                jnp.array(np.concatenate([q_pl, q_surf], axis=0)),
            'u':                jnp.array(np.concatenate([u_pl, u_surf], axis=0)),
            'v':                jnp.array(np.concatenate([v_pl, v_surf], axis=0)),
            'omega':            jnp.array(np.concatenate([w_pl, w_surf], axis=0)),
            'lsm':              jnp.array(lsm_2d),
            'skt':              jnp.array(skt_2d),
            'latitude':         np.asarray(self.ds['latitude'].values),
            'longitude':        np.asarray(self.ds['longitude'].values),
        }
        return stitched_state
