"""
Raw Data Extraction Module.

Reads the downloaded NetCDF files and prepares the 3D data arrays by stitching 
surface variables onto the bottom of the upper-air pressure levels.
"""

import xarray as xr
import numpy as np
import jax.numpy as jnp
from scipy.ndimage import uniform_filter

class ERA5Processor:
    """
    Processes raw ERA5 data to prepare it for the dynamical core.
    
    This class handles the "stitching" of surface data (2D) onto the 
    pressure levels (3D) to create a complete atmospheric state vector.
    """
    
    def __init__(self, pl_path, sl_path):
        """
        Initializes the processor and loads the dataset files.

        Args:
            pl_path (str | list[str]): ERA5 pressure-level data -- a single file,
                a list of daily files, or a glob pattern.
            sl_path (str | list[str]): ERA5 single-level data (same forms).

        A list/glob is opened lazily with ``xarray.open_mfdataset`` and
        concatenated along time, so per-timestep reads only load the relevant
        day-chunk -- the read-side counterpart to the daily download.
        """
        self.ds_pl = self._open_dataset(pl_path)
        self.ds_sl = self._open_dataset(sl_path)

    @staticmethod
    def _open_dataset(path):
        if isinstance(path, (list, tuple)):
            return xr.open_mfdataset(list(path), combine="by_coords")
        if isinstance(path, str) and any(ch in path for ch in "*?["):
            return xr.open_mfdataset(path, combine="by_coords")
        return xr.open_dataset(path)

    def get_stitched_state(self, time_idx=0, coarsen_window=None):
        """
        Extracts arrays, broadcasts pressure, and stitches the surface 
        to the bottom of the pressure levels for a given timestep.
        
        The vertical dimension of the output is (k_vertical + 1), where k_vertical
        is the number of pressure levels.
        
        Args:
            time_idx (int): The index of the timestep to extract.

        Returns:
            dict: A dictionary containing the atmospheric state variables,
                  ready for conversion to JAX arrays.
        """
        time_dim = 'valid_time' if 'valid_time' in self.ds_pl.dims else 'time'
        
        pl = self.ds_pl.isel({time_dim: time_idx})
        sl = self.ds_sl.isel({time_dim: time_idx})

        # Expand Single Levels (2D -> 3D)
        z_surf = np.expand_dims(sl['z'].values, axis=0)
        p_surf = np.expand_dims(sl['sp'].values, axis=0)
        t_surf = np.expand_dims(sl['t2m'].values, axis=0)
        u_surf = np.expand_dims(sl['u10'].values, axis=0)
        v_surf = np.expand_dims(sl['v10'].values, axis=0)
        
        # Calculate true surface specific humidity (q_surf) from 2m dewpoint temperature
        d2m_surf = np.expand_dims(sl['d2m'].values, axis=0)
        epsilon = 0.622
        # Tetens formula for saturation vapor pressure
        e_surf = 611.2 * np.exp(17.67 * (d2m_surf - 273.15) / (d2m_surf - 29.65))
        q_surf = (epsilon * e_surf) / (p_surf - (1.0 - epsilon) * e_surf)

        omega_surf = np.zeros_like(p_surf)

        # Create a sub-surface anchor to prevent flat-line extrapolation in valleys.
        # We extrapolate 3000m down using the standard lapse rate (0.0065 K/m).
        dz_sub = 3000.0
        z_sub = z_surf - dz_sub
        t_sub = t_surf + (0.0065 * dz_sub)
        p_sub = p_surf * (t_sub / t_surf) ** (9.81 / (287.05 * 0.0065))
        q_sub = q_surf  # Assume well-mixed specific humidity below surface
        u_sub = u_surf
        v_sub = v_surf
        omega_sub = omega_surf

        # Surface-only fields (2D), not stitched onto the 3D state.
        # skt is the skin temperature and varies hourly (used as the Dirichlet
        # surface temperature for the bulk drag and SHF schemes). lsm is the
        # ERA5 land-sea mask, time-invariant in practice; the value at this
        # timestep is the same as at any other.
        skt_2d = sl['skt'].values
        lsm_2d = sl['lsm'].values

        # Extract and format Pressure Levels
        z_pl = pl['z'].values
        t_pl = pl['t'].values
        q_pl = pl['q'].values
        u_pl = pl['u'].values
        v_pl = pl['v'].values
        omega_pl = pl['w'].values

        level_dim = 'pressure_level' if 'pressure_level' in pl.dims else 'level'
        
        # Broadcast 1D pressure coordinates into a full 3D array
        p_1d = pl[level_dim].values * 100.0 
        p_pl = np.broadcast_to(p_1d[:, None, None], t_pl.shape)

        # Stitch them together
        stitched_state = {
            'geopotential': jnp.array(np.concatenate([z_pl, z_surf, z_sub], axis=0)),
            'p': jnp.array(np.concatenate([p_pl, p_surf, p_sub], axis=0)),
            'T': jnp.array(np.concatenate([t_pl, t_surf, t_sub], axis=0)),
            'q': jnp.array(np.concatenate([q_pl, q_surf, q_sub], axis=0)),
            'u': jnp.array(np.concatenate([u_pl, u_surf, u_sub], axis=0)),
            'v': jnp.array(np.concatenate([v_pl, v_surf, v_sub], axis=0)),
            'omega': jnp.array(np.concatenate([omega_pl, omega_surf, omega_sub], axis=0)),
            'skt': jnp.array(skt_2d),
            'lsm': jnp.array(lsm_2d),
            'latitude': pl['latitude'].values,
            'longitude': pl['longitude'].values
        }
        
        if coarsen_window is not None and coarsen_window > 1:
            coarsened_state = {}
            for key, val in stitched_state.items():
                if key in ('latitude', 'longitude'):
                    coarsened_state[key] = val
                    continue
                arr = np.asarray(val)
                if arr.ndim == 2:
                    arr = uniform_filter(arr, size=coarsen_window, mode='nearest')
                elif arr.ndim == 3:
                    arr = uniform_filter(arr, size=(1, coarsen_window, coarsen_window), mode='nearest')
                coarsened_state[key] = jnp.asarray(arr)
            return coarsened_state
            
        return stitched_state
