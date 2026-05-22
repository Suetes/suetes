"""
Raw Data Extraction Module.

Reads the downloaded NetCDF files and prepares the 3D data arrays by stitching 
surface variables onto the bottom of the upper-air pressure levels.
"""

import xarray as xr
import numpy as np
import jax.numpy as jnp

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
            pl_path (str): File path to the ERA5 pressure-level data.
            sl_path (str): File path to the ERA5 single-level data.
        """
        self.ds_pl = xr.open_dataset(pl_path)
        self.ds_sl = xr.open_dataset(sl_path)

    def get_stitched_state(self, time_idx=0):
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
        
        q_surf = np.zeros_like(p_surf) 
        omega_surf = np.zeros_like(p_surf)

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
            'geopotential': jnp.array(np.concatenate([z_pl, z_surf], axis=0)),
            'p': jnp.array(np.concatenate([p_pl, p_surf], axis=0)),
            'T': jnp.array(np.concatenate([t_pl, t_surf], axis=0)),
            'q': jnp.array(np.concatenate([q_pl, q_surf], axis=0)),
            'u': jnp.array(np.concatenate([u_pl, u_surf], axis=0)),
            'v': jnp.array(np.concatenate([v_pl, v_surf], axis=0)),
            'omega': jnp.array(np.concatenate([omega_pl, omega_surf], axis=0)),
            'skt': jnp.array(skt_2d),
            'lsm': jnp.array(lsm_2d),
            'latitude': pl['latitude'].values,
            'longitude': pl['longitude'].values
        }
        
        return stitched_state
