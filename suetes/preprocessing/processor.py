import xarray as xr
import numpy as np
import jax.numpy as jnp

class ERA5Processor:
    def __init__(self, pl_path, sl_path):
        """Loads and holds the pressure level and single level datasets."""
        self.ds_pl = xr.open_dataset(pl_path)
        self.ds_sl = xr.open_dataset(sl_path)

    def get_stitched_state(self, time_idx=0):
        """
        Extracts arrays, broadcasts pressure, and stitches the surface 
        to the bottom of the pressure levels for a given timestep.
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
            'latitude': pl['latitude'].values,
            'longitude': pl['longitude'].values
        }
        
        return stitched_state