import jax.numpy as jnp

class BulkAerodynamicPBL:
    def __init__(self, grid, operators, Cd_land=0.005, Cd_ocean=0.001):
        self.grid = grid
        self.op = operators
        
        # For now, we use a uniform drag coefficient. 
        # Later, you can map Cd_land and Cd_ocean based on your topography/land-mask!
        self.Cd = Cd_ocean 

    def get_tendencies(self, state, bg):
        """
        Calculates the frictional deceleration for the lowest model layer.
        Returns tendencies in units of [m/s^2].
        """
        u, v = state['u'], state['v']
        
        # Bring horizontal winds to the mass points to calculate true wind speed
        u_m = self.op.avg(u, axis=0, from_loc='u', to_loc='m')
        v_m = self.op.avg(v, axis=1, from_loc='v', to_loc='m')
        
        # Calculate full 3D wind speed magnitude 
        speed_m_3d = jnp.sqrt(u_m**2 + v_m**2 + 1e-8) 
        
        # Map the 3D wind speed back to the staggered faces
        speed_u_3d = self.op.avg(speed_m_3d, axis=0, from_loc='m', to_loc='u')
        speed_v_3d = self.op.avg(speed_m_3d, axis=1, from_loc='m', to_loc='v')
        
        # Extract the surface layer (level 0) for the drag calculation
        speed_u_surf = speed_u_3d[:, :, 0]
        speed_v_surf = speed_v_3d[:, :, 0]
        
        dz_u_surf = bg['dz_u'][:, :, 0]
        dz_v_surf = bg['dz_v'][:, :, 0]
        
        drag_u_surf = -self.Cd * (speed_u_surf * u[:, :, 0]) / dz_u_surf
        drag_v_surf = -self.Cd * (speed_v_surf * v[:, :, 0]) / dz_v_surf
        
        # Construct the 3D tendency arrays (zeros everywhere except the surface)
        tend_u = jnp.zeros_like(u).at[:, :, 0].set(drag_u_surf)
        tend_v = jnp.zeros_like(v).at[:, :, 0].set(drag_v_surf)
        
        return {'u': tend_u, 'v': tend_v}