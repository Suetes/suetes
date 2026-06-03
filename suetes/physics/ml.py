import jax  
import jax.numpy as jnp     
import flax.linen as nn

class ColumnPhysicsCNN(nn.Module):
    """1D Convolutional Neural Parameterization for vertical columns."""
    @nn.compact
    def __call__(self, x):
        # Input shape expected: (batch, nz, features)
        x = nn.Conv(features=128, kernel_size=(3,), padding='SAME')(x)
        x = nn.swish(x)
        x = nn.Conv(features=128, kernel_size=(3,), padding='SAME')(x)
        x = nn.swish(x)
        x = nn.Conv(features=64, kernel_size=(3,), padding='SAME')(x)
        x = nn.swish(x)
        x = nn.Dense(3, kernel_init=jax.nn.initializers.normal(stddev=1e-3),
                     bias_init=jax.nn.initializers.zeros)(x)
        return x

class MLPhysicsClosure:
    def __init__(self, op, norm_stats):
        self.op = op
        self.mean = jnp.array(norm_stats['mean'])
        self.std = jnp.array(norm_stats['std'])
        self.model = ColumnPhysicsCNN() # Upgraded to CNN
        self.is_ml_closure = True
        
        # Precompute the longitude field for Local Solar Time calculations
        Xi_m, Yi_m = jnp.meshgrid(self.op.grid.x_m, self.op.grid.y_m, indexing='ij')
        _, self.lon_m = self.op.grid.proj.get_lat_lon(Xi_m, Yi_m)

    def get_tendencies(self, state, bg, ml_params=None):
        if ml_params is None:
            raise ValueError("GRAPH SEVERED: ml_params dropped before reaching the closure!")
            
        nn_params = ml_params['nn_params']
        u_m = self.op.avg(state['u'], axis=0, from_loc='u', to_loc='m')
        v_m = self.op.avg(state['v'], axis=1, from_loc='v', to_loc='m')
        z_m = self.op.grid.Z_m
        th_v = state['th_v']
        
        # We revert back to th_v_prime without the noisy manual gradient
        th_v_prime = th_v - bg['th_v']
        
        theta_surf = state.get('theta_surf', th_v[:, :, 0])
        z_surf = z_m[:, :, 0]
        lf = state.get('land_fraction', jnp.ones_like(th_v[:, :, 0]))
        
        # --- LOCAL SOLAR TIME CALCULATION ---
        t_curr = state.get('t_curr', 0.0) 
        utc_hour = (t_curr / 3600.0) % 24.0
        
        # Shift time by longitude (1 hour per 15 degrees)
        local_hour = (utc_hour + self.lon_m / 15.0) % 24.0
        
        sin_t = jnp.sin(2 * jnp.pi * local_hour / 24.0)
        cos_t = jnp.cos(2 * jnp.pi * local_hour / 24.0)
        
        # Broadcast all 2D fields to the 3D column
        theta_surf_3d = jnp.broadcast_to(theta_surf[..., None], th_v.shape)
        z_surf_3d = jnp.broadcast_to(z_surf[..., None], th_v.shape)
        lf_3d = jnp.broadcast_to(lf[..., None], th_v.shape)
        sin_t_3d = jnp.broadcast_to(sin_t[..., None], th_v.shape)
        cos_t_3d = jnp.broadcast_to(cos_t[..., None], th_v.shape)
        
        # Feature Stack: 9 Features (Manual gradient removed)
        X = jnp.stack([
            u_m, v_m, th_v_prime, z_m, 
            theta_surf_3d, z_surf_3d, lf_3d, sin_t_3d, cos_t_3d
        ], axis=-1)
        
        X_norm = (X - self.mean) / (self.std + 1e-8)
        nx, ny, nz, _ = X_norm.shape
        
        # Reshape specifically for the 1D CNN: (batch, sequence, features)
        X_cnn = X_norm.reshape((nx * ny, nz, 9)) 
        
        preds_cnn = self.model.apply({'params': nn_params}, X_cnn)
        preds = preds_cnn.reshape((nx, ny, nz, 3))
        
        MAX_TENDENCY = 5.0e-4 
        tend_u_m = MAX_TENDENCY * jnp.tanh(preds[..., 0])
        tend_v_m = MAX_TENDENCY * jnp.tanh(preds[..., 1])
        tend_th_v_m = MAX_TENDENCY * jnp.tanh(preds[..., 2])
        
        tend_u = self.op.avg(tend_u_m, axis=0, from_loc='m', to_loc='u')
        tend_v = self.op.avg(tend_v_m, axis=1, from_loc='m', to_loc='v')
        
        return {'u': tend_u, 'v': tend_v, 'th_v': tend_th_v_m}