r"""
Subgrid-Scale Physics and Parameterizations Module.

Contains the physical closures required to model processes that occur at scales 
smaller than the grid resolution, including turbulence, surface 
friction, and moist microphysics.
"""

import jax  
import jax.numpy as jnp     
import flax.linen as nn
from flax import serialization  
import pickle

class PhysicsSuite:
    r"""
    Unified API for orchestrating all physics parameterizations.
    
    The suite separates physics into two categories:
    1. `tendency_schemes`: Continuous processes (like turbulence) evaluated alongside 
       the dynamical core to produce $\partial / \partial t$ tendencies.
    2. `update_schemes`: Instantaneous adjustments (like condensation) applied at 
       the end of the timestep to strictly enforce physical limits.
    """
    def __init__(self):
        self.tendency_schemes = []
        self.update_schemes = []
        self.tracer_keys = []  

    def add_tendency_scheme(self, scheme):
        self.tendency_schemes.append(scheme)

    def add_update_scheme(self, scheme):
        self.update_schemes.append(scheme)

    def register_tracer(self, key):
        if key not in self.tracer_keys:
            self.tracer_keys.append(key)

    def get_explicit_tendencies(self, state, bg, interior_mask=None, ml_params=None):
        """Aggregates continuous momentum and thermodynamic tendencies from all schemes.

        If `interior_mask` is provided, it should be a dict keyed by field
        name ('u', 'v', 'w', 'th_v') with values in [0, 1]. The accumulated
        tendencies are multiplied by the mask before being returned, so
        physics tendencies are zeroed inside the Davies sponge zone where
        the LBC nudging would otherwise be fighting drag and diffusion
        every step.
        """
        tends_total = {'u': jnp.zeros_like(state['u']),
                       'v': jnp.zeros_like(state['v']),
                       'w': jnp.zeros_like(state['w']),
                       'th_v': jnp.zeros_like(state['th_v'])}

        for scheme in self.tendency_schemes:
            # Force the pass explicitly. No try/except!
            if getattr(scheme, 'is_ml_closure', False): 
                scheme_tends = scheme.get_tendencies(state, bg, ml_params=ml_params)
            else:
                scheme_tends = scheme.get_tendencies(state, bg)
            
            for k in scheme_tends:
                tends_total[k] += scheme_tends[k]

        if interior_mask is not None:
            for k in tends_total:
                if k in interior_mask:
                    tends_total[k] = tends_total[k] * interior_mask[k]

        return tends_total

    def apply_state_updates(self, state):
        """Sequentially applies instantaneous thermodynamic adjustments."""
        updated_state = state.copy()
        for scheme in self.update_schemes:
            updates = scheme.apply_update(updated_state)
            updated_state.update(updates)
        return updated_state


class ColumnPhysicsNet(nn.Module):
    """1D Neural Parameterization for subgrid tendencies."""
    hidden_dims: tuple = (128, 128, 64)
    
    @nn.compact
    def __call__(self, x):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.swish(x)
        x = nn.Dense(3, kernel_init=jax.nn.initializers.normal(stddev=1e-5), 
                     bias_init=jax.nn.initializers.zeros)(x)
        return x

class MLPhysicsClosure:
    def __init__(self, op, norm_stats):
        self.op = op
        self.mean = jnp.array(norm_stats['mean'])
        self.std = jnp.array(norm_stats['std'])
        self.model = ColumnPhysicsNet()
        self.is_ml_closure = True

    def get_tendencies(self, state, bg, ml_params=None):
        if ml_params is None:
            raise ValueError("GRAPH SEVERED: ml_params dropped before reaching the closure!")
            
        nn_params = ml_params['nn_params']
        u_m = self.op.avg(state['u'], axis=0, from_loc='u', to_loc='m')
        v_m = self.op.avg(state['v'], axis=1, from_loc='v', to_loc='m')
        z_m = self.op.grid.Z_m
        X = jnp.stack([u_m, v_m, state['th_v'], z_m], axis=-1)
        X_norm = (X - self.mean) / (self.std + 1e-8)
        
        nx, ny, nz, _ = X_norm.shape
        X_flat = X_norm.reshape((nx * ny * nz, 4))
        
        preds_flat = self.model.apply({'params': nn_params}, X_flat)
        preds = preds_flat.reshape((nx, ny, nz, 3))
        
        MAX_TENDENCY = 5.0e-4 
        tend_u_m = MAX_TENDENCY * jnp.tanh(preds[..., 0])
        tend_v_m = MAX_TENDENCY * jnp.tanh(preds[..., 1])
        tend_th_v_m = MAX_TENDENCY * jnp.tanh(preds[..., 2])
        
        tend_u = self.op.avg(tend_u_m, axis=0, from_loc='m', to_loc='u')
        tend_v = self.op.avg(tend_v_m, axis=1, from_loc='m', to_loc='v')
        
        return {'u': tend_u, 'v': tend_v, 'th_v': tend_th_v_m}