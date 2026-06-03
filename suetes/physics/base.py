import jax.numpy as jnp

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
                # Dynamically add the tendency array if this is a new prognostic variable
                if k not in tends_total:
                    tends_total[k] = jnp.zeros_like(scheme_tends[k])
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