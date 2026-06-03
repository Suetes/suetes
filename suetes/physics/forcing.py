import jax.numpy as jnp

class NewtonianRelaxation:
    r"""
    Newtonian Nudging towards a target state.

    Acts as a proxy for missing diabatic physics (radiation, land-surface) 
    by gently pulling the thermodynamic field towards the ERA5 background.
    $$ \frac{\partial \theta_v}{\partial t} = -\frac{1}{\tau_R} (\theta_v - \theta_{v,\text{ERA5}}) $$
    """
    def __init__(self, tau_relax_hours=6.0):
        # Convert relaxation time to seconds
        self.tau_relax = tau_relax_hours * 3600.0
        self.target_state = None

    def update_target(self, target_state):
        """Called dynamically in the integration loop to update the target ERA5 state."""
        self.target_state = target_state

    def get_tendencies(self, state, bg):
        # Extract target from state safely
        target_th_v = state.get('target_th_v')
        
        # Fallback to internal state if missing (e.g., initial baseline calculations)
        if target_th_v is None and self.target_state is not None:
            target_th_v = self.target_state['th_v']
            
        if target_th_v is None:
            return {'th_v': jnp.zeros_like(state['th_v'])}
            
        # Calculate the linear restoring tendency
        tend_th_v = -(state['th_v'] - target_th_v) / self.tau_relax
        
        return {'th_v': tend_th_v}