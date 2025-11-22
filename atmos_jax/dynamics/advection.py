import jax.numpy as jnp
from ..core.operators import CGridOperator

class SchaerAdvection:
    """
    Solves the Advective Form equation: d_rho/dt = - (u * d_rho/dx + w * d_rho/dz)
    using Curvilinear Chain Rule operators.
    """
    def __init__(self, grid, u_profile_fn):
        self.grid = grid
        # Enable periodic_x for correct boundary handling in Schaer test
        self.op = CGridOperator(grid.dx, grid.dz, periodic_x=True)
        
        # 1. Compute Physical u(z) at Mass Points
        # We use the physical Z coordinates at Mass points
        self.u_phys = u_profile_fn(self.grid.Z_m)
        
        # 2. Physical w is zero for this specific test case 
        self.w_phys = jnp.zeros_like(self.u_phys)

        # 3. Metrics for Mass Points (needed for Chain Rule)
        self.metrics = self.grid.metrics['m']

    def compute_rhs(self, state, forcing=None):
        rho = state['rho']
        
        # 1. Compute Physical Gradients (Chain Rule)
        grad_x, grad_z = self.op.grad_2d_curvilinear(rho, self.metrics)
        
        # 2. Advection Equation: - (u * grad_x + w * grad_z)
        rhs = - (self.u_phys * grad_x + self.w_phys * grad_z)
        
        return {'rho': rhs}