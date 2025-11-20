import os

euler_code = """
import jax.numpy as jnp
from ..core.operators import CGridOperator

class EulerSet1:
    def __init__(self, grid, constants, periodic_x=False):
        self.grid = grid
        self.c = constants
        self.periodic_x = periodic_x
        self.op = CGridOperator(grid.dx, grid.dz, periodic_x=periodic_x)
        self.sponge = self._init_sponge()

    def _init_sponge(self):
        if self.periodic_x:
            return jnp.zeros_like(self.grid.X_m)
            
        X, Z = self.grid.X_m, self.grid.Z_m
        tau = jnp.zeros_like(X)
        
        width_x = 0.15 * self.grid.Lx
        width_z = 0.25 * self.grid.Lz
        max_damp = 0.1
        
        dist_L = X - X[0,0]
        dist_R = X[-1,0] - X
        dist_T = self.grid.Lz - Z
        
        tau = jnp.where(dist_L < width_x, max_damp * jnp.sin(0.5*jnp.pi*(width_x - dist_L)/width_x)**2, tau)
        tau = jnp.where(dist_R < width_x, jnp.maximum(tau, max_damp * jnp.sin(0.5*jnp.pi*(width_x - dist_R)/width_x)**2), tau)
        tau = jnp.where(dist_T < width_z, jnp.maximum(tau, max_damp * jnp.sin(0.5*jnp.pi*(width_z - dist_T)/width_z)**2), tau)
        return tau

    def compute_rhs(self, state, forcing):
        u, w = state['u'], state['w']
        pi, th = state['pi_p'], state['theta_p']
        bg = state['background']
        
        u_ref = forcing.get('u_ref', 0.0)
        u_acc = forcing.get('u_acc', 0.0)
        nu4 = self.c.get('nu4', 0.0)
        nu = self.c.get('nu', 0.0)

        # --- 1. Pressure ---
        du_dx = self.op.diff_x_u_to_m(u)
        dw_dz = self.op.diff_z_w_to_m(w)
        
        # Metric correction
        if self.periodic_x:
             div_phys = du_dx + dw_dz
        else:
            u_at_m = self.op.avg_u_to_m(u)
            du_dzeta_m = jnp.pad((u_at_m[:, 2:] - u_at_m[:, :-2]) / (2*self.grid.dz), ((0,0),(1,1)), mode='edge')
            metrics = self.grid.metrics['m']
            div_phys = du_dx - (metrics['z_xi']/metrics['z_zeta'])*du_dzeta_m + (1.0/metrics['z_zeta'])*dw_dz
        
        rhs_pi = - (self.c['R']/self.c['cv']) * (bg['pi'] + pi) * div_phys
        
        # FIX: Add vertical advection of mean pressure
        w_at_m = self.op.avg_w_to_m(w)
        rhs_pi -= w_at_m * bg['dpi_dz']
        
        rhs_pi += self.op.advect_2d(pi, u, w, 'mass') + self.op.hyper_diff_2d(pi, nu4) + self.op.laplacian_2d(pi, nu)
        rhs_pi -= self.sponge * pi
        
        # --- 2. U-Momentum ---
        dpi_dx = self.op.diff_x_m_to_u(pi) 
        
        if self.periodic_x:
            grad_p_x = dpi_dx
        else:
            dpi_dz_at_u = self.op.avg_w_to_u(self.op.diff_z_m_to_w(pi))
            metrics = self.grid.metrics['u']
            grad_p_x = dpi_dx - (metrics['z_xi']/metrics['z_zeta']) * dpi_dz_at_u
        
        th_at_u = self.op.avg_m_to_u(bg['theta'] + th)
        
        rhs_u = -self.c['cp'] * th_at_u * grad_p_x
        rhs_u += u_acc + self.op.advect_2d(u, u, w, 'u') + self.op.hyper_diff_2d(u, nu4) + self.op.laplacian_2d(u, nu)
        rhs_u -= self.op.avg_m_to_u(self.sponge) * (u - u_ref)
        
        # --- 3. W-Momentum ---
        dpi_dz_w = self.op.diff_z_m_to_w(pi)
        
        if self.periodic_x:
             grad_p_z = dpi_dz_w
        else:
            metrics = self.grid.metrics['w']
            grad_p_z = (1.0/metrics['z_zeta']) * dpi_dz_w
        
        th_tot_w = self.op.avg_m_to_w(bg['theta'] + th)
        
        # Buoyancy
        # Use potential temp perturbation for buoyancy
        theta_p_w = self.op.avg_m_to_w(th)
        theta_bar_w = self.op.avg_m_to_w(bg['theta'])
        buoyancy = self.c['g'] * (theta_p_w / theta_bar_w)
        
        rhs_w = -self.c['cp'] * th_tot_w * grad_p_z + buoyancy
        rhs_w += self.op.advect_2d(w, u, w, 'w') + self.op.hyper_diff_2d(w, nu4) + self.op.laplacian_2d(w, nu)
        rhs_w -= self.op.avg_m_to_w(self.sponge) * w
        
        # --- 4. Potential Temp ---
        # w_at_m calculated above
        rhs_th = - w_at_m * bg['dtheta_dz']
        rhs_th += self.op.advect_2d(th, u, w, 'mass') + self.op.hyper_diff_2d(th, nu4) + self.op.laplacian_2d(th, nu)
        rhs_th -= self.sponge * th
        
        return {'u': rhs_u, 'w': rhs_w, 'pi_p': rhs_pi, 'theta_p': rhs_th}
"""

path = "atmos_jax/dynamics/euler.py"
with open(path, "w") as f:
    f.write(euler_code)

print(f"[+] Fixed physics in {path} (Added mean pressure advection)")
