import jax.numpy as jnp
from suetes.slice2d.operators import CGridOperator

class VerticalSlice:
    def __init__(self, grid, constants, damp_height=20000.0, max_damp=0.5, N_bv=0.01):
        self.grid = grid
        self.c = constants
        self.op = CGridOperator(grid.dx, grid.dz, periodic_x=grid.periodic_x) 
        self.theta_0 = 300.0
        
        zeta_c = jnp.linspace(0, self.grid.Lz, self.grid.nz + 1)
        zeta_m = 0.5 * (zeta_c[:-1] + zeta_c[1:])
        
        Z_ref_m = jnp.broadcast_to(zeta_m, self.grid.Z_m.shape)
        Z_ref_w = jnp.broadcast_to(zeta_c, self.grid.Z_w.shape)
        
        if N_bv > 0.0:
            self.theta_bg = self.theta_0 * jnp.exp((N_bv**2 / self.c['g']) * Z_ref_m)
            pi_factor = (self.c['g']**2) / (self.c['cp'] * self.theta_0 * N_bv**2)
            self.pi_bg = 1.0 + pi_factor * (jnp.exp(-(N_bv**2 / self.c['g']) * Z_ref_m) - 1.0)
            self.dpi0_dz_w = -self.c['g'] / (self.c['cp'] * self.theta_0 * jnp.exp((N_bv**2 / self.c['g']) * Z_ref_w))
        else:
            self.theta_bg = self.theta_0 * jnp.ones_like(Z_ref_m) 
            self.pi_bg = 1.0 - (self.c['g'] * Z_ref_m) / (self.c['cp'] * self.theta_0)
            self.dpi0_dz_w = -self.c['g'] / (self.c['cp'] * self.theta_0) * jnp.ones_like(Z_ref_w)
        
        z_w = self.grid.Z_w[:, :]
        z_top = jnp.max(z_w)
        self.tau_damp = jnp.where(
            z_w > damp_height,
            max_damp * 0.5 * (1.0 + jnp.tanh(jnp.pi * (z_w - damp_height) / (z_top - damp_height) - jnp.pi/2)),
            0.0
        )

        self.metrics = {
            'm': self._compute_discrete_metrics(self.grid.Z_m),
            'u': self._compute_discrete_metrics(self.grid.Z_u),
            'w': self._compute_discrete_metrics(self.grid.Z_w)
        }

    def _compute_discrete_metrics(self, Z_grid):
        Z_pad_z = jnp.pad(Z_grid, ((0, 0), (1, 1)), mode='edge')
        z_zeta = (Z_pad_z[:, 2:] - Z_pad_z[:, :-2]) / (2.0 * self.grid.dz)

        if getattr(self.grid, 'periodic_x', False):
            z_xi = (jnp.roll(Z_grid, -1, axis=0) - jnp.roll(Z_grid, 1, axis=0)) / (2.0 * self.grid.dx)
        else:
            Z_pad_x = jnp.pad(Z_grid, ((1, 1), (0, 0)), mode='edge')
            z_xi = (Z_pad_x[2:, :] - Z_pad_x[:-2, :]) / (2.0 * self.grid.dx)
            
        return {'z_xi': z_xi, 'z_zeta': z_zeta}

    def _get_metrics(self, grid_type: str):
        return self.metrics[grid_type]