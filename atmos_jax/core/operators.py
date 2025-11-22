import jax.numpy as jnp

class CGridOperator:
    """
    Spatial discretization operators for the Arakawa C-grid.
    Supports Wall-Bounded (2nd Order) and Periodic (4th Order) modes.
    """
    def __init__(self, dx, dz, periodic_x=False):
        self.dx = dx
        self.dz = dz
        self.periodic_x = periodic_x

    # ===========================================================
    # HELPERS
    # ===========================================================
    def _diff_centered_2nd_padded(self, f, axis):
        if axis == 0: # X
            inner = (f[1:, :] - f[:-1, :]) / self.dx
            return jnp.pad(inner, ((1, 1), (0, 0)), constant_values=0.0)
        else: # Z
            inner = (f[:, 1:] - f[:, :-1]) / self.dz
            return jnp.pad(inner, ((0, 0), (1, 1)), constant_values=0.0)

    def _diff_centered_4th(self, f, axis):
        if axis == 0 and self.periodic_x:
            f_p2 = jnp.roll(f, -2, axis=0)
            f_p1 = jnp.roll(f, -1, axis=0)
            f_m1 = jnp.roll(f, 1, axis=0)
            f_m2 = jnp.roll(f, 2, axis=0)
            return (-f_p2 + 8*f_p1 - 8*f_m1 + f_m2) / (12 * self.dx)
        else:
            return self._diff_centered_2nd_padded(f, axis)

    # ===========================================================
    # DIFFERENTIATION
    # ===========================================================
    def diff_x_m_to_u(self, f):
        if self.periodic_x:
            # 4th order periodic
            pad = jnp.pad(f, ((2, 2), (0, 0)), mode='wrap')
            grad = (27.0*(pad[2:-2]-pad[1:-3]) - (pad[3:-1]-pad[0:-4])) / (24.0 * self.dx)
            
            # FIX: Always concatenate for periodic to match U-grid size (nx+1)
            # This ensures grad_p_x matches th_at_u
            return jnp.concatenate([grad, grad[:1, :]], axis=0)
            
        return self._diff_centered_2nd_padded(f, 0)

    def diff_x_u_to_m(self, f):
        if self.periodic_x:
            # Input f is U-grid (nx+1). We take first nx points.
            pad = jnp.pad(f[:-1], ((2, 2), (0, 0)), mode='wrap')
            return (27.0*(pad[3:-1]-pad[2:-2]) - (pad[4:]-pad[1:-3])) / (24.0 * self.dx)
        return (f[1:, :] - f[:-1, :]) / self.dx
    
    def diff_z_m_to_w(self, f): return self._diff_centered_2nd_padded(f, 1)
    def diff_z_w_to_m(self, f): return (f[:, 1:] - f[:, :-1]) / self.dz

    # ===========================================================
    # AVERAGING
    # ===========================================================
    def avg_u_to_m(self, f):
        if self.periodic_x:
            # u is (nx+1). Use first nx points.
            pad = jnp.pad(f[:-1], ((2, 2), (0, 0)), mode='wrap')
            return (9.0*(pad[2:-2]+pad[3:-1]) - (pad[1:-3]+pad[4:])) / 16.0
        return 0.5 * (f[1:, :] + f[:-1, :])

    def avg_m_to_u(self, f):
        if self.periodic_x:
            pad = jnp.pad(f, ((2, 2), (0, 0)), mode='wrap')
            avg = (9.0*(pad[2:-2]+pad[1:-3]) - (pad[3:-1]+pad[0:-4])) / 16.0
            return jnp.concatenate([avg, avg[:1, :]], axis=0)
        inner = 0.5 * (f[1:, :] + f[:-1, :])
        return jnp.pad(inner, ((1, 1), (0, 0)), mode='edge')

    def avg_w_to_u(self, f): return self.avg_m_to_u(self.avg_w_to_m(f))
    def avg_u_to_w(self, f): return self.avg_m_to_w(self.avg_u_to_m(f))
    
    def avg_w_to_m(self, f): return 0.5 * (f[:, 1:] + f[:, :-1])
    def avg_m_to_w(self, f):
        inner = 0.5 * (f[:, 1:] + f[:, :-1])
        return jnp.pad(inner, ((0, 0), (1, 1)), mode='edge')

    # ===========================================================
    # ADVECTION & GRADIENTS
    # ===========================================================
    def grad_2d_curvilinear(self, f, metrics):
        """
        Computes physical gradients (df/dx, df/dz) using the chain rule.
        """
        # 1. Logical Gradient d/d_xi (at Mass points)
        if self.periodic_x:
            # Center diff on periodic domain (nx)
            d_xi = (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2 * self.dx)
        else:
            d_xi = jnp.pad((f[2:] - f[:-2]) / (2*self.dx), ((1,1), (0,0)), mode='edge')

        # 2. Logical Gradient d/d_zeta
        f_pad = jnp.pad(f, ((0, 0), (1, 1)), mode='edge')
        d_zeta = (f_pad[:, 2:] - f_pad[:, :-2]) / (2 * self.dz)

        if metrics is None:
            return d_xi, d_zeta

        # 3. Chain Rule
        z_xi = metrics['z_xi']
        z_zeta = metrics['z_zeta']

        grad_z = d_zeta / z_zeta
        grad_x = d_xi - (z_xi * grad_z)

        return grad_x, grad_z

    def advect_2d(self, f, u, w, loc, metrics=None):
        """
        Full 2D advection: -(u.grad)f
        """
        if loc == 'mass':
            u_loc = self.avg_u_to_m(u)
            w_loc = self.avg_w_to_m(w)
        elif loc == 'u':
            u_loc = u
            w_loc = self.avg_m_to_u(self.avg_w_to_m(w)) 
        elif loc == 'w':
            u_loc = self.avg_m_to_w(self.avg_u_to_m(u))
            w_loc = w

        grad_x, grad_z = self.grad_2d_curvilinear(f, metrics)
        return -(u_loc * grad_x + w_loc * grad_z)

    # ===========================================================
    # DIFFUSION
    # ===========================================================
    def hyper_diff_2d(self, f, nu4):
        if nu4 == 0: return 0.0
        d4x = jnp.zeros_like(f)
        d4z = jnp.zeros_like(f)

        if self.periodic_x:
            f_val = f[:-1, :] if f.shape[0]%2!=0 else f
            pad = jnp.pad(f_val, ((2, 2), (0, 0)), mode='wrap')
            inner = (pad[:-4] - 4*pad[1:-3] + 6*pad[2:-2] - 4*pad[3:-1] + pad[4:]) / self.dx**4
            d4x = jnp.concatenate([inner, inner[:1]], axis=0) if f.shape[0]%2!=0 else inner
        elif f.shape[0] >= 5:
            inner = (f[:-4] - 4*f[1:-3] + 6*f[2:-2] - 4*f[3:-1] + f[4:]) / (self.dx**4)
            d4x = jnp.pad(inner, ((2, 2), (0, 0)), constant_values=0.0)

        if f.shape[1] >= 5:
            inner = (f[:, :-4] - 4*f[:, 1:-3] + 6*f[:, 2:-2] - 4*f[:, 3:-1] + f[:, 4:]) / (self.dz**4)
            d4z = jnp.pad(inner, ((0, 0), (2, 2)), constant_values=0.0)
            
        return -nu4 * (d4x + d4z)

    def laplacian_2d(self, f, nu):
        if nu == 0: return 0.0
        if self.periodic_x:
             if f.shape[0] % 2 != 0: # U-point?
                f_valid = f[:-1, :]
                inner = (jnp.roll(f_valid, -1, axis=0) - 2*f_valid + jnp.roll(f_valid, 1, axis=0)) / self.dx**2
                d2x = jnp.concatenate([inner, inner[:1, :]], axis=0)
             else:
                d2x = (jnp.roll(f, -1, axis=0) - 2*f + jnp.roll(f, 1, axis=0)) / self.dx**2
        else:
            if f.shape[0] > f.shape[1]: 
                 inner_x = (f[2:, :] - 2*f[1:-1, :] + f[:-2, :]) / (self.dx**2)
                 d2x = jnp.pad(inner_x, ((1, 1), (0, 0)), constant_values=0.0)
            else:
                 padded_f = jnp.pad(f, ((1, 1), (0, 0)), mode='edge')
                 d2x = (padded_f[2:, :] - 2*padded_f[1:-1, :] + padded_f[:-2, :]) / (self.dx**2)

        if f.shape[1] > f.shape[0]:
            inner_z = (f[:, 2:] - 2*f[:, 1:-1] + f[:, :-2]) / (self.dz**2)
            d2z = jnp.pad(inner_z, ((0, 0), (1, 1)), constant_values=0.0)
        else:
            padded_f = jnp.pad(f, ((0, 0), (1, 1)), mode='edge')
            d2z = (padded_f[:, 2:] - 2*padded_f[:, 1:-1] + padded_f[:, :-2]) / (self.dz**2)
            
        return nu * (d2x + d2z)