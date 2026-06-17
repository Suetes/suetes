"""
Discrete Spatial Operators Module.

Provides finite-difference stencils, spatial averaging operators, and 
high-order interpolators for integrating PDEs on an Arakawa C-grid.
"""

import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jnd

class CGridOperator3D:
    """
    Finite-difference and averaging operators for staggered C-grid variables.
    """
    def __init__(self, grid):
        """
        Initializes the C-grid operators.

        Args:
            grid (RegionalGrid3D): The computational grid geometry.
        """
        self.grid = grid
        # Pre-broadcast the mass weighting factors for use in derivatives
        self.m_factors = {
            loc: jnp.expand_dims(m_2d, axis=-1) 
            for loc, m_2d in self.grid.m_factors.items()
        }

    def _apply_padding(self, f, axis, from_loc, to_loc):
        """
        Applies padding to ensure boundary values are available for centered stencils.
        
        Uses stop_gradient on the duplicated edge cells. This ensures the forward 
        pass smoothly extrapolates (open boundary), but the adjoint pass drops 
        the boundary sensitivities rather than accumulating them.
        """
        if from_loc == 'm' and to_loc in ['u', 'v', 'w']:
            
            # Slice out the edge values and detach them from the adjoint graph
            if axis == 0:
                left_edge  = jax.lax.stop_gradient(f[0:1, ...])
                right_edge = jax.lax.stop_gradient(f[-1:, ...])
                return jnp.concatenate([left_edge, f, right_edge], axis=0)
                
            elif axis == 1:
                left_edge  = jax.lax.stop_gradient(f[:, 0:1, ...])
                right_edge = jax.lax.stop_gradient(f[:, -1:, ...])
                return jnp.concatenate([left_edge, f, right_edge], axis=1)
                
            elif axis == 2:
                left_edge  = jax.lax.stop_gradient(f[:, :, 0:1])
                right_edge = jax.lax.stop_gradient(f[:, :, -1:])
                return jnp.concatenate([left_edge, f, right_edge], axis=2)

        return f

    def diff(self, f, axis, from_loc, to_loc):
        r"""
        Computes the discrete spatial derivative across a staggered boundary.

        Incorporates the local map factor $m$ when differentiating along horizontal axes. 
        For example, a derivative along the $x$-axis from a mass point to a $u$ face is:

        $$ \delta_x f = m_u \frac{f_{i} - f_{i-1}}{\Delta x} $$

        Args:
            f (jnp.ndarray): The input field array.
            axis (int): The axis of differentiation (0 for x, 1 for y, 2 for z).
            from_loc (str): Original grid staggering ('m', 'u', 'v', 'w').
            to_loc (str): Target grid staggering ('m', 'u', 'v', 'w').

        Returns:
            jnp.ndarray: The differentiated field mapped to `to_loc`.
        """
        delta = self.grid.delta[axis]
        derivative = jnp.diff(f, axis=axis) / delta
        derivative = self._apply_padding(derivative, axis, from_loc, to_loc)
        
        if axis in [0, 1]: 
            derivative = derivative * self.m_factors[to_loc]
        return derivative

    def avg(self, f, axis, from_loc, to_loc):
        r"""
        Computes the 2-point spatial average to translate fields between staggerings.

        $$ \overline{f}^x = \frac{1}{2}(f_{i} + f_{i-1}) $$

        Args:
            f (jnp.ndarray): The input field array.
            axis (int): The axis of averaging (0 for x, 1 for y, 2 for z).
            from_loc (str): Original grid staggering ('m', 'u', 'v', 'w').
            to_loc (str): Target grid staggering ('m', 'u', 'v', 'w').

        Returns:
            jnp.ndarray: The averaged field mapped to `to_loc`.
        """
        avg_val = 0.5 * (f[:-1] + f[1:]) if axis == 0 else \
                  0.5 * (f[:, :-1] + f[:, 1:]) if axis == 1 else \
                  0.5 * (f[:, :, :-1] + f[:, :, 1:])
        return self._apply_padding(avg_val, axis, from_loc, to_loc)

def cubic_weight(p0, p1, p2, p3, t):
    r"""
    Evaluates a 1D cubic spline interpolant using 4 grid points.

    $$ f(t) = \left(-\frac{1}{2}p_0 + \frac{3}{2}p_1 - \frac{3}{2}p_2 + \frac{1}{2}p_3\right)t^3 + \left(p_0 - \frac{5}{2}p_1 + 2p_2 - \frac{1}{2}p_3\right)t^2 + \left(-\frac{1}{2}p_0 + \frac{1}{2}p_2\right)t + p_1 $$

    where $t \in [0, 1]$ is the fractional distance between the central nodes $p_1$ and $p_2$.
    """
    return (-0.5*p0 + 1.5*p1 - 1.5*p2 + 0.5*p3) * t**3 + \
           (p0 - 2.5*p1 + 2.0*p2 - 0.5*p3) * t**2 + \
           (-0.5*p0 + 0.5*p2) * t + p1

def tensor_product_interp_3d(field, coords, use_limiter=False):
    r"""
    Evaluates a 3D tricubic interpolation using a 64-point local stencil.

    Used primarily in the Semi-Lagrangian scheme to evaluate fields at the continuous 
    departure point $\mathbf{x}_d$.

    If `use_limiter` is True, applies a quasi-monotone limiter that bounds the 
    interpolated value by the extrema of the immediate 8-point cubic neighborhood 
    to prevent unphysical overshoots (e.g., negative mass or water vapor):
    
    $$ \min(f_{nb}) \le f(\mathbf{x}_d) \le \max(f_{nb}) $$
    
    where $f_{nb}$ are the values at the 8 surrounding discrete grid points.

    Args:
        field (jnp.ndarray): The 3D data grid to interpolate from.
        coords (tuple): A tuple $(x, y, z)$ of fractional continuous indices.
        use_limiter (bool): Whether to apply the bounding limiter. Defaults to False.
            
    Returns:
        jnp.ndarray: The interpolated values.
    """
    nx, ny, nz = field.shape
    x, y, z = coords[0], coords[1], coords[2]

    x, y, z = jnp.clip(x, 0, nx - 1), jnp.clip(y, 0, ny - 1), jnp.clip(z, 0, nz - 1)
    x0, y0, z0 = jnp.floor(x).astype(jnp.int32), jnp.floor(y).astype(jnp.int32), jnp.floor(z).astype(jnp.int32)
    dx, dy, dz = x - x0, y - y0, z - z0

    def get_stencil(idx, max_idx):
        return (jnp.maximum(idx - 1, 0), idx, jnp.minimum(idx + 1, max_idx), jnp.minimum(idx + 2, max_idx))

    ix, iy, iz = get_stencil(x0, nx - 1), get_stencil(y0, ny - 1), get_stencil(z0, nz - 1)

    def interp_z(x_idx, y_idx):
        p0, p1, p2, p3 = field[x_idx, y_idx, iz[0]], field[x_idx, y_idx, iz[1]], field[x_idx, y_idx, iz[2]], field[x_idx, y_idx, iz[3]]
        return cubic_weight(p0, p1, p2, p3, dz)

    def interp_y(x_idx):
        p0, p1, p2, p3 = interp_z(x_idx, iy[0]), interp_z(x_idx, iy[1]), interp_z(x_idx, iy[2]), interp_z(x_idx, iy[3])
        return cubic_weight(p0, p1, p2, p3, dy)

    val0, val1, val2, val3 = interp_y(ix[0]), interp_y(ix[1]), interp_y(ix[2]), interp_y(ix[3])
    f_interp = cubic_weight(val0, val1, val2, val3, dx)

    if use_limiter:
        x1 = jnp.minimum(x0 + 1, nx - 1)
        y1 = jnp.minimum(y0 + 1, ny - 1)
        z1 = jnp.minimum(z0 + 1, nz - 1)

        # Base case
        f_min = f_max = field[x0, y0, z0]

        # Chain the 7 other neighbors
        for i, j, k in [(x1,y0,z0), (x0,y1,z0), (x1,y1,z0), 
                        (x0,y0,z1), (x1,y0,z1), (x0,y1,z1), (x1,y1,z1)]:
            neighbor = field[i, j, k]
            f_min = jnp.minimum(f_min, neighbor)
            f_max = jnp.maximum(f_max, neighbor)

        return jnp.clip(f_interp, f_min, f_max)

    return f_interp