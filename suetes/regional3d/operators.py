"""
Discrete Spatial Operators Module.

Contains the finite-difference stencils, spatial averaging routines, and 
high-order interpolators required for integrating PDEs on an Arakawa C-grid.
"""

import jax.numpy as jnp
import jax.scipy.ndimage as jnd

class CGridOperator3D:
    """
    Provides discrete derivative and averaging operators tailored to the 
    staggered locations of the Arakawa C-grid.
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

        For momentum fields (u, v, w), we pad with the 'edge' value (first/last
        row) to implement a Neumann boundary condition (zero gradient).
        """
        pad_width = [(0, 0), (0, 0), (0, 0)]
        if from_loc == 'm' and to_loc in ['u', 'v', 'w']:
            pad_width[axis] = (1, 1)
            return jnp.pad(f, pad_width, mode='edge')
        return f

    def diff(self, f, axis, from_loc, to_loc):
        r"""
        Computes the spatial derivative of a field across a staggered boundary, 
        incorporating the local map factor.

        For example, computing the $x$-derivative of a field $\pi$ defined at 
        mass points ($m$) onto the $u$-velocity faces:

        $$ \left( \frac{\partial \pi}{\partial x} \right)_u \approx m_u \frac{\pi_{i} - \pi_{i-1}}{\Delta x} $$

        Args:
            f (jnp.ndarray): The input field.
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

        $$ \bar{f}^x \approx \frac{1}{2}(f_{i+1/2} + f_{i-1/2}) $$

        Args:
            f (jnp.ndarray): The input field.
            axis (int): The axis of averaging (0 for x, 1 for y, 2 for z).
            from_loc (str): Original grid staggering.
            to_loc (str): Target grid staggering.

        Returns:
            jnp.ndarray: The averaged field mapped to `to_loc`.
        """
        avg_val = 0.5 * (f[:-1] + f[1:]) if axis == 0 else \
                  0.5 * (f[:, :-1] + f[:, 1:]) if axis == 1 else \
                  0.5 * (f[:, :, :-1] + f[:, :, 1:])
        return self._apply_padding(avg_val, axis, from_loc, to_loc)

def cubic_weight(p0, p1, p2, p3, t):
    r"""
    Evaluates a 1D cubic spline interpolant.

    $$ f(t) = \left(-\frac{1}{2}p_0 + \frac{3}{2}p_1 - \frac{3}{2}p_2 + \frac{1}{2}p_3\right)t^3 + \dots $$
    """
    return (-0.5*p0 + 1.5*p1 - 1.5*p2 + 0.5*p3) * t**3 + \
           (p0 - 2.5*p1 + 2.0*p2 - 0.5*p3) * t**2 + \
           (-0.5*p0 + 0.5*p2) * t + p1

def tensor_product_interp_3d(field, coords, use_limiter=False):
    """
    Performs 3D tricubic interpolation using a 64-point local stencil.
    
    Crucial for the Semi-Lagrangian advection scheme to evaluate the field value 
    at continuous departure points $\mathbf{x}_d$ without excessive numerical damping.

    Args:
        field (jnp.ndarray): The 3D data grid to interpolate from.
        coords (tuple): A tuple $(x, y, z)$ of fractional continuous indices.
        use_limiter (bool): If True, applies a quasi-monotone limiter bounding 
            the interpolated value by its immediate 8-point neighborhood to 
            prevent unphysical extrema (e.g., negative water vapor).
            
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