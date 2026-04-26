import jax.numpy as jnp

class CGridOperator3D:
    def __init__(self, grid):
        self.grid = grid
        self.m_factors = {
            loc: jnp.expand_dims(m_2d, axis=-1) 
            for loc, m_2d in self.grid.m_factors.items()
        }

    def _apply_padding(self, f, axis, from_loc, to_loc):
        pad_width = [(0, 0), (0, 0), (0, 0)]
        if from_loc == 'm' and to_loc in ['u', 'v', 'w']:
            pad_width[axis] = (1, 1)
            return jnp.pad(f, pad_width, mode='constant', constant_values=0.0)
        return f

    def diff(self, f, axis, from_loc, to_loc):
        delta = self.grid.delta[axis]
        derivative = jnp.diff(f, axis=axis) / delta
        derivative = self._apply_padding(derivative, axis, from_loc, to_loc)
        
        if axis in [0, 1]: 
            derivative = derivative * self.m_factors[to_loc]
        return derivative

    def avg(self, f, axis, from_loc, to_loc):
        avg_val = 0.5 * (f[:-1] + f[1:]) if axis == 0 else \
                  0.5 * (f[:, :-1] + f[:, 1:]) if axis == 1 else \
                  0.5 * (f[:, :, :-1] + f[:, :, 1:])
        return self._apply_padding(avg_val, axis, from_loc, to_loc)

def cubic_weight(p0, p1, p2, p3, t):
    return (-0.5*p0 + 1.5*p1 - 1.5*p2 + 0.5*p3) * t**3 + \
           (p0 - 2.5*p1 + 2.0*p2 - 0.5*p3) * t**2 + \
           (-0.5*p0 + 0.5*p2) * t + p1

def tensor_product_interp_3d(field, coords):
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
    return cubic_weight(val0, val1, val2, val3, dx)