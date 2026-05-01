import jax.numpy as jnp
from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.boundaries import DaviesSponge

nx, ny, nz = 32, 32, 15
dx, dy, dz = 1000.0, 1000.0, 500.0
grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, 45.0, 0.0)
sponge = DaviesSponge(grid, sponge_depth=5)
model_state = {'u': jnp.ones((nx+1, ny, nz)) * 10.0}
ext_state = {'u': jnp.zeros((nx+1, ny, nz))}
blended = sponge.blend(model_state, ext_state)
print("Center:", blended['u'][nx//2, ny//2, nz//2])
print("Edge:", blended['u'][0, ny//2, nz//2])
print("Mask edge:", sponge.masks['u'][0, ny//2, nz//2])
