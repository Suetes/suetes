import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import jax
import jax.numpy as jnp
from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.physics.turbulence import SmagorinskyLillySGS

# 3D Grid Parameters for Turbulence Testing
DX = 500.0
DY = 500.0
DZ = 100.0  # High vertical resolution to capture shear
NX = 20
NY = 20
NZ = 20

CONSTANTS = {'g': 9.81, 'Rd': 287.0, 'cp': 1004.0, 'cvd': 717.0, 'p0': 100000.0}

def create_3d_environment():
    grid = RegionalGrid3D(nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, dz=DZ, lat_center=45.0, lon_center=0.0)
    operators = CGridOperator3D(grid)
    
    bg = {
        'dz_m_full': jnp.full((NX, NY, NZ), DZ),
        'dz_w_full': jnp.full((NX, NY, NZ+1), DZ),
        'dz_u': jnp.full((NX+1, NY, NZ), DZ),
        'dz_v': jnp.full((NX, NY+1, NZ), DZ),
        'th_v_w': jnp.linspace(290.0, 310.0, NZ+1)[None, None, :], 
    }
    return grid, operators, bg

def test_smagorinsky_3d_shear():
    grid, operators, bg = create_3d_environment()
    shape_m = (grid.nx, grid.ny, grid.nz)
    
    # 1. Initialize a neutral base state
    state = {
        'u': jnp.zeros((grid.nx+1, grid.ny, grid.nz)),
        'v': jnp.zeros((grid.nx, grid.ny+1, grid.nz)),
        'w': jnp.zeros((grid.nx, grid.ny, grid.nz+1)),
        # Unstable layer in the middle to ensure Ri < Ri_c (f_Ri > 0)
        'th_v': jnp.broadcast_to(jnp.linspace(300.0, 290.0, grid.nz)[None, None, :], shape_m)
    }
    
    # 2. Inject a 3D localized wind shear anomaly (a "jet core")
    # Centered at x=10, y=10, z=10
    jet_u = jnp.zeros_like(state['u'])
    jet_u = jet_u.at[9:12, 9:12, 9:12].set(15.0)
    state['u'] = jet_u
    
    # Instantiate the SGS scheme
    sgs = SmagorinskyLillySGS(grid, operators, CONSTANTS, dt=100.0, Cs=0.15)
    
    # Calculate tendencies
    tends = sgs.get_tendencies(state, bg)
    
    # 3. Validation
    # The diffusion scheme should act to smooth out the jet core.
    # Therefore, inside the jet core, the tendency must be negative (decelerating).
    core_tendency = tends['u'][10, 10, 10]
    
    # Just outside the jet core (e.g., above or below), the tendency must be positive 
    # as momentum is diffused outward into the slower surrounding air.
    edge_tendency = tends['u'][10, 10, 12]
    
    print(f"Jet Core U-Tendency: {core_tendency:.4f} m/s^2")
    print(f"Jet Edge U-Tendency: {edge_tendency:.4f} m/s^2")
    
    assert core_tendency < 0.0, "SGS failed to decelerate the shear peak."
    assert edge_tendency > 0.0, "SGS failed to diffuse momentum outward."

if __name__ == "__main__":
    test_smagorinsky_3d_shear()
    print("Smagorinsky 3D SGS test passed successfully.")