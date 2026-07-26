import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import jax
import jax.numpy as jnp
import pytest
from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D

from suetes.physics.surface import BucketLSM
from suetes.physics.turbulence import McFarlaneVerticalDiffusion
from suetes.physics.microphysics import KesslerWarmRain

# Pull strict configurations from run_simulation_ablation.py
DX = 22000.0
DY = 22000.0
DZ = 500.0
NZ = 32
DT = 120.0

CONSTANTS = {
    'g': 9.81, 'Rd': 287.0, 'cp': 1004.0, 'cvd': 717.0,
    'p0': 100000.0, 'epsilon': 0.622,
}

def create_mock_environment(nx=3, ny=3):
    """Generates a scale-matched grid, operators, and a base standard atmosphere background."""
    grid = RegionalGrid3D(nx=nx, ny=ny, nz=NZ, dx=DX, dy=DY, dz=DZ, lat_center=47.5, lon_center=-97.0)
    operators = CGridOperator3D(grid)
    
    # Construct a stable hydrostatic background dictionary
    bg = {
        'dz_m_full': jnp.full((nx, ny, NZ), DZ),
        'dz_w_full': jnp.full((nx, ny, NZ+1), DZ),
        'dz_u': jnp.full((nx+1, ny, NZ), DZ),
        'dz_v': jnp.full((nx, ny+1, NZ), DZ),
        'th_v_w': jnp.linspace(280.0, 340.0, NZ+1)[None, None, :], # Stable stratification
    }
    return grid, operators, bg


def test_surface_fluxes_scale():
    grid, operators, bg = create_mock_environment(nx=3, ny=3)
    
    # State setup
    shape_m = (grid.nx, grid.ny, grid.nz)
    state = {
        'u': jnp.full((grid.nx+1, grid.ny, grid.nz), 20.0),
        'v': jnp.zeros((grid.nx, grid.ny+1, grid.nz)),
        'th_v': jnp.full(shape_m, 290.0),
        'q': jnp.full(shape_m, 0.001),
        'pi': jnp.full(shape_m, 1.0),
        'theta_surf': jnp.full((grid.nx, grid.ny), 305.0), # Warmer skin temperature
        'land_fraction': jnp.ones((grid.nx, grid.ny))      # 100% Land
    }
    
    lsm = BucketLSM(grid, operators, CONSTANTS)
    tends = lsm.get_tendencies(state, bg)
    
    # Assert physical expectations at mesoscale grid spacing
    assert jnp.all(tends['u'][:, :, 0] < 0)  # Frictional deceleration
    assert jnp.all(tends['th_v'][:, :, 0] > 0)  # Positive Sensible Heat Flux
    assert jnp.all(tends['th_v'][:, :, 1:] == 0) # Limited purely to the lowest surface layer


def test_microphysics_saturation():
    grid, operators, bg = create_mock_environment(nx=3, ny=3)
    shape_m = (grid.nx, grid.ny, grid.nz)
    
    # Enforce heavy supersaturation in a column
    state = {
        'th_v': jnp.full(shape_m, 280.0),
        'q': jnp.full(shape_m, 0.025),
        'q_c': jnp.zeros(shape_m),
        'q_r': jnp.zeros(shape_m),
        'pi': jnp.full(shape_m, 0.7), # Lower pressure aloft
    }
    
    micro = KesslerWarmRain(CONSTANTS)
    updated_state = micro.apply_update(state)

    assert jnp.all(updated_state['q'] < state['q'])       # Vapor consumed
    assert jnp.all(updated_state['th_v'] > state['th_v']) # Latent heating warming
    assert jnp.all(updated_state['q_c'] >= 0)            # Valid positive cloud space


def test_kessler_rain_evaporates_and_cools_unsaturated_air():
    grid, _, _ = create_mock_environment(nx=1, ny=1)
    shape_m = (grid.nx, grid.ny, grid.nz)
    state = {
        'th_v': jnp.full(shape_m, 300.0),
        'q': jnp.full(shape_m, 0.002),
        'q_c': jnp.zeros(shape_m),
        'q_r': jnp.full(shape_m, 0.001),
        'pi': jnp.full(shape_m, 1.0),
        'rho': jnp.full(shape_m, 1.15),
    }

    micro = KesslerWarmRain(CONSTANTS, dt=1.0, grid=grid)
    updated = micro.apply_update(state)

    assert jnp.all(updated['q_r'] < state['q_r'])
    assert jnp.all(updated['q'] > state['q'])
    assert jnp.all(updated['th_v'] < state['th_v'])


def test_vertical_diffusion_stability():
    grid, operators, bg = create_mock_environment(nx=3, ny=3)
    shape_m = (grid.nx, grid.ny, grid.nz)
    
    # Base state with a sharp wind jet at level 15
    u_base = jnp.zeros((grid.nx+1, grid.ny, grid.nz))
    u_base = u_base.at[:, :, 15].set(10.0) 
    
    state_unstable = {
        'u': u_base,
        'v': jnp.zeros((grid.nx, grid.ny+1, grid.nz)),
        # Decreasing theta_v with height (unstable)
        'th_v': jnp.broadcast_to(jnp.linspace(300.0, 280.0, grid.nz)[None, None, :], shape_m)
    }
    
    state_stable = {
        'u': u_base,
        'v': jnp.zeros((grid.nx, grid.ny+1, grid.nz)),
        # Strongly increasing theta_v with height (stable inversion)
        'th_v': jnp.broadcast_to(jnp.linspace(280.0, 320.0, grid.nz)[None, None, :], shape_m)
    }
    
    vd = McFarlaneVerticalDiffusion(grid, operators, CONSTANTS)
    
    tends_unstable = vd.get_tendencies(state_unstable, bg)
    tends_stable = vd.get_tendencies(state_stable, bg)
    
    # Calculate the magnitude of the momentum diffusion at the jet core
    diff_unstable = jnp.abs(tends_unstable['u'][1, 1, 15])
    diff_stable = jnp.abs(tends_stable['u'][1, 1, 15])
    
    # Physical Assertion: Unstable stratification should result in significantly more mixing
    assert diff_unstable > diff_stable * 5.0
    
    # Ensure surface fluxes are strictly zeroed out (handled by BucketLSM instead)
    assert jnp.all(tends_unstable['u'][:, :, 0] == 0.0)


def main():
    test_surface_fluxes_scale()
    test_microphysics_saturation()
    test_vertical_diffusion_stability()
    print("All tests passed.")

if __name__ == "__main__":
    main()
