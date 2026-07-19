"""
=============================================================================
         SUETES: SISL 3D RISING BUBBLE SELF-CONVERGENCE SUITE
=============================================================================
This benchmark evaluates self-convergence of the Semi-Implicit Semi-Lagrangian
(SISL) dynamical core on the 3D Rising Bubble test case (`build_dynamical_core`).

Two timestep scaling regimes are supported and verified:
  - dt_mode="scaling": Timestep dt scales linearly with grid spacing dx (dt ~ dx),
    evaluating combined spatial and temporal convergence.
  - dt_mode="constant": Timestep dt is held constant across grid resolutions,
    isolating spatial operator accuracy and convergence.
=============================================================================
"""

import os
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from suetes.shared.driver import Simulation
from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.steppers import build_dynamical_core

# Global Physical Constants
g, cp, cvd, Rd, p0 = 9.81, 1004.0, 717.0, 287.0, 100000.0


def run_bubble_at_resolution(dx, dt):
    nx = int(10000 / dx)
    nz = int(10000 / dx)
    ny = 3

    core_kwargs = {
        "dt": dt,
        "nu_div_factor": 0.0,
        "nu_h_factor": 0.0,
        "damp_height": 7500.0,
        "max_damp": 0.05,
        "alpha": 0.5,
        "solver_tol": 1e-10,
        "solver_maxiter": 100,
        "solver_restart": 20
    }

    grid = RegionalGrid3D(nx, ny, nz, dx, dx, dx, lat_center=0.0, lon_center=0.0)
    op = CGridOperator3D(grid)
    constants = {'g': g, 'cp': cp, 'Rd': Rd, 'cvd': cvd, 'p0': p0}

    tmp_phys = Euler3D(grid, op, constants, dt=dt)
    bg_ref = {
        'rho': tmp_phys.c['p0'] / (tmp_phys.c['Rd'] * tmp_phys.theta_bg) * \
               (tmp_phys.pi_bg ** (tmp_phys.c['cvd'] / tmp_phys.c['Rd'])),
        'pi': tmp_phys.pi_bg,
        'th_v': tmp_phys.theta_bg
    }

    state = {
        'u': jnp.zeros((nx + 1, ny, nz)),
        'v': jnp.zeros((nx, ny + 1, nz)),
        'w': jnp.zeros((nx, ny, nz + 1)),
        'pi': bg_ref['pi'],
        'eta_dot': jnp.zeros((nx, ny, nz + 1)),
        'rho': bg_ref['rho'],
    }

    X, Y, Z = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_m, indexing='ij')
    bubble = jnp.where((X**2 + (Z - 2000.0)**2) <= 1500.0**2,
                       2.0 * jnp.cos(0.5 * jnp.pi * jnp.sqrt(X**2 + (Z - 2000.0)**2) / 1500.0)**2, 0.0)

    state['th_v'] = bg_ref['th_v'] + bubble
    state['rho'] = constants['p0'] / (constants['Rd'] * state['th_v']) * \
                   (bg_ref['pi'] ** (constants['cvd'] / constants['Rd']))

    stepper, dt = build_dynamical_core(
        core_type="sisl", grid=grid, operators=op, constants=constants,
        initial_state=state, **core_kwargs
    )

    sim = Simulation(step_fn=lambda s, i: (stepper.step(s, i * dt, None, lambda x, f: x), jnp.max(jnp.abs(s['w']))), dt=dt)
    return sim.run(state, t_start=0.0, t_end=200.0, chunk_steps=int(200.0 / dt))


def test_bubble_convergence(dt_mode="scaling", resolutions=[250.0, 125.0, 62.5]):
    print(f"\n=======================================================")
    print(f" SISL 3D RISING BUBBLE SELF-CONVERGENCE (dt_mode={dt_mode})")
    print("=======================================================")

    if dt_mode == "scaling":
        dts = [10.0, 5.0, 2.5]
    elif dt_mode == "constant":
        dts = [0.5, 0.5, 0.5]
    else:
        raise ValueError(f"Unknown dt_mode: {dt_mode}")

    print(f"Resolutions (dx): {resolutions} meters")
    print(f"Timesteps   (dt): {dts} seconds")

    res_250 = run_bubble_at_resolution(resolutions[0], dts[0])
    res_125 = run_bubble_at_resolution(resolutions[1], dts[1])
    res_062 = run_bubble_at_resolution(resolutions[2], dts[2])

    def block_average_2d(field_2d, factor):
        nx, nz = field_2d.shape
        return jnp.mean(field_2d.reshape(nx // factor, factor, nz // factor, factor), axis=(1, 3))

    err_coarse = float(jnp.sqrt(jnp.mean((block_average_2d(res_125['th_v'][:, 1, :], 2) - res_250['th_v'][:, 1, :])**2)))
    err_fine = float(jnp.sqrt(jnp.mean((block_average_2d(res_062['th_v'][:, 1, :], 4) - block_average_2d(res_125['th_v'][:, 1, :], 2))**2)))
    order = np.log2(err_coarse / err_fine)

    # Interior-only (cropped) convergence to avoid boundary/sponge impacts
    crop = 6
    val_250 = res_250['th_v'][crop:-crop, 1, crop:-crop]
    val_125_avg = block_average_2d(res_125['th_v'][:, 1, :], 2)[crop:-crop, crop:-crop]
    val_062_avg = block_average_2d(res_062['th_v'][:, 1, :], 4)[crop:-crop, crop:-crop]

    err_coarse_int = float(jnp.sqrt(jnp.mean((val_125_avg - val_250)**2)))
    err_fine_int = float(jnp.sqrt(jnp.mean((val_062_avg - val_125_avg)**2)))
    order_int = np.log2(err_coarse_int / err_fine_int)

    print(f"Global L2 Error (250m vs 125m):        {err_coarse:.2e}")
    print(f"Global L2 Error (125m vs  62m):        {err_fine:.2e}")
    print(f"Global Empirical Order of Convergence:   {order:.2f}")
    print(f"Interior L2 Error (250m vs 125m):      {err_coarse_int:.2e}")
    print(f"Interior L2 Error (125m vs  62m):      {err_fine_int:.2e}")
    print(f"Interior Empirical Order of Convergence: {order_int:.2f} (Expected ~2.00)")


if __name__ == "__main__":
    test_bubble_convergence(dt_mode="scaling")
    test_bubble_convergence(dt_mode="constant")
