"""
=============================================================================
         SUETES: SISL COMPONENT & OPERATOR CONVERGENCE SUITE
=============================================================================
This benchmark and unit verification suite evaluates every individual building
block and spatial/temporal operator of the Semi-Implicit Semi-Lagrangian (SISL)
dynamical core:

  - Unit 1: Tricubic Spatial Interpolation (tensor_product_interp_3d)
  - Unit 2: C-Grid Spatial Operators (CGridOperator3D.diff & .avg)
  - Unit 3: Midpoint Trajectory Solver (compute_departure_indices)
  - Unit 4: GMRES Preconditioned Matrix Inversion (SemiImplicitSolver3D.solve)
  - Unit 5: Explicit Spatial Tendency Convergence (Euler3D.get_tendencies)
  - Unit 6: Implicit Linear Matrix Operator Convergence (Euler3D.linear_operator)
  - Unit 7: 3D Semi-Lagrangian Advection Convergence (advect_cubic)
  - Unit 8: Flux-Form Split-Explicit Advection Convergence (advect_3d_split)
=============================================================================
"""

import os
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D, tensor_product_interp_3d
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.steppers import (
    SemiLagrangianAdvector3D,
    SemiImplicitSolver3D,
    FluxFormAdvector,
)

# Global Domain & Constants
Lx, Ly, Lz = 2000000.0, 2000000.0, 10000.0
g, cp, cvd, Rd, p0 = 9.81, 1004.0, 717.0, 287.0, 100000.0


def compute_safe_err(final, initial, mask):
    """Computes masked L2 RMS error over active grid points."""
    return float(jnp.sqrt(jnp.sum((final - initial)**2 * mask) / jnp.sum(mask)))


def get_exact_sine_field(x, y, z, lx=Lx, ly=Ly, lz=Lz):
    return jnp.sin(2.0 * jnp.pi * x / lx) * jnp.cos(2.0 * jnp.pi * y / ly) * jnp.sin(2.0 * jnp.pi * z / lz)


# =============================================================================
# UNIT 1: TRICUBIC SPATIAL INTERPOLATION
# =============================================================================
def test_unit_1_interpolation(resolutions=[16, 32, 64]):
    print("\n=======================================================")
    print(" UNIT 1: TRICUBIC SPATIAL INTERPOLATION CONVERGENCE")
    print("=======================================================")
    lx, ly, lz = 10000.0, 10000.0, 10000.0
    results = []

    for N in resolutions:
        dx, dy, dz = lx / N, ly / N, lz / N
        grid = RegionalGrid3D(N, N, N, dx, dy, dz, lat_center=0.0, lon_center=0.0)
        X, Y, Z = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_m, indexing='ij')
        field = get_exact_sine_field(X, Y, Z, lx, ly, lz)

        # Fractional target offset (X + 0.37*dx, Y + 0.23*dy, Z + 0.45*dz)
        X_target = X + 0.37 * dx
        Y_target = Y + 0.23 * dy
        Z_target = Z + 0.45 * dz

        coords = (
            (X_target + lx / 2.0) / dx - 0.5,
            (Y_target + ly / 2.0) / dy - 0.5,
            Z_target / dz - 0.5
        )

        interp_field = tensor_product_interp_3d(field, coords, use_limiter=False)
        exact_field = get_exact_sine_field(X_target, Y_target, Z_target, lx, ly, lz)

        margin = N // 4
        err = jnp.sqrt(jnp.mean((interp_field[margin:-margin, margin:-margin, margin:-margin] -
                                 exact_field[margin:-margin, margin:-margin, margin:-margin])**2))
        results.append((dx, float(err)))
        print(f"N = {N:2d} | dx = {dx:6.1f}m | L2 Error: {float(err):.2e}")

    print("  => Rates (N->2N):")
    for i in range(len(resolutions) - 1):
        rate = np.log2(results[i][1] / results[i+1][1])
        print(f"     {resolutions[i]:2d}->{resolutions[i+1]:2d}: Rate = {rate:.2f} (Expected ~3.00-4.00)")


# =============================================================================
# UNIT 2: C-GRID SPATIAL OPERATORS (DIFF & AVG)
# =============================================================================
def test_unit_2_operators(resolutions=[16, 32, 64]):
    print("\n=======================================================")
    print(" UNIT 2: C-GRID OPERATORS (DIFF & AVG) CONVERGENCE")
    print("=======================================================")
    lx, ly, lz = 10000.0, 10000.0, 10000.0
    results_diff = []
    results_avg = []

    for N in resolutions:
        dx, dy, dz = lx / N, ly / N, lz / N
        grid = RegionalGrid3D(N, N, N, dx, dy, dz, lat_center=0.0, lon_center=0.0)
        # Force exact cartesian for pure discrete derivative test
        for k in grid.m_factors:
            grid.m_factors[k] = jnp.ones_like(grid.m_factors[k])
        op = CGridOperator3D(grid)

        X, Y, Z = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_m, indexing='ij')
        f_m = get_exact_sine_field(X, Y, Z, lx, ly, lz)

        df_dx_num = op.diff(f_m, axis=0, from_loc='m', to_loc='u')
        X_u, Y_u, Z_u = jnp.meshgrid(grid.x_c, grid.y_m, grid.z_m, indexing='ij')
        df_dx_exact = (2.0 * jnp.pi / lx) * jnp.cos(2.0 * jnp.pi * X_u / lx) * jnp.cos(2.0 * jnp.pi * Y_u / ly) * jnp.sin(2.0 * jnp.pi * Z_u / lz)

        margin = N // 4
        err_diff = float(jnp.sqrt(jnp.mean((df_dx_num[margin:-margin, margin:-margin, margin:-margin] -
                                            df_dx_exact[margin:-margin, margin:-margin, margin:-margin])**2)))
        results_diff.append((dx, err_diff))

        avg_num = op.avg(f_m, axis=0, from_loc='m', to_loc='u')
        # Compare with the continuous field at the face.  Including the cosine
        # attenuation factor here would instead test an exact discrete identity
        # and leave only roundoff error, from which no convergence rate exists.
        avg_exact = get_exact_sine_field(X_u, Y_u, Z_u, lx, ly, lz)
        err_avg = float(jnp.sqrt(jnp.mean((avg_num[margin:-margin, margin:-margin, margin:-margin] -
                                           avg_exact[margin:-margin, margin:-margin, margin:-margin])**2)))
        results_avg.append((dx, err_avg))

        print(f"N = {N:2d} | L2 Diff Error: {err_diff:.2e} | L2 Avg Error: {err_avg:.2e}")

    print("  => Diff Rates (N->2N):")
    for i in range(len(resolutions) - 1):
        rate = np.log2(results_diff[i][1] / results_diff[i+1][1])
        print(f"     {resolutions[i]:2d}->{resolutions[i+1]:2d}: Rate = {rate:.2f} (Expected ~2.00)")
    print("  => Avg Rates (N->2N):")
    for i in range(len(resolutions) - 1):
        rate = np.log2(results_avg[i][1] / results_avg[i+1][1])
        print(f"     {resolutions[i]:2d}->{resolutions[i+1]:2d}: Rate = {rate:.2f} (Expected ~2.00)")


# =============================================================================
# UNIT 3: MIDPOINT TRAJECTORY SOLVER
# =============================================================================
def test_unit_3_trajectories(dt_values=(400.0, 200.0, 100.0, 50.0)):
    print("\n=======================================================")
    print(" UNIT 3: MIDPOINT TRAJECTORY SOLVER CONVERGENCE")
    print("=======================================================")
    results = []

    # For dx/dt = a*x, the exact backward characteristic is
    # x_d = x_a*exp(-a*dt). Linear interpolation of this velocity is exact, so
    # the measured error isolates the implicit midpoint trajectory integrator.
    r = 64
    dx, dy, dz = Lx / r, Ly / r, Lz / r
    grid = RegionalGrid3D(r, r, r, dx, dy, dz, lat_center=0.0, lon_center=0.0)
    for k in grid.m_factors:
        grid.m_factors[k] = jnp.ones_like(grid.m_factors[k])

    c_dict = {'g': g, 'cp': cp, 'cvd': cvd, 'Rd': Rd, 'p0': p0}
    physics = Euler3D(grid, CGridOperator3D(grid), c_dict, dt=1.0)
    X_u, _, _ = jnp.meshgrid(grid.x_c, grid.y_m, grid.z_m, indexing='ij')
    X_m, _, _ = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_m, indexing='ij')
    strain_rate = 1.0e-4
    state = {
        'u': strain_rate * X_u,
        'v': jnp.zeros((r, r + 1, r)),
        'w': jnp.zeros((r, r, r + 1)),
        'eta_dot': jnp.zeros((r, r, r + 1)),
    }

    for dt_test in dt_values:
        advector = SemiLagrangianAdvector3D(grid, physics, dt=dt_test)
        coords_num = advector.compute_departure_indices(
            state, loc='m', iterations=8
        )
        exact_x_phys = X_m * jnp.exp(-strain_rate * dt_test)
        exact_x_idx = (exact_x_phys + Lx / 2.0) / dx - 0.5

        # Stay far from nearest-neighbour boundary extension.
        margin = r // 4
        error_phys = dx * jnp.sqrt(jnp.mean(
            (coords_num[0, margin:-margin, margin:-margin, margin:-margin]
             - exact_x_idx[margin:-margin, margin:-margin, margin:-margin]) ** 2
        ))
        results.append((dt_test, float(error_phys)))
        print(f"dt = {dt_test:6.1f}s | Exact-characteristic error: {error_phys:.2e} m")

    print("  => Temporal rates (dt->dt/2):")
    for i in range(len(results) - 1):
        rate = np.log2(results[i][1] / results[i + 1][1])
        print(
            f"     {results[i][0]:6.1f}->{results[i+1][0]:6.1f}s: "
            f"Rate = {rate:.2f} (Expected ~3.00 local order)"
        )


# =============================================================================
# UNIT 4: GMRES PRECONDITIONED IMPLICIT INVERSION
# =============================================================================
def test_unit_4_gmres_inversion(resolutions=[8, 16, 32, 64]):
    print("\n=======================================================")
    print(" UNIT 4: GMRES IMPLICIT SOLVER INVERSION ACCURACY")
    print("=======================================================")
    results = []

    for r in resolutions:
        dx, dy, dz = Lx / r, Ly / r, Lz / r
        grid = RegionalGrid3D(r, r, r, dx, dy, dz, lat_center=60.0, lon_center=30.0)
        op = CGridOperator3D(grid)
        c_dict = {'g': g, 'cp': cp, 'cvd': cvd, 'Rd': Rd, 'p0': p0}

        dt_test = 60.0
        alpha_test = 0.55
        physics = Euler3D(grid, op, c_dict, dt=dt_test, N_bv=0.01)
        solver = SemiImplicitSolver3D(physics, dt_test, alpha=alpha_test, solver_tol=1e-12)

        key = jax.random.PRNGKey(42)
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)

        x_true = {
            'u': jax.random.uniform(k1, (r + 1, r, r)) * 5.0,
            'v': jax.random.uniform(k2, (r, r + 1, r)) * 5.0,
            'w': jax.random.uniform(k3, (r, r, r + 1)) * 0.1,
            'pi': jax.random.uniform(k4, (r, r, r)) * 0.001,
            'eta_dot': jax.random.uniform(k5, (r, r, r + 1)) * 0.001
        }
        x_true['w'] = x_true['w'].at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
        x_true['eta_dot'] = x_true['eta_dot'].at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)

        rho_bg = p0 / (Rd * physics.theta_bg) * (physics.pi_bg ** (cvd / Rd))
        bg_ref = {'rho': rho_bg, 'pi': physics.pi_bg, 'th_v': physics.theta_bg}
        bg = physics.precompute_bg(bg_ref)

        rhs_prime = physics.linear_operator(x_true, bg, dt=dt_test, alpha=alpha_test)
        x_num = solver.solve(rhs_prime, bg, x0=None)
        relative_residual = float(
            solver.relative_residual(x_num, rhs_prime, bg)
        )

        mask_u = jnp.ones_like(x_true['u'], dtype=bool)
        mask_w = jnp.ones_like(x_true['w'], dtype=bool)
        mask_pi = jnp.ones_like(x_true['pi'], dtype=bool)
        err_u = compute_safe_err(x_num['u'], x_true['u'], mask_u)
        err_w = compute_safe_err(x_num['w'], x_true['w'], mask_w)
        err_pi = compute_safe_err(x_num['pi'], x_true['pi'], mask_pi)

        results.append((dx, err_u, err_w, err_pi))
        print(
            f"N = {r:3d} | GMRES Inversion Error -> u: {err_u:.2e} "
            f"| w: {err_w:.2e} | pi: {err_pi:.2e} "
            f"| relative residual: {relative_residual:.2e}"
        )
        assert np.isfinite(relative_residual)
        assert relative_residual < 1.0e-9, (
            f"GMRES did not satisfy the implicit system at N={r}: "
            f"relative residual={relative_residual:.3e}"
        )


# =============================================================================
# UNIT 5 & 6 & 7: SPATIAL TENDENCY, LINEAR OPERATOR & ADVECTION PHASE TESTS
# =============================================================================
def test_units_5_6_7_component_phases(resolutions=[16, 32, 64]):
    print("\n=======================================================")
    print(" UNITS 5-7: EXPLICIT TENDENCIES, LINEAR OPERATOR & ADVECTION")
    print("=======================================================")
    theta_0 = 300.0
    N_bv = 0.01

    def u_f(x, y, z): return 1.0 * jnp.sin(2.0 * jnp.pi * x / Lx)**2 * jnp.sin(2.0 * jnp.pi * y / Ly)**2 * jnp.sin(jnp.pi * z / Lz)**2
    def v_f(x, y, z): return 1.0 * jnp.sin(2.0 * jnp.pi * x / Lx)**2 * jnp.sin(2.0 * jnp.pi * y / Ly)**2 * jnp.sin(jnp.pi * z / Lz)**2
    def w_f(x, y, z): return 0.1 * jnp.sin(2.0 * jnp.pi * x / Lx)**2 * jnp.sin(2.0 * jnp.pi * y / Ly)**2 * jnp.sin(jnp.pi * z / Lz)**2
    def pip_f(x, y, z): return 0.01 * jnp.sin(2.0 * jnp.pi * x / Lx)**2 * jnp.sin(2.0 * jnp.pi * y / Ly)**2 * jnp.sin(jnp.pi * z / Lz)**2
    def thvp_f(x, y, z): return 5.0 * jnp.sin(2.0 * jnp.pi * x / Lx)**2 * jnp.sin(2.0 * jnp.pi * y / Ly)**2 * jnp.sin(jnp.pi * z / Lz)**2
    def th_bg_f(z): return theta_0 * jnp.exp((N_bv**2 / g) * z)
    def pi_bg_f(z): return 1.0 + (g**2 / (cp * theta_0 * N_bv**2)) * (jnp.exp(-N_bv**2 * z / g) - 1.0)
    def rho_bg_f(z): return p0 / (Rd * th_bg_f(z)) * (pi_bg_f(z) ** (cvd / Rd))

    @jax.jit
    def ana_tend_u(x, y, z, f_val):
        m = 1.0 + (x**2 + y**2) / (4.0 * 6371229.0**2)
        u, v = u_f(x, y, z), v_f(x, y, z)
        th_v = th_bg_f(z) + thvp_f(x, y, z)
        dpip_dx = jax.grad(lambda xv: pip_f(xv, y, z))(x)
        dm_dx = x / (2.0 * 6371229.0**2)
        dm_dy = y / (2.0 * 6371229.0**2)
        metric = (v * dm_dx - u * dm_dy) * v
        return -cp * th_v * m * dpip_dx + f_val * v + metric

    @jax.jit
    def ana_tend_w(x, y, z):
        dpip_dz = jax.grad(lambda zv: pip_f(x, y, zv))(z)
        return (
            -cp * (th_bg_f(z) + thvp_f(x, y, z)) * dpip_dz
            + g * (thvp_f(x, y, z) / th_bg_f(z))
        )

    @jax.jit
    def ana_imp_u(x, y, z, f_val):
        m = 1.0 + (x**2 + y**2) / (4.0 * 6371229.0**2)
        dpip_dx = jax.grad(lambda xv: pip_f(xv, y, z))(x)
        return -cp * th_bg_f(z) * m * dpip_dx + f_val * v_f(x, y, z)

    @jax.jit
    def ana_imp_w(x, y, z):
        dpip_dz = jax.grad(lambda zv: pip_f(x, y, zv))(z)
        return -cp * th_bg_f(z) * dpip_dz

    vmap_u = jax.vmap(jax.vmap(jax.vmap(ana_tend_u, in_axes=(0, 0, 0, None)), in_axes=(0, 0, 0, 0)), in_axes=(0, 0, 0, 0))
    vmap_w = jax.vmap(jax.vmap(jax.vmap(ana_tend_w, in_axes=(0, 0, 0)), in_axes=(0, 0, 0)), in_axes=(0, 0, 0))
    vmap_u_imp = jax.vmap(jax.vmap(jax.vmap(ana_imp_u, in_axes=(0, 0, 0, None)), in_axes=(0, 0, 0, 0)), in_axes=(0, 0, 0, 0))
    vmap_w_imp = jax.vmap(jax.vmap(jax.vmap(ana_imp_w, in_axes=(0, 0, 0)), in_axes=(0, 0, 0)), in_axes=(0, 0, 0))

    results = []
    for r in resolutions:
        dx, dy, dz = Lx / r, Ly / r, Lz / r
        grid = RegionalGrid3D(r, r, r, dx, dy, dz, lat_center=60.0, lon_center=30.0)
        op = CGridOperator3D(grid)
        c_dict = {'g': g, 'cp': cp, 'cvd': cvd, 'Rd': Rd, 'p0': p0}
        dt = 60.0
        alpha = 0.55
        physics = Euler3D(grid, op, c_dict, dt=dt, N_bv=N_bv)
        advector = SemiLagrangianAdvector3D(grid, physics, dt=dt)

        X_u, Y_u, Z_u = jnp.meshgrid(grid.x_c, grid.y_m, grid.z_m, indexing='ij')
        X_v, Y_v, Z_v = jnp.meshgrid(grid.x_m, grid.y_c, grid.z_m, indexing='ij')
        X_w, Y_w, Z_w = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_c, indexing='ij')
        X_m, Y_m, Z_m = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_m, indexing='ij')

        bg_ref = {'rho': rho_bg_f(Z_m), 'pi': physics.pi_bg, 'th_v': physics.theta_bg}
        bg = physics.precompute_bg(bg_ref)

        state_prime = {
            'u': u_f(X_u, Y_u, Z_u), 'v': v_f(X_v, Y_v, Z_v), 'w': w_f(X_w, Y_w, Z_w),
            'pi': pip_f(X_m, Y_m, Z_m), 'th_v_prime_m': thvp_f(X_m, Y_m, Z_m),
            'th_v_prime_w': op.avg(thvp_f(X_m, Y_m, Z_m), axis=2, from_loc='m', to_loc='w')
        }
        state_prime['th_v_prime_u'] = op.avg(state_prime['th_v_prime_m'], axis=0, from_loc='m', to_loc='u')
        state_prime['th_v_prime_v'] = op.avg(state_prime['th_v_prime_m'], axis=1, from_loc='m', to_loc='v')
        state_prime['eta_dot'] = state_prime['w'] / grid.dz_w_full

        vertical_buffer = 2000.0
        mask_u = (
            (jnp.abs(X_u) < Lx / 2.0 - 400000.0)
            & (jnp.abs(Y_u) < Ly / 2.0 - 400000.0)
            & (Z_u >= vertical_buffer) & (Z_u <= Lz - vertical_buffer)
        )
        mask_w = (
            (jnp.abs(X_w) < Lx / 2.0 - 400000.0)
            & (jnp.abs(Y_w) < Ly / 2.0 - 400000.0)
            & (Z_w >= vertical_buffer) & (Z_w <= Lz - vertical_buffer)
        )
        mask_m = (
            (jnp.abs(X_m) < Lx / 2.0 - 400000.0)
            & (jnp.abs(Y_m) < Ly / 2.0 - 400000.0)
            & (Z_m >= vertical_buffer) & (Z_m <= Lz - vertical_buffer)
        )

        # Unit 5: Explicit Tendencies
        tends_exp = physics.get_tendencies(state_prime, bg, is_explicit=True)
        err_u_exp = compute_safe_err(tends_exp['u'], vmap_u(X_u, Y_u, Z_u, grid.f_u), mask_u)
        err_w_exp = compute_safe_err(tends_exp['w'], vmap_w(X_w, Y_w, Z_w), mask_w)

        # Unit 6: Implicit Matrix Operator
        L_out = physics.linear_operator(state_prime, bg, dt=dt, alpha=alpha)
        t_lin_u = (state_prime['u'] - L_out['u']) / (alpha * dt)
        t_lin_w = (state_prime['w'] - L_out['w']) / (alpha * dt)
        err_u_imp = compute_safe_err(t_lin_u, vmap_u_imp(X_u, Y_u, Z_u, grid.f_u), mask_u)
        err_w_imp = compute_safe_err(t_lin_w, vmap_w_imp(X_w, Y_w, Z_w), mask_w)

        # Unit 7: Semi-Lagrangian Advection
        coords_m = advector.compute_departure_indices(state_prime, loc='m', iterations=3)
        q_adv = advector.advect_cubic(state_prime['th_v_prime_m'], coords_m, use_limiter=False)
        q_ana = thvp_f(X_m - dt * op.avg(state_prime['u'], axis=0, from_loc='u', to_loc='m'),
                       Y_m - dt * op.avg(state_prime['v'], axis=1, from_loc='v', to_loc='m'),
                       Z_m - dt * op.avg(state_prime['w'], axis=2, from_loc='w', to_loc='m'))
        err_q_adv = compute_safe_err(q_adv, q_ana, mask_m)

        results.append((dx, err_u_exp, err_w_exp, err_u_imp, err_w_imp, err_q_adv))
        print(f"N = {r:2d} | Exp u: {err_u_exp:.2e} | Imp u: {err_u_imp:.2e} | Adv q: {err_q_adv:.2e}")

    print("  => Explicit Tendency Rates (u, w):")
    for i in range(len(resolutions) - 1):
        ru = np.log2(results[i][1] / results[i+1][1])
        rw = np.log2(results[i][2] / results[i+1][2])
        print(f"     {resolutions[i]:2d}->{resolutions[i+1]:2d}: u={ru:.2f}, w={rw:.2f}")

    print("  => Implicit Linear Operator Rates (u, w):")
    for i in range(len(resolutions) - 1):
        ru = np.log2(results[i][3] / results[i+1][3])
        rw = np.log2(results[i][4] / results[i+1][4])
        print(f"     {resolutions[i]:2d}->{resolutions[i+1]:2d}: u={ru:.2f}, w={rw:.2f} (Expected ~2.00)")

    print("  => Semi-Lagrangian Advection Rates (q):")
    for i in range(len(resolutions) - 1):
        rq = np.log2(results[i][5] / results[i+1][5])
        print(f"     {resolutions[i]:2d}->{resolutions[i+1]:2d}: q={rq:.2f} (Expected ~2.00)")


# =============================================================================
# UNIT 8: FLUX-FORM SPLIT-EXPLICIT ADVECTION
# =============================================================================
def test_unit_8_ffsl_advection(resolutions=[16, 32, 64]):
    print("\n=======================================================")
    print(" UNIT 8: FLUX-FORM SPLIT-EXPLICIT ADVECTION CONVERGENCE")
    print("=======================================================")
    lx, ly, lz = 10000.0, 10000.0, 10000.0
    results = []

    for N in resolutions:
        dx, dy, dz = lx / N, ly / N, lz / N
        dt = (dx / 125.0) * 5.0
        grid = RegionalGrid3D(N, N, N, dx, dy, dz, lat_center=0.0, lon_center=0.0)
        advector = FluxFormAdvector(grid, dt)

        state = {
            'u': jnp.ones((N + 1, N, N)) * 10.0,
            'v': jnp.zeros((N, N + 1, N)),
            'eta_dot': jnp.zeros((N, N, N + 1))
        }
        bg_precomputed = {'dz_m_full': jnp.ones((N, N, N)) * dz}

        X, Y, Z = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_m, indexing='ij')
        rho_initial = jnp.sin(2.0 * jnp.pi * X / lx) + 2.0
        rho_next = advector.advect_3d_split(rho_initial, state, bg_precomputed)

        rho_exact = jnp.sin(2.0 * jnp.pi * (X - 10.0 * dt) / lx) + 2.0
        margin = N // 4
        err = float(jnp.sqrt(jnp.mean((rho_next[margin:-margin, margin:-margin, margin:-margin] -
                                       rho_exact[margin:-margin, margin:-margin, margin:-margin])**2)))
        results.append((dx, err))
        print(f"N = {N:2d} | dt = {dt:5.2f}s | L2 Error: {err:.2e}")

    print("  => Rates (N->2N):")
    for i in range(len(resolutions) - 1):
        rate = np.log2(results[i][1] / results[i+1][1])
        print(f"     {resolutions[i]:2d}->{resolutions[i+1]:2d}: Rate = {rate:.2f} (Expected ~2.00)")


if __name__ == "__main__":
    test_unit_1_interpolation([16, 32, 64])
    test_unit_2_operators([16, 32, 64])
    test_unit_3_trajectories([400.0, 200.0, 100.0, 50.0])
    test_unit_4_gmres_inversion([8, 16, 32, 64])
    test_units_5_6_7_component_phases([16, 32, 64])
    test_unit_8_ffsl_advection([16, 32, 64])
