import os
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from suetes.regional3d.geometry import RegionalGrid3D, ObliqueStereographic
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.steppers import build_dynamical_core

# Physical Constants
R_earth = 6371229.0
Omega = 7.2921e-5
g = 9.81
cp = 1004.0
cvd = 717.0
Rd = 287.0
p0 = 100000.0

Lx, Ly, Lz = 2000000.0, 2000000.0, 10000.0
theta_0 = 300.0
N_bv = 0.01

def u_func(x, y, z):
    return 1.0 * jnp.sin(2.0 * jnp.pi * x / Lx)**2 * jnp.sin(2.0 * jnp.pi * y / Ly)**2 * jnp.sin(jnp.pi * z / Lz)**2

def v_func(x, y, z):
    return 1.0 * jnp.sin(2.0 * jnp.pi * x / Lx)**2 * jnp.sin(2.0 * jnp.pi * y / Ly)**2 * jnp.sin(jnp.pi * z / Lz)**2

def w_func(x, y, z):
    return 0.1 * jnp.sin(2.0 * jnp.pi * x / Lx)**2 * jnp.sin(2.0 * jnp.pi * y / Ly)**2 * jnp.sin(jnp.pi * z / Lz)**2

def pip_func(x, y, z):
    return 0.01 * jnp.sin(2.0 * jnp.pi * x / Lx)**2 * jnp.sin(2.0 * jnp.pi * y / Ly)**2 * jnp.sin(jnp.pi * z / Lz)**2

def thvp_func(x, y, z):
    return 5.0 * jnp.sin(2.0 * jnp.pi * x / Lx)**2 * jnp.sin(2.0 * jnp.pi * y / Ly)**2 * jnp.sin(jnp.pi * z / Lz)**2

def th_bg_func(z):
    return theta_0 * jnp.exp((N_bv**2 / g) * z)

def pi_bg_func(z):
    return 1.0 + (g**2 / (cp * theta_0 * N_bv**2)) * (jnp.exp(-N_bv**2 * z / g) - 1.0)

def rho_bg_func(z):
    pi_val = pi_bg_func(z)
    th_v_val = th_bg_func(z)
    return p0 / (Rd * th_v_val) * (pi_val ** (cvd / Rd))

@jax.jit
def analytic_tend_u(x, y, z, f_val):
    m = 1.0 + (x**2 + y**2) / (4.0 * R_earth**2)
    dm_dx = x / (2.0 * R_earth**2)
    dm_dy = y / (2.0 * R_earth**2)
    u = u_func(x, y, z)
    v = v_func(x, y, z)
    w = w_func(x, y, z)
    th_v = th_bg_func(z) + thvp_func(x, y, z)
    
    du_dx = jax.grad(lambda x_val: u_func(x_val, y, z))(x)
    du_dy = jax.grad(lambda y_val: u_func(x, y_val, z))(y)
    du_dz = jax.grad(lambda z_val: u_func(x, y, z_val))(z)
    dpip_dx = jax.grad(lambda x_val: pip_func(x_val, y, z))(x)
    
    advect = -m * (u * du_dx + v * du_dy) - w * du_dz
    press_grad = -cp * th_v * m * dpip_dx
    coriolis_metric = (f_val + v * dm_dx - u * dm_dy) * v
    return advect + press_grad + coriolis_metric

@jax.jit
def analytic_tend_v(x, y, z, f_val):
    m = 1.0 + (x**2 + y**2) / (4.0 * R_earth**2)
    dm_dx = x / (2.0 * R_earth**2)
    dm_dy = y / (2.0 * R_earth**2)
    u = u_func(x, y, z)
    v = v_func(x, y, z)
    w = w_func(x, y, z)
    th_v = th_bg_func(z) + thvp_func(x, y, z)
    
    dv_dx = jax.grad(lambda x_val: v_func(x_val, y, z))(x)
    dv_dy = jax.grad(lambda y_val: v_func(x, y_val, z))(y)
    dv_dz = jax.grad(lambda z_val: v_func(x, y, z_val))(z)
    dpip_dy = jax.grad(lambda y_val: pip_func(x, y_val, z))(y)
    
    advect = -m * (u * dv_dx + v * dv_dy) - w * dv_dz
    press_grad = -cp * th_v * m * dpip_dy
    coriolis_metric = -(f_val + v * dm_dx - u * dm_dy) * u
    return advect + press_grad + coriolis_metric

@jax.jit
def analytic_tend_w(x, y, z):
    m = 1.0 + (x**2 + y**2) / (4.0 * R_earth**2)
    u = u_func(x, y, z)
    v = v_func(x, y, z)
    w = w_func(x, y, z)
    th_bg_val = th_bg_func(z)
    thvp_val = thvp_func(x, y, z)
    th_v = th_bg_val + thvp_val
    
    dw_dx = jax.grad(lambda x_val: w_func(x_val, y, z))(x)
    dw_dy = jax.grad(lambda y_val: w_func(x, y_val, z))(y)
    dw_dz = jax.grad(lambda z_val: w_func(x, y, z_val))(z)
    dpip_dz = jax.grad(lambda z_val: pip_func(x, y, z_val))(z)
    
    advect = -m * (u * dw_dx + v * dw_dy) - w * dw_dz
    press_grad = -cp * th_v * dpip_dz
    buoyancy = g * (thvp_val / th_bg_val)
    return advect + press_grad + buoyancy

@jax.jit
def analytic_tend_pi(x, y, z):
    m = 1.0 + (x**2 + y**2) / (4.0 * R_earth**2)
    th_v_bg = th_bg_func(z)
    pi_bg = pi_bg_func(z)
    rho_bg = rho_bg_func(z)
    C_pi = (Rd / cvd) * (pi_bg / (rho_bg * th_v_bg))
    
    def flux_x_fn(x_val):
        return u_func(x_val, y, z) * rho_bg_func(z) * th_bg_func(z) / (1.0 + (x_val**2 + y**2) / (4.0 * R_earth**2))
    def flux_y_fn(y_val):
        return v_func(x, y_val, z) * rho_bg_func(z) * th_bg_func(z) / (1.0 + (x**2 + y_val**2) / (4.0 * R_earth**2))
    def flux_z_fn(z_val):
        return w_func(x, y, z_val) * rho_bg_func(z_val) * th_bg_func(z_val)
        
    dflux_x_dx = jax.grad(flux_x_fn)(x)
    dflux_y_dy = jax.grad(flux_y_fn)(y)
    dflux_z_dz = jax.grad(flux_z_fn)(z)
    return -C_pi * (m**2 * (dflux_x_dx + dflux_y_dy) + dflux_z_dz)

@jax.jit
def analytic_tend_th(x, y, z):
    m = 1.0 + (x**2 + y**2) / (4.0 * R_earth**2)
    u = u_func(x, y, z)
    v = v_func(x, y, z)
    w = w_func(x, y, z)
    
    dthvp_dx = jax.grad(lambda x_val: thvp_func(x_val, y, z))(x)
    dthvp_dy = jax.grad(lambda y_val: thvp_func(x, y_val, z))(y)
    dthvp_dz = jax.grad(lambda z_val: thvp_func(x, y, z_val))(z)
    dth_bg_dz = jax.grad(th_bg_func)(z)
    
    advect = -m * (u * dthvp_dx + v * dthvp_dy) - w * dthvp_dz
    background_transport = -w * dth_bg_dz
    return advect + background_transport

vmap_u = jax.vmap(jax.vmap(jax.vmap(analytic_tend_u, in_axes=(0, 0, 0, None)), in_axes=(0, 0, 0, 0)), in_axes=(0, 0, 0, 0))
vmap_v = jax.vmap(jax.vmap(jax.vmap(analytic_tend_v, in_axes=(0, 0, 0, None)), in_axes=(0, 0, 0, 0)), in_axes=(0, 0, 0, 0))
vmap_w = jax.vmap(jax.vmap(jax.vmap(analytic_tend_w, in_axes=(0, 0, 0)), in_axes=(0, 0, 0)), in_axes=(0, 0, 0))
vmap_pi = jax.vmap(jax.vmap(jax.vmap(analytic_tend_pi, in_axes=(0, 0, 0)), in_axes=(0, 0, 0)), in_axes=(0, 0, 0))
vmap_th = jax.vmap(jax.vmap(jax.vmap(analytic_tend_th, in_axes=(0, 0, 0)), in_axes=(0, 0, 0)), in_axes=(0, 0, 0))

def run_debug_at_res(n):
    nx, ny, nz = n, n, n
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz
    grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_center=60.0, lon_center=30.0)
    op = CGridOperator3D(grid)
    constants_dict = {'g': g, 'cp': cp, 'cvd': cvd, 'Rd': Rd, 'p0': p0}
    
    dt = 320.0 * (16.0 / n)
    core_kwargs = {
        "dt": dt, "nu_div_factor": 0.0, "nu_h_factor": 0.0, "alpha": 0.5,
        "damp_height": 20000.0,
        "solver_tol": 1e-14, "solver_maxiter": 100, "solver_restart": 100
    }
    
    X_u, Y_u, Z_u = jnp.meshgrid(grid.x_c, grid.y_m, grid.z_m, indexing='ij')
    X_v, Y_v, Z_v = jnp.meshgrid(grid.x_m, grid.y_c, grid.z_m, indexing='ij')
    X_w, Y_w, Z_w = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_c, indexing='ij')
    X_m, Y_m, Z_m = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_m, indexing='ij')
    
    theta_bg = theta_0 * jnp.exp((N_bv**2 / g) * Z_m)
    delta_z = Z_m[:, :, 1:] - Z_m[:, :, :-1]
    th_v_w_bg = 0.5 * (theta_bg[:, :, 1:] + theta_bg[:, :, :-1])
    delta_pi = -(g * delta_z) / (cp * th_v_w_bg)
    pi_cumsum_rev = jnp.cumsum(delta_pi[..., ::-1], axis=-1)[..., ::-1]
    z_max = jnp.max(grid.Z_m)
    pi_anchor_top = 1.0 + (g**2 / (cp * theta_0 * N_bv**2)) * (jnp.exp(-N_bv**2 * z_max / g) - 1.0)
    pi_top_3d = jnp.full((grid.nx, grid.ny, 1), pi_anchor_top)
    pi_bg_discrete = jnp.concatenate([pi_top_3d - pi_cumsum_rev, pi_top_3d], axis=-1)
    
    initial_state = {
        'u': u_func(X_u, Y_u, Z_u),
        'v': v_func(X_v, Y_v, Z_v),
        'w': w_func(X_w, Y_w, Z_w),
        'pi': pi_bg_discrete + pip_func(X_m, Y_m, Z_m),
        'th_v': theta_bg + thvp_func(X_m, Y_m, Z_m),
    }
    initial_state['eta_dot'] = initial_state['w'] / grid.dz_w_full
    initial_state['rho'] = p0 / (Rd * initial_state['th_v']) * (initial_state['pi'] ** (cvd / Rd))
    
    buffer = 450000.0 
    def get_eval_mask(X, Y):
        dist_x = Lx / 2.0 - jnp.abs(X)
        dist_y = Ly / 2.0 - jnp.abs(Y)
        return (dist_x >= buffer) & (dist_y >= buffer)

    eval_u = get_eval_mask(X_u, Y_u)
    eval_v = get_eval_mask(X_v, Y_v)
    eval_w = get_eval_mask(X_w, Y_w)
    eval_m = get_eval_mask(X_m, Y_m)
    
    sponge_width = 300000.0 
    def get_sponge_mask(X, Y):
        dist_x = Lx / 2.0 - jnp.abs(X)
        dist_y = Ly / 2.0 - jnp.abs(Y)
        alpha_x = jnp.clip(dist_x / sponge_width, 0.0, 1.0)
        alpha_y = jnp.clip(dist_y / sponge_width, 0.0, 1.0)
        smooth_x = alpha_x**2 * (3.0 - 2.0 * alpha_x)
        smooth_y = alpha_y**2 * (3.0 - 2.0 * alpha_y)
        return smooth_x * smooth_y

    sponge_u = get_sponge_mask(X_u, Y_u)
    sponge_v = get_sponge_mask(X_v, Y_v)
    sponge_w = get_sponge_mask(X_w, Y_w)
    sponge_m = get_sponge_mask(X_m, Y_m)
    
    kwargs = core_kwargs.copy()
    kwargs['alpha'] = 0.50
    kwargs['solver_tol'] = 1e-12
    kwargs['solver_maxiter'] = 100
    kwargs['solver_restart'] = 50
    stepper, dt = build_dynamical_core(
        core_type="sisl", grid=grid, operators=op, constants=constants_dict,
        initial_state=initial_state, N_bv=N_bv, **kwargs
    )
    if r == 8:
        print(f"[DEBUG build] stepper.alpha = {stepper.alpha}, stepper.implicit_solver.alpha = {stepper.implicit_solver.alpha}")
    
    ana_u = vmap_u(X_u, Y_u, Z_u, grid.f_u)
    ana_v = vmap_v(X_v, Y_v, Z_v, grid.f_v)
    ana_w = vmap_w(X_w, Y_w, Z_w)
    ana_pi = vmap_pi(X_m, Y_m, Z_m)
    ana_th = vmap_th(X_m, Y_m, Z_m)
    
    alpha_val = stepper.alpha
    ana_u_exp = ana_u
    ana_v_exp = ana_v
    ana_w_exp = ana_w
    ana_th_exp = (1.0 - alpha_val) * ana_th
    
    original_get_tendencies = stepper.physics.get_tendencies
    
    def get_tendencies_with_forcing(state_prime, bg, is_explicit=False, ml_params=None):
        tends = original_get_tendencies(state_prime, bg, is_explicit, ml_params)
        if is_explicit:
            tends['th_v'] -= ana_th
        return tends
        
    stepper.physics.get_tendencies = get_tendencies_with_forcing
    
    original_solve = stepper.implicit_solver.solve
    def solve_with_mms_forcing(rhs_prime, bg_precomputed, x0=None):
        if x0 is not None:
            L_exact = stepper.physics.linear_operator(x0, bg_precomputed, dt, alpha=alpha_val)
            for k in ['u', 'v', 'w', 'pi', 'eta_dot']:
                if k in rhs_prime and k in L_exact:
                    rhs_prime[k] = rhs_prime[k] + (L_exact[k] - rhs_prime[k])
        return original_solve(rhs_prime, bg_precomputed, x0)
    stepper.implicit_solver.solve = solve_with_mms_forcing
    
    def bc_fn(s, f):
        for k in ['u', 'v', 'w']:
            s[k] = s[k].at[0, :, :].set(initial_state[k][0, :, :]).at[-1, :, :].set(initial_state[k][-1, :, :])
            s[k] = s[k].at[:, 0, :].set(initial_state[k][:, 0, :]).at[:, -1, :].set(initial_state[k][:, -1, :])
        s['w'] = s['w'].at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
        s['eta_dot'] = s['eta_dot'].at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
        return s

    def compute_safe_err(diff, mask):
        return jnp.sqrt(jnp.sum(diff**2 * mask) / jnp.sum(mask))

    # --- STEP INSPECTION: Run 1 step and check components ---
    s = initial_state
    
    # Let's manually trace inside stepper.step to inspect components right before and after implicit solve
    coords_u = stepper.advector.compute_departure_indices(s, loc='u')
    coords_v = stepper.advector.compute_departure_indices(s, loc='v')
    coords_w = stepper.advector.compute_departure_indices(s, loc='w')
    coords_m = stepper.advector.compute_departure_indices(s, loc='m')
    
    bg_state_ref = {
        'rho': stepper.physics.c['p0'] / (stepper.physics.c['Rd'] * stepper.physics.theta_bg) * \
               (stepper.physics.pi_bg ** (stepper.physics.c['cvd'] / stepper.physics.c['Rd'])),
        'pi': stepper.physics.pi_bg,
        'th_v': stepper.physics.theta_bg
    }
    bg_precomputed = stepper.physics.precompute_bg(bg_state_ref)
    th_v_prime_n = s['th_v'] - stepper.physics.theta_bg
    state_prime_n = {
        'u': s['u'], 'v': s['v'], 'w': s['w'], 'th_v': s['th_v'],
        'pi': s['pi'] - stepper.physics.pi_bg, 'eta_dot': s['eta_dot'],
        'th_v_prime_u': stepper.physics.op.avg(th_v_prime_n, axis=0, from_loc='m', to_loc='u'),
        'th_v_prime_v': stepper.physics.op.avg(th_v_prime_n, axis=1, from_loc='m', to_loc='v'),
        'th_v_prime_w': stepper.physics.op.avg(th_v_prime_n, axis=2, from_loc='m', to_loc='w')
    }
    tends_n = stepper.physics.get_tendencies(state_prime_n, bg_precomputed, True, None)
    
    u_in = s['u'] + dt * ((1.0 - alpha_val) * tends_n['u'] + alpha_val * tends_n.get('phys_diff_u', 0.0))
    v_in = s['v'] + dt * ((1.0 - alpha_val) * tends_n['v'] + alpha_val * tends_n.get('phys_diff_v', 0.0))
    w_in = s['w'] + dt * ((1.0 - alpha_val) * tends_n['w'] + alpha_val * tends_n.get('phys_diff_w', 0.0))
    pi_prime_in = state_prime_n['pi'] + (1.0 - alpha_val) * dt * tends_n['pi']
    th_v_prime_in = th_v_prime_n + dt * tends_n['th_v']
    
    # Check what happens to pi_prime_in after 1 explicit tendency step
    err_pi_prime_in = compute_safe_err(pi_prime_in - state_prime_n['pi'], eval_m)
    
    # Advect
    rhs_u = stepper.advector.advect_cubic(u_in, coords_u, stepper.use_limiter)
    rhs_v = stepper.advector.advect_cubic(v_in, coords_v, stepper.use_limiter)
    rhs_w = stepper.advector.advect_cubic(w_in, coords_w, stepper.use_limiter)
    rhs_pi_prime = pi_prime_in
    th_v_prime_next = stepper.advector.advect_cubic(th_v_prime_in, coords_m, stepper.use_limiter)
    
    th_v_next = th_v_prime_next + stepper.physics.theta_bg
    th_v_next_mms = th_v_next - alpha_val * dt * ana_th
    
    err_th_v_next_mms = compute_safe_err(th_v_next_mms - initial_state['th_v'], eval_m)
    
    # Buoyancy & Kinematic w
    th_v_prime_next_for_buoy = th_v_next - stepper.physics.theta_bg
    th_v_prime_w_next = stepper.physics.op.avg(th_v_prime_next_for_buoy, axis=2, from_loc='m', to_loc='w')
    rhs_w += alpha_val * dt * (stepper.physics.c['g'] * (th_v_prime_w_next / bg_precomputed['th_v_w']))
    
    u_m_rhs = stepper.physics.op.avg(rhs_u, axis=0, from_loc='u', to_loc='m')
    u_w_rhs = stepper.physics.op.avg(u_m_rhs, axis=2, from_loc='m', to_loc='w')
    v_m_rhs = stepper.physics.op.avg(rhs_v, axis=1, from_loc='v', to_loc='m')
    v_w_rhs = stepper.physics.op.avg(v_m_rhs, axis=2, from_loc='m', to_loc='w')
    m_w = jnp.expand_dims(stepper.physics.grid.m_factors['w'], axis=-1)
    rhs_kinematic_bottom = m_w[:, :, 0] * (
        u_w_rhs[:, :, 0] * stepper.physics.grid.z_xi_w[:, :, 0] + 
        v_w_rhs[:, :, 0] * stepper.physics.grid.z_eta_w[:, :, 0]
    )
    rhs_w = rhs_w.at[:, :, 0].set(rhs_kinematic_bottom)
    rhs_w = rhs_w.at[:, :, -1].set(0.0)
    
    # Residual n for eta_dot
    u_m = stepper.physics.op.avg(s['u'], axis=0, from_loc='u', to_loc='m')
    u_w = stepper.physics.op.avg(u_m, axis=2, from_loc='m', to_loc='w')
    v_m = stepper.physics.op.avg(s['v'], axis=1, from_loc='v', to_loc='m')
    v_w = stepper.physics.op.avg(v_m, axis=2, from_loc='m', to_loc='w')
    residual_n = (
        bg_precomputed['dz_w_full'] * s['eta_dot'] + 
        u_w * stepper.physics.grid.z_xi_w + 
        v_w * stepper.physics.grid.z_eta_w - 
        s['w']
    )
    R_eta_dot = -(1.0 - alpha_val) * stepper.advector.advect_cubic(residual_n, coords_w, stepper.use_limiter)
    R_eta_dot = R_eta_dot.at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
    
    # Implicit solve
    rhs_prime = {'u': rhs_u, 'v': rhs_v, 'w': rhs_w, 'pi': rhs_pi_prime, 'eta_dot': R_eta_dot}
    
    # Check exact residual of linear system A_fn(state_prime_n) - solve_with_mms_forcing(rhs_prime)
    # First, apply the MMS forcing to rhs_prime exactly as solve() does:
    rhs_prime_forced = {
        'u': rhs_u - alpha_val * dt * ana_u,
        'v': rhs_v - alpha_val * dt * ana_v,
        'w': rhs_w - alpha_val * dt * ana_w,
        'pi': rhs_pi_prime - alpha_val * dt * ana_pi,
        'eta_dot': R_eta_dot
    }
    # Now evaluate L(state_prime_n) using linear_operator
    L_exact = stepper.physics.linear_operator(state_prime_n, bg_precomputed, dt, alpha=alpha_val)
    err_L_u = compute_safe_err(L_exact['u'] - rhs_prime_forced['u'], eval_u)
    err_L_v = compute_safe_err(L_exact['v'] - rhs_prime_forced['v'], eval_v)
    err_L_w = compute_safe_err(L_exact['w'] - rhs_prime_forced['w'], eval_w)
    err_L_pi = compute_safe_err(L_exact['pi'] - rhs_prime_forced['pi'], eval_m)
    err_L_eta = compute_safe_err(L_exact['eta_dot'] - rhs_prime_forced['eta_dot'], eval_w)
    
    if r == 16 or r == 64:
        diff_w = jnp.abs(L_exact['w'] - rhs_prime_forced['w']) * eval_w
        print(f"\n[DEBUG L_w at Res {r}] Max by k: " + ", ".join([f"k={k}: {float(jnp.max(diff_w[:, :, k])):.2e}" for k in range(min(5, nz))]) + f" ... k={nz-1}: {float(jnp.max(diff_w[:, :, -1])):.2e}")
        diff_eta = jnp.abs(L_exact['eta_dot'] - rhs_prime_forced['eta_dot']) * eval_w
        print(f"[DEBUG L_eta at Res {r}] Max by k: " + ", ".join([f"k={k}: {float(jnp.max(diff_eta[:, :, k])):.2e}" for k in range(min(5, nz))]) + f" ... k={nz-1}: {float(jnp.max(diff_eta[:, :, -1])):.2e}")
        
        # Also check x_sol - x_exact by k right after solve
    state_prime_next = stepper.implicit_solver.solve(rhs_prime, bg_precomputed, x0=state_prime_n)
    if r == 16 or r == 64:
        diff_sol_w = jnp.abs(state_prime_next['w'] - state_prime_n['w']) * eval_w
        print(f"[DEBUG sol_w - exact at Res {r}] Max by k: " + ", ".join([f"k={k}: {float(jnp.max(diff_sol_w[:, :, k])):.2e}" for k in range(min(5, nz))]) + f" ... k={nz-1}: {float(jnp.max(diff_sol_w[:, :, -1])):.2e}")
        diff_sol_pi = jnp.abs(state_prime_next['pi'] - state_prime_n['pi']) * eval_m
        print(f"[DEBUG sol_pi - exact at Res {r}] Max by k: " + ", ".join([f"k={k}: {float(jnp.max(diff_sol_pi[:, :, k])):.2e}" for k in range(min(5, nz))]) + f" ... k={nz-1}: {float(jnp.max(diff_sol_pi[:, :, -1])):.2e}\n")
    
    # Check L(state_prime_next) - rhs_prime_forced
    L_sol = stepper.physics.linear_operator(state_prime_next, bg_precomputed, dt, alpha=alpha_val)
    res_sol_u = compute_safe_err(L_sol['u'] - rhs_prime_forced['u'], eval_u)
    res_sol_w = compute_safe_err(L_sol['w'] - rhs_prime_forced['w'], eval_w)
    res_sol_pi = compute_safe_err(L_sol['pi'] - rhs_prime_forced['pi'], eval_m)
    
    err_sol_u = compute_safe_err(state_prime_next['u'] - state_prime_n['u'], eval_u)
    err_sol_v = compute_safe_err(state_prime_next['v'] - state_prime_n['v'], eval_v)
    err_sol_w = compute_safe_err(state_prime_next['w'] - state_prime_n['w'], eval_w)
    err_sol_pi = compute_safe_err(state_prime_next['pi'] - state_prime_n['pi'], eval_m)
    err_sol_eta = compute_safe_err(state_prime_next['eta_dot'] - state_prime_n['eta_dot'], eval_w)
    
    pi_next = state_prime_next['pi'] + stepper.physics.pi_bg
    err_pi_next = compute_safe_err(pi_next - initial_state['pi'], eval_m)
    err_w_next = compute_safe_err(state_prime_next['w'] - initial_state['w'], eval_w)
    
    # Run full 1 step via stepper.step + mms adjustments
    s_step1 = stepper.step(s, 0.0, None, bc_fn)
    s_step1['rho'] = p0 / (Rd * s_step1['th_v']) * (s_step1['pi'] ** (cvd / Rd))
    
    err_step1_u = compute_safe_err(s_step1['u'] - initial_state['u'], eval_u)
    err_step1_v = compute_safe_err(s_step1['v'] - initial_state['v'], eval_v)
    err_step1_w = compute_safe_err(s_step1['w'] - initial_state['w'], eval_w)
    err_step1_pi = compute_safe_err(s_step1['pi'] - initial_state['pi'], eval_m)
    err_step1_th = compute_safe_err(s_step1['th_v'] - initial_state['th_v'], eval_m)
    
    # Run full 2 steps
    s_step2 = stepper.step(s_step1, dt, None, bc_fn)
    s_step2['rho'] = p0 / (Rd * s_step2['th_v']) * (s_step2['pi'] ** (cvd / Rd))
    
    err_step2_u = compute_safe_err(s_step2['u'] - initial_state['u'], eval_u)
    err_step2_v = compute_safe_err(s_step2['v'] - initial_state['v'], eval_v)
    err_step2_w = compute_safe_err(s_step2['w'] - initial_state['w'], eval_w)
    err_step2_pi = compute_safe_err(s_step2['pi'] - initial_state['pi'], eval_m)
    err_step2_th = compute_safe_err(s_step2['th_v'] - initial_state['th_v'], eval_m)
    
    return (float(dx), float(err_th_v_next_mms), float(err_pi_next), float(err_w_next),
            float(err_step1_u), float(err_step1_v), float(err_step1_w), float(err_step1_pi), float(err_step1_th),
            float(err_step2_u), float(err_step2_v), float(err_step2_w), float(err_step2_pi), float(err_step2_th),
            float(err_L_u), float(err_L_v), float(err_L_w), float(err_L_pi), float(err_L_eta),
            float(res_sol_u), float(res_sol_w), float(res_sol_pi),
            float(err_sol_u), float(err_sol_v), float(err_sol_w), float(err_sol_pi), float(err_sol_eta))

if __name__ == "__main__":
    resolutions = [8, 16, 32, 64]
    print("=== SISL Component Inspection across Resolutions ===")
    results = []
    for r in resolutions:
        res = run_debug_at_res(r)
        results.append(res)
        print(f"Res {r:3d} | dx: {res[0]/1000.:.1f}km")
        print(f"  [L(x_exact)-RHS]    u: {res[14]:.2e} | w: {res[16]:.2e} | pi: {res[17]:.2e} | eta: {res[18]:.2e}")
        print(f"  [L(x_sol)  -RHS]    u: {res[19]:.2e} | w: {res[20]:.2e} | pi: {res[21]:.2e}")
        print(f"  [x_sol - x_exact]   u: {res[22]:.2e} | w: {res[24]:.2e} | pi: {res[25]:.2e} | eta: {res[26]:.2e}")
        print(f"  [1 Step Final]      u: {res[4]:.2e} | w: {res[6]:.2e} | pi: {res[7]:.2e} | th: {res[8]:.2e}")
        print(f"  [2 Steps Final]     u: {res[9]:.2e} | w: {res[11]:.2e} | pi: {res[12]:.2e} | th: {res[13]:.2e}")
        
    print("\n--- 1 Step Convergence Rates ---")
    for i in range(len(resolutions) - 1):
        dx1 = results[i][0]
        dx2 = results[i+1][0]
        factor = dx1 / dx2
        print(f"Res {resolutions[i]} -> {resolutions[i+1]}:")
        print(f"  [L(x_exact)-RHS]    Rate L_u: {np.log(results[i][14]/results[i+1][14])/np.log(factor):.2f} | Rate L_w: {np.log(results[i][16]/results[i+1][16])/np.log(factor):.2f} | Rate L_pi: {np.log(results[i][17]/results[i+1][17])/np.log(factor):.2f}")
        print(f"  [L(x_sol)  -RHS]    Rate sol_u: {np.log(results[i][19]/results[i+1][19])/np.log(factor):.2f} | Rate sol_w: {np.log(results[i][20]/results[i+1][20])/np.log(factor):.2f} | Rate sol_pi: {np.log(results[i][21]/results[i+1][21])/np.log(factor):.2f}")
        print(f"  [x_sol - x_exact]   Rate err_u: {np.log(results[i][22]/results[i+1][22])/np.log(factor):.2f} | Rate err_w: {np.log(results[i][24]/results[i+1][24])/np.log(factor):.2f} | Rate err_pi: {np.log(results[i][25]/results[i+1][25])/np.log(factor):.2f} | Rate err_eta: {np.log(results[i][26]/results[i+1][26])/np.log(factor):.2f}")
        print(f"  [2 Steps Final]     Rate u: {np.log(results[i][9]/results[i+1][9])/np.log(factor):.2f} | Rate w: {np.log(results[i][11]/results[i+1][11])/np.log(factor):.2f} | Rate pi: {np.log(results[i][12]/results[i+1][12])/np.log(factor):.2f} | Rate th: {np.log(results[i][13]/results[i+1][13])/np.log(factor):.2f}")
