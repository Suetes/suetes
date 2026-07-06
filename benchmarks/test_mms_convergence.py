import os
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

# Enable X64 for high-precision convergence testing
jax.config.update("jax_enable_x64", True)

from suetes.regional3d.geometry import RegionalGrid3D, ObliqueStereographic
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.steppers import build_dynamical_core
from suetes.shared.driver import Simulation

# Physical Constants
R_earth = 6371229.0
Omega = 7.2921e-5
g = 9.81
cp = 1004.0
cvd = 717.0
Rd = 287.0
p0 = 100000.0

# Domain dimensions
Lx, Ly, Lz = 2000000.0, 2000000.0, 10000.0
theta_0 = 300.0
N_bv = 0.01

proj = ObliqueStereographic(lat_center=60.0, lon_center=30.0, R_earth=R_earth)

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

# =============================================================================
# Universal Eulerian Analytical Tendencies
# =============================================================================
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

# =============================================================================
# Implicit Linear Solver Parts (For split-forcing in SISL)
# =============================================================================
@jax.jit
def analytic_tend_u_imp(x, y, z, f_val):
    m = 1.0 + (x**2 + y**2) / (4.0 * R_earth**2)
    th_bg_val = th_bg_func(z)
    dpip_dx = jax.grad(lambda x_val: pip_func(x_val, y, z))(x)
    v = v_func(x, y, z)
    return -cp * th_bg_val * m * dpip_dx + f_val * v

@jax.jit
def analytic_tend_v_imp(x, y, z, f_val):
    m = 1.0 + (x**2 + y**2) / (4.0 * R_earth**2)
    th_bg_val = th_bg_func(z)
    dpip_dy = jax.grad(lambda y_val: pip_func(x, y_val, z))(y)
    u = u_func(x, y, z)
    return -cp * th_bg_val * m * dpip_dy - f_val * u

@jax.jit
def analytic_tend_w_imp(x, y, z):
    th_bg_val = th_bg_func(z)
    dpip_dz = jax.grad(lambda z_val: pip_func(x, y, z_val))(z)
    thvp_val = thvp_func(x, y, z)
    buoyancy = g * (thvp_val / th_bg_val)
    return -cp * th_bg_val * dpip_dz + buoyancy

vmap_u = jax.vmap(jax.vmap(jax.vmap(analytic_tend_u, in_axes=(0, 0, 0, None)), in_axes=(0, 0, 0, 0)), in_axes=(0, 0, 0, 0))
vmap_v = jax.vmap(jax.vmap(jax.vmap(analytic_tend_v, in_axes=(0, 0, 0, None)), in_axes=(0, 0, 0, 0)), in_axes=(0, 0, 0, 0))
vmap_w = jax.vmap(jax.vmap(jax.vmap(analytic_tend_w, in_axes=(0, 0, 0)), in_axes=(0, 0, 0)), in_axes=(0, 0, 0))
vmap_pi = jax.vmap(jax.vmap(jax.vmap(analytic_tend_pi, in_axes=(0, 0, 0)), in_axes=(0, 0, 0)), in_axes=(0, 0, 0))
vmap_th = jax.vmap(jax.vmap(jax.vmap(analytic_tend_th, in_axes=(0, 0, 0)), in_axes=(0, 0, 0)), in_axes=(0, 0, 0))

vmap_u_imp = jax.vmap(jax.vmap(jax.vmap(analytic_tend_u_imp, in_axes=(0, 0, 0, None)), in_axes=(0, 0, 0, 0)), in_axes=(0, 0, 0, 0))
vmap_v_imp = jax.vmap(jax.vmap(jax.vmap(analytic_tend_v_imp, in_axes=(0, 0, 0, None)), in_axes=(0, 0, 0, 0)), in_axes=(0, 0, 0, 0))
vmap_w_imp = jax.vmap(jax.vmap(jax.vmap(analytic_tend_w_imp, in_axes=(0, 0, 0)), in_axes=(0, 0, 0)), in_axes=(0, 0, 0))

def run_mms_at_resolution(core_type, n, num_steps=5):
    nx, ny, nz = n, n, n
    dx = Lx / nx
    dy = Ly / ny
    dz = Lz / nz
    
    grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_center=60.0, lon_center=30.0)
    op = CGridOperator3D(grid)
    constants_dict = {'g': g, 'cp': cp, 'cvd': cvd, 'Rd': Rd, 'p0': p0}
    t_end_fixed = 640.0 
    dt_ref = t_end_fixed / 2.0  # 2 steps at reference resolution N=16
    dt = dt_ref * (16.0 / n)
    
    if core_type == "sisl":
        core_kwargs = {
            "dt": dt, "nu_div_factor": 0.0, "nu_h_factor": 0.0, "alpha": 0.5,
            "damp_height": 20000.0,
            "solver_tol": 1e-14, "solver_maxiter": 100, "solver_restart": 100
        }
    elif core_type == "split-explicit":
        core_kwargs = {"dt": dt, "ns": 10, "nu_div_factor": 0.0, "nu_h_factor": 0.0, "alpha": 0.55}
    else:
        raise ValueError(f"Unknown core: {core_type}")

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
    
    stepper, dt = build_dynamical_core(
        core_type=core_type, grid=grid, operators=op, constants=constants_dict,
        initial_state=initial_state, N_bv=N_bv, **core_kwargs
    )
    
    ana_u = vmap_u(X_u, Y_u, Z_u, grid.f_u)
    ana_v = vmap_v(X_v, Y_v, Z_v, grid.f_v)
    ana_w = vmap_w(X_w, Y_w, Z_w)
    ana_pi = vmap_pi(X_m, Y_m, Z_m)
    ana_th = vmap_th(X_m, Y_m, Z_m)
    
    ana_u_imp = vmap_u_imp(X_u, Y_u, Z_u, grid.f_u)
    ana_v_imp = vmap_v_imp(X_v, Y_v, Z_v, grid.f_v)
    ana_w_imp = vmap_w_imp(X_w, Y_w, Z_w)
    
    alpha_val = stepper.alpha
    
    # Isolate explicit residuals using the full Eulerian reference frame
    ana_u_exp = (ana_u - alpha_val * ana_u_imp) / (1.0 - alpha_val)
    ana_v_exp = (ana_v - alpha_val * ana_v_imp) / (1.0 - alpha_val)
    ana_w_exp = (ana_w - alpha_val * ana_w_imp) / (1.0 - alpha_val)
    ana_th_exp = ana_th
    
    original_get_tendencies = stepper.physics.get_tendencies
    
    def get_tendencies_with_forcing(state_prime, bg, is_explicit=False, ml_params=None):
        tends = original_get_tendencies(state_prime, bg, is_explicit, ml_params)
        if is_explicit:
            if core_type == "sisl":
                tends['u'] -= ana_u_exp
                tends['v'] -= ana_v_exp
                tends['w'] -= ana_w_exp
                tends['pi'] -= ana_pi
                tends['th_v'] -= ana_th_exp
                
                tends['u'] = tends['u'] * sponge_u + ana_u_exp * (1.0 - sponge_u)
                tends['v'] = tends['v'] * sponge_v + ana_v_exp * (1.0 - sponge_v)
                tends['w'] = tends['w'] * sponge_w + ana_w_exp * (1.0 - sponge_w)
                tends['pi'] = tends['pi'] * sponge_m + ana_pi * (1.0 - sponge_m)
                tends['th_v'] = tends['th_v'] * sponge_m + ana_th_exp * (1.0 - sponge_m)
            else:
                tends['u'] -= ana_u
                tends['v'] -= ana_v
                tends['w'] -= ana_w
                tends['pi'] -= ana_pi
                tends['th_v'] -= ana_th
        return tends
        
    stepper.physics.get_tendencies = get_tendencies_with_forcing
    
    if core_type == "sisl":
        original_solve = stepper.implicit_solver.solve
        def solve_with_mms_forcing(rhs_prime, bg_precomputed, x0=None):
            rhs_prime['u'] -= alpha_val * dt * ana_u_imp
            rhs_prime['v'] -= alpha_val * dt * ana_v_imp
            rhs_prime['w'] -= alpha_val * dt * ana_w_imp
            rhs_prime['pi'] -= alpha_val * dt * ana_pi
            return original_solve(rhs_prime, bg_precomputed, x0)
        stepper.implicit_solver.solve = solve_with_mms_forcing
        
    num_steps = int(t_end_fixed / dt)
    actual_t_end = num_steps * dt
    
    def bc_fn(s, f):
        for k in ['u', 'v', 'w']:
            s[k] = s[k].at[0, :, :].set(initial_state[k][0, :, :]).at[-1, :, :].set(initial_state[k][-1, :, :])
            s[k] = s[k].at[:, 0, :].set(initial_state[k][:, 0, :]).at[:, -1, :].set(initial_state[k][:, -1, :])
        s['w'] = s['w'].at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
        s['eta_dot'] = s['eta_dot'].at[:, :, 0].set(0.0).at[:, :, -1].set(0.0)
        return s

    def step_fn_mms(s, i):
        s_next = stepper.step(s, i*dt, None, bc_fn)
        s_next['u'] = s_next['u'] * sponge_u + initial_state['u'] * (1.0 - sponge_u)
        s_next['v'] = s_next['v'] * sponge_v + initial_state['v'] * (1.0 - sponge_v)
        s_next['w'] = s_next['w'] * sponge_w + initial_state['w'] * (1.0 - sponge_w)
        s_next['eta_dot'] = s_next['eta_dot'] * sponge_w + initial_state['eta_dot'] * (1.0 - sponge_w)
        s_next['pi'] = s_next['pi'] * sponge_m + initial_state['pi'] * (1.0 - sponge_m)
        s_next['th_v'] = s_next['th_v'] * sponge_m + initial_state['th_v'] * (1.0 - sponge_m)
        s_next['rho'] = p0 / (Rd * s_next['th_v']) * (s_next['pi'] ** (cvd / Rd))
        if 'u_prev' in s_next:
            s_next['u_prev'] = s_next['u_prev'] * sponge_u + initial_state['u'] * (1.0 - sponge_u)
            s_next['v_prev'] = s_next['v_prev'] * sponge_v + initial_state['v'] * (1.0 - sponge_v)
            s_next['w_prev'] = s_next['w_prev'] * sponge_w + initial_state['w'] * (1.0 - sponge_w)
            s_next['eta_dot_prev'] = s_next['eta_dot_prev'] * sponge_w + initial_state['eta_dot'] * (1.0 - sponge_w)
            s_next['tend_th_v_prev'] = s_next['tend_th_v_prev'] * sponge_m
        return s_next, 0.0

    sim = Simulation(step_fn=step_fn_mms, dt=dt)
    final_state = sim.run(initial_state, t_start=0.0, t_end=actual_t_end, chunk_steps=num_steps)

    def compute_safe_err(final, initial, mask):
        sq_err = (final - initial)**2
        return jnp.sqrt(jnp.sum(sq_err * mask) / jnp.sum(mask))

    err_u = float(compute_safe_err(final_state['u'], initial_state['u'], eval_u))
    err_v = float(compute_safe_err(final_state['v'], initial_state['v'], eval_v))
    err_w = float(compute_safe_err(final_state['w'], initial_state['w'], eval_w))
    err_pi = float(compute_safe_err(final_state['pi'], initial_state['pi'], eval_m))
    err_th = float(compute_safe_err(final_state['th_v'], initial_state['th_v'], eval_m))
    
    return dx, err_u, err_v, err_w, err_pi, err_th

def run_study(core_type):
    print(f"\n========================================")
    print(f"Running MMS study for {core_type.upper()} core...")
    print(f"========================================")
    
    resolutions = [8, 16, 32, 64, 128]
    results = []
    
    for r in resolutions:
        dx, eu, ev, ew, epi, eth = run_mms_at_resolution(core_type, r)
        results.append((dx, eu, ev, ew, epi, eth))
        print(f"Res: {r:2d} | dx: {dx/1000.:.1f}km | err_u: {eu:.2e} | err_v: {ev:.2e} | err_w: {ew:.2e} | err_pi: {epi:.2e} | err_th: {eth:.2e}")
        
    print("\n--- Spatial Convergence Rates ---")
    rates = {'u': [], 'v': [], 'w': [], 'pi': [], 'th': []}
    for i in range(len(resolutions) - 1):
        dx1, u1, v1, w1, pi1, th1 = results[i]
        dx2, u2, v2, w2, pi2, th2 = results[i+1]
        
        factor = dx1 / dx2
        rate_u = np.log(u1 / u2) / np.log(factor) if u2 > 0 else np.nan
        rate_v = np.log(v1 / v2) / np.log(factor) if v2 > 0 else np.nan
        rate_w = np.log(w1 / w2) / np.log(factor) if w2 > 0 else np.nan
        rate_pi = np.log(pi1 / pi2) / np.log(factor) if pi2 > 0 else np.nan
        rate_th = np.log(th1 / th2) / np.log(factor) if th2 > 0 else np.nan
        
        rates['u'].append(rate_u)
        rates['v'].append(rate_v)
        rates['w'].append(rate_w)
        rates['pi'].append(rate_pi)
        rates['th'].append(rate_th)
        
        print(f"Res {resolutions[i]} -> {resolutions[i+1]}:")
        print(f"  Rate u : {rate_u:.2f} (Expected ~2.00)" if not np.isnan(rate_u) else "  Rate u : N/A")
        print(f"  Rate v : {rate_v:.2f} (Expected ~2.00)" if not np.isnan(rate_v) else "  Rate v : N/A")
        print(f"  Rate w : {rate_w:.2f} (Expected ~2.00)" if not np.isnan(rate_w) else "  Rate w : N/A")
        print(f"  Rate pi: {rate_pi:.2f} (Expected ~2.00)" if not np.isnan(rate_pi) else "  Rate pi: N/A")
        print(f"  Rate th: {rate_th:.2f} (Expected ~2.00)" if not np.isnan(rate_th) else "  Rate th: N/A")
        
    return resolutions, results, rates

if __name__ == "__main__":
    res_sisl, results_sisl, rates_sisl = run_study("sisl")
    res_se, results_se, rates_se = run_study("split-explicit")
    
    plt.figure(figsize=(10, 8))
    dx_vals = [res[0] for res in results_sisl]
    err_u_sisl = [res[1] for res in results_sisl]
    err_u_se = [res[1] for res in results_se]
    
    valid_rates_sisl = [r for r in rates_sisl["u"] if not np.isnan(r)]
    valid_rates_se = [r for r in rates_se["u"] if not np.isnan(r)]
    avg_rate_sisl = np.mean(valid_rates_sisl) if valid_rates_sisl else 0.0
    avg_rate_se = np.mean(valid_rates_se) if valid_rates_se else 0.0
    
    plt.loglog(dx_vals, err_u_sisl, 'o-', label=f'SISL u-error (Avg Rate = {avg_rate_sisl:.2f})', linewidth=2, markersize=8)
    plt.loglog(dx_vals, err_u_se, 's-', label=f'Split-Explicit u-error (Avg Rate = {avg_rate_se:.2f})', linewidth=2, markersize=8)
    
    ref_start = max(err_u_sisl[0], err_u_se[0]) * 1.5
    ref_line = ref_start * (np.array(dx_vals) / dx_vals[0])**2
    plt.loglog(dx_vals, ref_line, 'k--', label='Theoretical 2nd Order', alpha=0.7)
    
    plt.xlabel(r'Grid Spacing $\Delta x$ (m)', fontsize=12)
    plt.ylabel('$L_2$ Error in $u$ (m/s)', fontsize=12)
    plt.title('MMS Time-Integrated Convergence Study: 3D Dynamical Core', fontsize=14, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=10, loc='lower right')
    
    out_path = 'output/plots/mms_convergence_study.png'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved MMS convergence plot to {out_path}")