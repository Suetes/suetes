import os
import jax
import jax.numpy as jnp
import numpy as np

# Enable X64 for high-precision convergence testing
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)

from suetes.regional3d.geometry import RegionalGrid3D, ObliqueStereographic
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.euler import Euler3D

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

# Manufactured Solutions
def u_func(x, y, z):
    return 10.0 * jnp.sin(2.0 * jnp.pi * x / Lx) * jnp.cos(2.0 * jnp.pi * y / Ly) * jnp.sin(jnp.pi * z / Lz)

def v_func(x, y, z):
    return 15.0 * jnp.cos(2.0 * jnp.pi * x / Lx) * jnp.sin(2.0 * jnp.pi * y / Ly) * jnp.sin(jnp.pi * z / Lz)

def w_func(x, y, z):
    return 1.0 * jnp.sin(2.0 * jnp.pi * x / Lx) * jnp.cos(2.0 * jnp.pi * y / Ly) * jnp.cos(jnp.pi * z / Lz)

def pip_func(x, y, z):
    return 0.01 * jnp.sin(2.0 * jnp.pi * x / Lx) * jnp.sin(2.0 * jnp.pi * y / Ly) * jnp.sin(jnp.pi * z / Lz)

def thvp_func(x, y, z):
    return 5.0 * jnp.cos(2.0 * jnp.pi * x / Lx) * jnp.cos(2.0 * jnp.pi * y / Ly) * jnp.sin(jnp.pi * z / Lz)

def th_bg_func(z):
    return theta_0 + 0.01 * z

def pi_bg_func(z):
    return jnp.ones_like(z)

def rho_bg_func(z):
    return p0 / (Rd * theta_0) * jnp.ones_like(z)

# Analytical RHS Tendencies
@jax.jit
def analytic_tend_u(x, y, z):
    m = 1.0 + (x**2 + y**2) / (4.0 * R_earth**2)
    dm_dx = x / (2.0 * R_earth**2)
    dm_dy = y / (2.0 * R_earth**2)
    
    lat, _ = proj.get_lat_lon(x, y)
    f = 2.0 * Omega * jnp.sin(jnp.radians(lat))
    
    u_val = u_func(x, y, z)
    v_val = v_func(x, y, z)
    
    th_bg_val = th_bg_func(z)
    thvp_val = thvp_func(x, y, z)
    th_v = th_bg_val + thvp_val
    
    dpip_dx = jax.grad(lambda x_val: pip_func(x_val, y, z))(x)
    metric = v_val * dm_dx - u_val * dm_dy
    
    return -cp * th_v * m * dpip_dx + f * v_val + metric * v_val

@jax.jit
def analytic_tend_v(x, y, z):
    m = 1.0 + (x**2 + y**2) / (4.0 * R_earth**2)
    dm_dx = x / (2.0 * R_earth**2)
    dm_dy = y / (2.0 * R_earth**2)
    
    lat, _ = proj.get_lat_lon(x, y)
    f = 2.0 * Omega * jnp.sin(jnp.radians(lat))
    
    u_val = u_func(x, y, z)
    v_val = v_func(x, y, z)
    
    th_bg_val = th_bg_func(z)
    thvp_val = thvp_func(x, y, z)
    th_v = th_bg_val + thvp_val
    
    dpip_dy = jax.grad(lambda y_val: pip_func(x, y_val, z))(y)
    metric = v_val * dm_dx - u_val * dm_dy
    
    return -cp * th_v * m * dpip_dy - f * u_val - metric * u_val

@jax.jit
def analytic_tend_w(x, y, z):
    th_bg_val = th_bg_func(z)
    thvp_val = thvp_func(x, y, z)
    th_v = th_bg_val + thvp_val
    
    dpip_dz = jax.grad(lambda z_val: pip_func(x, y, z_val))(z)
    
    return -cp * th_v * dpip_dz

@jax.jit
def analytic_tend_pi(x, y, z):
    m = 1.0 + (x**2 + y**2) / (4.0 * R_earth**2)
    
    def flux_x(x_val):
        rho = rho_bg_func(z)
        th = th_bg_func(z)
        u = u_func(x_val, y, z)
        m_local = 1.0 + (x_val**2 + y**2) / (4.0 * R_earth**2)
        return rho * th * u / m_local
        
    def flux_y(y_val):
        rho = rho_bg_func(z)
        th = th_bg_func(z)
        v = v_func(x, y_val, z)
        m_local = 1.0 + (x**2 + y_val**2) / (4.0 * R_earth**2)
        return rho * th * v / m_local
        
    def flux_z(z_val):
        rho = rho_bg_func(z_val)
        th = th_bg_func(z_val)
        w = w_func(x, y, z_val)
        return rho * th * w
        
    dflux_x = jax.grad(flux_x)(x)
    dflux_y = jax.grad(flux_y)(y)
    dflux_z = jax.grad(flux_z)(z)
    
    div_total = m**2 * (dflux_x + dflux_y) + dflux_z
    
    C_pi = (Rd / cvd) * pi_bg_func(z) / (rho_bg_func(z) * th_bg_func(z))
    return -C_pi * div_total

@jax.jit
def analytic_tend_th(x, y, z):
    w_val = w_func(x, y, z)
    dth_bg_dz = jax.grad(th_bg_func)(z)
    return -w_val * dth_bg_dz

# Vmap analytical functions
vmap_u = jax.vmap(jax.vmap(jax.vmap(analytic_tend_u, in_axes=(0, 0, 0)), in_axes=(0, 0, 0)), in_axes=(0, 0, 0))
vmap_v = jax.vmap(jax.vmap(jax.vmap(analytic_tend_v, in_axes=(0, 0, 0)), in_axes=(0, 0, 0)), in_axes=(0, 0, 0))
vmap_w = jax.vmap(jax.vmap(jax.vmap(analytic_tend_w, in_axes=(0, 0, 0)), in_axes=(0, 0, 0)), in_axes=(0, 0, 0))
vmap_pi = jax.vmap(jax.vmap(jax.vmap(analytic_tend_pi, in_axes=(0, 0, 0)), in_axes=(0, 0, 0)), in_axes=(0, 0, 0))
vmap_th = jax.vmap(jax.vmap(jax.vmap(analytic_tend_th, in_axes=(0, 0, 0)), in_axes=(0, 0, 0)), in_axes=(0, 0, 0))


def evaluate_resolution(n):
    # Construct grid
    nx, ny, nz = n, n, n
    dx = Lx / nx
    dy = Ly / ny
    dz = Lz / nz
    
    grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_center=60.0, lon_center=30.0)
    op = CGridOperator3D(grid)
    
    constants_dict = {'g': 0.0, 'cp': cp, 'cvd': cvd, 'Rd': Rd, 'p0': p0}
    euler = Euler3D(grid, op, constants_dict, dt=1.0, N_bv=0.0, nu_h_factor=0.0, nu_div_factor=0.0)
    
    # Evaluate grid points for each stagger
    X_u, Y_u, Z_u = jnp.meshgrid(grid.x_c, grid.y_m, grid.z_m, indexing='ij')
    X_v, Y_v, Z_v = jnp.meshgrid(grid.x_m, grid.y_c, grid.z_m, indexing='ij')
    X_w, Y_w, Z_w = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_c, indexing='ij')
    X_m, Y_m, Z_m = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_m, indexing='ij')
    
    # State values
    state_prime = {
        'u': u_func(X_u, Y_u, Z_u),
        'v': v_func(X_v, Y_v, Z_v),
        'w': w_func(X_w, Y_w, Z_w),
        'pi': pip_func(X_m, Y_m, Z_m),
        'th_v_prime_u': thvp_func(X_u, Y_u, Z_u),
        'th_v_prime_v': thvp_func(X_v, Y_v, Z_v),
        'th_v_prime_w': thvp_func(X_w, Y_w, Z_w),
        'th_v': th_bg_func(X_m) + thvp_func(X_m, Y_m, Z_m),
    }
    
    # Compute contravariant velocity eta_dot
    state_prime['eta_dot'] = state_prime['w'] / grid.dz_w_full
    
    bg_state_ref = {
        'rho': rho_bg_func(Z_m),
        'pi': pi_bg_func(Z_m),
        'th_v': th_bg_func(Z_m)
    }
    bg = euler.precompute_bg(bg_state_ref)
    
    # Get model tendencies
    model_tends = euler.get_tendencies(state_prime, bg, is_explicit=True)
    
    # Compute analytical tendencies
    ana_u = vmap_u(X_u, Y_u, Z_u)
    ana_v = vmap_v(X_v, Y_v, Z_v)
    ana_w = vmap_w(X_w, Y_w, Z_w)
    ana_pi = vmap_pi(X_m, Y_m, Z_m)
    ana_th = vmap_th(X_m, Y_m, Z_m)
    
    # L2 Errors (ignoring boundary points where derivatives are padded/extrapolated)
    sl = slice(2, -2)
    
    err_u = jnp.sqrt(jnp.mean((model_tends['u'][sl, sl, sl] - ana_u[sl, sl, sl])**2))
    err_v = jnp.sqrt(jnp.mean((model_tends['v'][sl, sl, sl] - ana_v[sl, sl, sl])**2))
    err_w = jnp.sqrt(jnp.mean((model_tends['w'][sl, sl, sl] - ana_w[sl, sl, sl])**2))
    err_pi = jnp.sqrt(jnp.mean((model_tends['pi'][sl, sl, sl] - ana_pi[sl, sl, sl])**2))
    err_th = jnp.sqrt(jnp.mean((model_tends['th_v'][sl, sl, sl] - ana_th[sl, sl, sl])**2))
    
    # All convergence evaluations are complete
        
    return dx, float(err_u), float(err_v), float(err_w), float(err_pi), float(err_th)

if __name__ == "__main__":
    resolutions = [16, 32, 64]
    results = []
    for r in resolutions:
        dx, eu, ev, ew, epi, eth = evaluate_resolution(r)
        results.append((dx, eu, ev, ew, epi, eth))
        print(f"Res: {r:2d} | dx: {dx/1000.:.1f}km | err_u: {eu:.2e} | err_v: {ev:.2e} | err_w: {ew:.2e} | err_pi: {epi:.2e} | err_th: {eth:.2e}")
        if r == 16:
            pass
        
    print("\n--- Spatial Convergence Rates ---")
    for i in range(len(resolutions) - 1):
        dx1, u1, v1, w1, pi1, th1 = results[i]
        dx2, u2, v2, w2, pi2, th2 = results[i+1]
        
        factor = dx1 / dx2
        rate_u = np.log(u1 / u2) / np.log(factor) if u2 > 0 else np.nan
        rate_v = np.log(v1 / v2) / np.log(factor) if v2 > 0 else np.nan
        rate_w = np.log(w1 / w2) / np.log(factor) if w2 > 0 else np.nan
        rate_pi = np.log(pi1 / pi2) / np.log(factor) if pi2 > 0 else np.nan
        rate_th = np.log(th1 / th2) / np.log(factor) if th2 > 0 else np.nan
        
        print(f"Res {resolutions[i]} -> {resolutions[i+1]}:")
        print(f"  Rate u : {rate_u:.2f} (Expected ~2.00)" if not np.isnan(rate_u) else "  Rate u : N/A")
        print(f"  Rate v : {rate_v:.2f} (Expected ~2.00)" if not np.isnan(rate_v) else "  Rate v : N/A")
        print(f"  Rate w : {rate_w:.2f} (Expected ~2.00)" if not np.isnan(rate_w) else "  Rate w : N/A")
        print(f"  Rate pi: {rate_pi:.2f} (Expected ~2.00)" if not np.isnan(rate_pi) else "  Rate pi: N/A")
        print(f"  Rate th: {rate_th:.2f} (Expected ~2.00)" if not np.isnan(rate_th) else "  Rate th: N/A")
