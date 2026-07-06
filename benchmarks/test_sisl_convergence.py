import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from suetes.shared.driver import Simulation
from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.operators import CGridOperator3D, tensor_product_interp_3d
from suetes.regional3d.steppers import SemiLagrangianAdvector3D, SISLStepper3D, build_dynamical_core
from suetes.regional3d.euler import Euler3D

# Length scale of domain
L_x, L_y, L_z = 10000.0, 10000.0, 10000.0

def get_exact_function(x, y, z):
    return jnp.sin(2.0 * jnp.pi * x / L_x) * jnp.cos(2.0 * jnp.pi * y / L_y) * jnp.sin(2.0 * jnp.pi * z / L_z)

def test_interpolation():
    print("\n=== TEST 1: Spatial Interpolation Convergence (Tricubic) ===")
    resolutions = [16, 32, 64]
    errors = []
    
    for N in resolutions:
        dx = L_x / N
        dy = L_y / N
        dz = L_z / N
        
        grid = RegionalGrid3D(N, N, N, dx, dy, dz, lat_center=0.0, lon_center=0.0)
        
        # Grid center coordinates
        X, Y, Z = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_m, indexing='ij')
        field = get_exact_function(X, Y, Z)
        
        # Interpolate to offset coords: X_target = X + 0.37 * dx
        # Fractional coordinates: x_target = x_idx + 0.37
        X_target = X + 0.37 * dx
        Y_target = Y + 0.23 * dy
        Z_target = Z + 0.45 * dz
        
        # Fractional indices corresponding to the physical target coordinates
        # Since grid.x_m[i] = i*dx - Lx/2 + dx/2, we have:
        # X_target = x_idx * dx - Lx/2 + dx/2  =>  x_idx = (X_target + Lx/2) / dx - 0.5
        coords = (
            (X_target + L_x / 2.0) / dx - 0.5,
            (Y_target + L_y / 2.0) / dy - 0.5,
            Z_target / dz - 0.5
        )
        
        interp_field = tensor_product_interp_3d(field, coords, use_limiter=False)
        exact_field = get_exact_function(X_target, Y_target, Z_target)
        
        # Compute L2 error (interior to avoid boundary extrapolation effects)
        # We look at the middle 50% of the domain
        margin = N // 4
        err = jnp.sqrt(jnp.mean((interp_field[margin:-margin, margin:-margin, margin:-margin] - 
                                 exact_field[margin:-margin, margin:-margin, margin:-margin])**2))
        errors.append(float(err))
        print(f"N = {N:2d} | dx = {dx:6.1f}m | L2 Error: {err:.2e}")
        
    for i in range(len(errors) - 1):
        order = np.log2(errors[i] / errors[i+1])
        print(f"  Order (N={resolutions[i]} -> N={resolutions[i+1]}): {order:.2f}")

def test_operators():
    print("\n=== TEST 2: C-Grid Operator (diff and avg) Convergence ===")
    resolutions = [16, 32, 64]
    
    # We will test diff along axis 0 (x-axis) from mass ('m') to 'u'
    # df/dx of sin(2*pi*x/Lx) is (2*pi/Lx) * cos(2*pi*x/Lx)
    errors_diff = []
    errors_avg = []
    
    for N in resolutions:
        dx = L_x / N
        dy = L_y / N
        dz = L_z / N
        
        grid = RegionalGrid3D(N, N, N, dx, dy, dz, lat_center=0.0, lon_center=0.0)
        op = CGridOperator3D(grid)
        
        X, Y, Z = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_m, indexing='ij')
        f_m = get_exact_function(X, Y, Z)
        
        # Derivative w.r.t x (axis 0) from 'm' to 'u'
        df_dx_num = op.diff(f_m, axis=0, from_loc='m', to_loc='u')
        
        # Exact derivative at u-grid points
        # u points are at grid.x_c
        X_u, Y_u, Z_u = jnp.meshgrid(grid.x_c, grid.y_m, grid.z_m, indexing='ij')
        df_dx_exact = (2.0 * jnp.pi / L_x) * jnp.cos(2.0 * jnp.pi * X_u / L_x) * jnp.cos(2.0 * jnp.pi * Y_u / L_y) * jnp.sin(2.0 * jnp.pi * Z_u / L_z)
        
        margin = N // 4
        err_diff = jnp.sqrt(jnp.mean((df_dx_num[margin:-margin, margin:-margin, margin:-margin] - 
                                      df_dx_exact[margin:-margin, margin:-margin, margin:-margin])**2))
        errors_diff.append(float(err_diff))
        
        # Average along x (axis 0) from 'm' to 'u'
        avg_num = op.avg(f_m, axis=0, from_loc='m', to_loc='u')
        avg_exact = jnp.cos(jnp.pi * dx / L_x) * get_exact_function(X_u, Y_u, Z_u)
        
        err_avg = jnp.sqrt(jnp.mean((avg_num[margin:-margin, margin:-margin, margin:-margin] - 
                                    avg_exact[margin:-margin, margin:-margin, margin:-margin])**2))
        errors_avg.append(float(err_avg))
        
        print(f"N = {N:2d} | L2 Diff Error: {err_diff:.2e} | L2 Avg Error: {err_avg:.2e}")
        
    print("Diff Order:")
    for i in range(len(errors_diff) - 1):
        order = np.log2(errors_diff[i] / errors_diff[i+1])
        print(f"  Order (N={resolutions[i]} -> N={resolutions[i+1]}): {order:.2f}")
        
    print("Avg Order:")
    for i in range(len(errors_avg) - 1):
        order = np.log2(errors_avg[i] / errors_avg[i+1])
        print(f"  Order (N={resolutions[i]} -> N={resolutions[i+1]}): {order:.2f}")

def test_trajectory():
    print("\n=== TEST 3: Trajectory Solver (compute_departure_indices) ===")
    resolutions = [16, 32, 64]
    errors = []
    
    U0, V0 = 50.0, 30.0
    
    for N in resolutions:
        dx = L_x / N
        dy = L_y / N
        dz = L_z / N
        
        # Scaling dt with dx
        dt = (dx / 125.0) * 5.0
        
        grid = RegionalGrid3D(N, N, N, dx, dy, dz, lat_center=0.0, lon_center=0.0)
        op = CGridOperator3D(grid)
        constants = {'g': 9.81, 'cp': 1004.0, 'Rd': 287.0, 'cvd': 717.0, 'p0': 100000.0}
        
        X_u, Y_u, Z_u = jnp.meshgrid(grid.x_c, grid.y_m, grid.z_m, indexing='ij')
        X_v, Y_v, Z_v = jnp.meshgrid(grid.x_m, grid.y_c, grid.z_m, indexing='ij')
        X_w, Y_w, Z_w = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_c, indexing='ij')
        
        u = U0 * jnp.sin(2.0 * jnp.pi * Y_u / L_y)
        v = V0 * jnp.cos(2.0 * jnp.pi * Z_v / L_z)
        eta_dot = jnp.zeros_like(X_w)
        
        state = {'u': u, 'v': v, 'w': jnp.zeros_like(X_w), 'eta_dot': eta_dot}
        
        phys = Euler3D(grid, op, constants, dt=dt)
        advector = SemiLagrangianAdvector3D(grid, phys, dt)
        
        dep_coords = advector.compute_departure_indices(state, loc='m', iterations=2)
        
        X_m, Y_m, Z_m = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_m, indexing='ij')
        
        v_const = V0 * jnp.cos(2.0 * jnp.pi * Z_m / L_z)
        v_const_safe = jnp.where(jnp.abs(v_const) < 1e-10, 1e-10, v_const)
        
        dy_dep = -dt * v_const
        
        dx_dep = - (L_y * U0) / (2.0 * jnp.pi * v_const_safe) * (
            jnp.cos(2.0 * jnp.pi * Y_m / L_y) - jnp.cos(2.0 * jnp.pi * (Y_m - v_const_safe * dt) / L_y)
        )
        dx_dep = jnp.where(jnp.abs(v_const) < 1e-10, -dt * U0 * jnp.sin(2.0 * jnp.pi * Y_m / L_y), dx_dep)
        
        x_d_exact = X_m - dx_dep
        y_d_exact = Y_m + dy_dep
        z_d_exact = Z_m
        
        x_d_exact_idx = (x_d_exact + L_x / 2.0) / dx - 0.5
        y_d_exact_idx = (y_d_exact + L_y / 2.0) / dy - 0.5
        z_d_exact_idx = z_d_exact / dz - 0.5
        
        margin = N // 4
        err_x = jnp.sqrt(jnp.mean((dep_coords[0][margin:-margin, margin:-margin, margin:-margin] - x_d_exact_idx[margin:-margin, margin:-margin, margin:-margin])**2))
        err_y = jnp.sqrt(jnp.mean((dep_coords[1][margin:-margin, margin:-margin, margin:-margin] - y_d_exact_idx[margin:-margin, margin:-margin, margin:-margin])**2))
        
        mid = N // 2
        print(f"DEBUG N={N} middle point:")
        print(f"  Exact physical x_d: {x_d_exact[mid, mid, mid]:.2f}, y_d: {y_d_exact[mid, mid, mid]:.2f}")
        print(f"  Exact index    x_d: {x_d_exact_idx[mid, mid, mid]:.4f}, y_d: {y_d_exact_idx[mid, mid, mid]:.4f}")
        print(f"  Numeric index  x_d: {dep_coords[0][mid, mid, mid]:.4f}, y_d: {dep_coords[1][mid, mid, mid]:.4f}")
        print(f"  Delta index    x: {dep_coords[0][mid, mid, mid] - x_d_exact_idx[mid, mid, mid]:.4f}, y: {dep_coords[1][mid, mid, mid] - y_d_exact_idx[mid, mid, mid]:.4f}")
        
        err = float(jnp.sqrt(err_x**2 + err_y**2))
        errors.append(err)
        print(f"N = {N:2d} | dt = {dt:5.2f}s | L2 Departure Index Error: {err:.2e}")
        
    for i in range(len(errors) - 1):
        order = np.log2(errors[i] / errors[i+1])
        print(f"  Order (N={resolutions[i]} -> N={resolutions[i+1]}): {order:.2f}")

def run_bubble_at_resolution(dx, dt):
    nx = int(10000 / dx)
    nz = int(10000 / dx)
    ny = 3
    
    core_kwargs = {"dt": dt, "nu_div_factor": 0.0, "nu_h_factor": 0.0, "damp_height": 7500.0, "max_damp": 0.05, "alpha": 0.5,
        "solver_tol": 1e-10, "solver_maxiter": 100, "solver_restart": 20}

    grid = RegionalGrid3D(nx, ny, nz, dx, dx, dx, lat_center=0.0, lon_center=0.0)
    op = CGridOperator3D(grid)
    constants = {'g': 9.81, 'cp': 1004.0, 'Rd': 287.0, 'cvd': 717.0, 'p0': 100000.0}

    tmp_phys = Euler3D(grid, op, constants, dt=dt)
    bg_ref = {
        'rho': tmp_phys.c['p0'] / (tmp_phys.c['Rd'] * tmp_phys.theta_bg) * \
               (tmp_phys.pi_bg ** (tmp_phys.c['cvd'] / tmp_phys.c['Rd'])),
        'pi': tmp_phys.pi_bg, 'th_v': tmp_phys.theta_bg
    }

    state = {
        'u': jnp.zeros((nx+1, ny, nz)), 'v': jnp.zeros((nx, ny+1, nz)), 'w': jnp.zeros((nx, ny, nz+1)),
        'pi': bg_ref['pi'], 'eta_dot': jnp.zeros((nx, ny, nz+1)), 'rho': bg_ref['rho'],
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

    sim = Simulation(step_fn=lambda s, i: (stepper.step(s, i*dt, None, lambda x, f: x), jnp.max(jnp.abs(s['w']))), dt=dt)
    return sim.run(state, t_start=0.0, t_end=200.0, chunk_steps=int(200.0/dt))

def test_bubble_convergence(dt_mode="scaling"):
    print(f"\n=== TEST 4: Full SISL Bubble Convergence (dt_mode={dt_mode}) ===")
    
    resolutions = [250.0, 125.0, 62.5]
    
    if dt_mode == "scaling":
        dts = [10.0, 5.0, 2.5]
    else: # constant dt
        dts = [0.5, 0.5, 0.5]
        
    print(f"Resolutions: {resolutions}")
    print(f"Timesteps: {dts}")
    
    res_250 = run_bubble_at_resolution(250.0, dts[0])
    res_125 = run_bubble_at_resolution(125.0, dts[1])
    res_062 = run_bubble_at_resolution(62.5, dts[2])
    
    def block_average_2d(field_2d, factor):
        nx, nz = field_2d.shape
        return jnp.mean(field_2d.reshape(nx // factor, factor, nz // factor, factor), axis=(1, 3))
        
    err_coarse = float(jnp.sqrt(jnp.mean((block_average_2d(res_125['th_v'][:, 1, :], 2) - res_250['th_v'][:, 1, :])**2)))
    err_fine = float(jnp.sqrt(jnp.mean((block_average_2d(res_062['th_v'][:, 1, :], 4) - block_average_2d(res_125['th_v'][:, 1, :], 2))**2)))
    order = np.log2(err_coarse / err_fine)
    
    # Interior-only (cropped) convergence to avoid boundary/sponge impacts
    crop = 6 # Crop 6 cells from boundaries of 250m grid
    val_250 = res_250['th_v'][crop:-crop, 1, crop:-crop]
    val_125_avg = block_average_2d(res_125['th_v'][:, 1, :], 2)[crop:-crop, crop:-crop]
    val_062_avg = block_average_2d(res_062['th_v'][:, 1, :], 4)[crop:-crop, crop:-crop]
    
    err_coarse_int = float(jnp.sqrt(jnp.mean((val_125_avg - val_250)**2)))
    err_fine_int = float(jnp.sqrt(jnp.mean((val_062_avg - val_125_avg)**2)))
    order_int = np.log2(err_coarse_int / err_fine_int)
    
    print(f"Global L2 Error (250m vs 125m): {err_coarse:.2e}")
    print(f"Global L2 Error (125m vs  62m): {err_fine:.2e}")
    print(f"Global Empirical Order of Convergence: {order:.2f}")
    print(f"Interior-only L2 Error (250m vs 125m): {err_coarse_int:.2e}")
    print(f"Interior-only L2 Error (125m vs  62m): {err_fine_int:.2e}")
    print(f"Interior-only Empirical Order of Convergence: {order_int:.2f}")

def test_ffsl_advection():
    print("\n=== TEST 5: FluxFormAdvector (advect_3d_split) Spatial Convergence ===")
    from suetes.regional3d.steppers import FluxFormAdvector
    resolutions = [16, 32, 64]
    errors = []
    
    # Smooth density field: rho(x) = sin(2*pi*x/Lx) + 2.0 (always positive)
    # Constant velocity: u = 10 m/s, v = 0, w = 0
    # Time step: dt = (dx / 125.0) * 5.0
    # Run for 1 step, compare to exact shifted analytical solution: rho_exact(x) = rho(x - u*dt)
    
    for N in resolutions:
        dx = L_x / N
        dy = L_y / N
        dz = L_z / N
        
        dt = (dx / 125.0) * 5.0
        
        grid = RegionalGrid3D(N, N, N, dx, dy, dz, lat_center=0.0, lon_center=0.0)
        advector = FluxFormAdvector(grid, dt)
        
        # We need state with u, v, eta_dot
        state = {
            'u': jnp.ones((N+1, N, N)) * 10.0,
            'v': jnp.zeros((N, N+1, N)),
            'eta_dot': jnp.zeros((N, N, N+1))
        }
        
        # Precomputed background state
        bg_precomputed = {
            'dz_m_full': jnp.ones((N, N, N)) * dz
        }
        
        X, Y, Z = jnp.meshgrid(grid.x_m, grid.y_m, grid.z_m, indexing='ij')
        rho_initial = jnp.sin(2.0 * jnp.pi * X / L_x) + 2.0
        
        rho_next = advector.advect_3d_split(rho_initial, state, bg_precomputed)
        
        # Exact solution is shifted: rho_exact(x) = sin(2*pi*(x - u*dt)/Lx) + 2.0
        x_shifted = X - 10.0 * dt
        rho_exact = jnp.sin(2.0 * jnp.pi * x_shifted / L_x) + 2.0
        
        margin = N // 4
        err = jnp.sqrt(jnp.mean((rho_next[margin:-margin, margin:-margin, margin:-margin] - 
                                 rho_exact[margin:-margin, margin:-margin, margin:-margin])**2))
        errors.append(float(err))
        print(f"N = {N:2d} | dt = {dt:5.2f}s | L2 Error: {err:.2e}")
        
    for i in range(len(errors) - 1):
        order = np.log2(errors[i] / errors[i+1])
        print(f"  Order (N={resolutions[i]} -> N={resolutions[i+1]}): {order:.2f}")

if __name__ == "__main__":
    test_interpolation()
    test_operators()
    test_trajectory()
    test_ffsl_advection()
    test_bubble_convergence(dt_mode="scaling")
    test_bubble_convergence(dt_mode="constant")
