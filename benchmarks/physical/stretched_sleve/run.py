import argparse
from pathlib import Path
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.steppers import SISLStepper3D
from suetes.shared.driver import Simulation
from suetes.shared.artifacts import ArtifactLayout, save_plot_dataset
from suetes.shared.transforms import SleveSimple, BaseTransform

class StretchedSleveSimple(BaseTransform):
    def __init__(self, stretch_kappa=3.0, scale_s=4000.0, n=1.35):
        self.kappa = stretch_kappa
        self.ss = scale_s
        self.n = n

    def __call__(self, xi, zeta, h, Lz):
        eta = zeta / Lz
        zeta_stretched = jnp.where(
            jnp.abs(self.kappa) > 1e-5,
            Lz * (jnp.exp(self.kappa * eta) - 1.0) / (jnp.exp(self.kappa) - 1.0),
            zeta
        )
        b_s = jnp.sinh((Lz - zeta_stretched) / self.ss) / jnp.sinh(Lz / self.ss)
        return zeta_stretched + h * (b_s ** self.n)


def run_stress_test(
    experiment='density_current',
    use_stretched=True,
    *,
    data_dir: Path,
    figure_dir: Path,
    render: bool,
):
    name = f"{experiment}_{'stretched' if use_stretched else 'standard'}"
    print(f"\n{'='*50}\nLaunching Setup: {name}\n{'='*50}")
    
    # ---------------------------------------------------------
    # EXPERIMENT CONFIGURATION
    # ---------------------------------------------------------
    if experiment == 'density_current':
        # Flat terrain, zero background wind, neutral stability
        h_func = lambda x, y: jnp.zeros_like(x)
        u_bg = 0.0
        N_bv = 0.00
        t_end = 900.0   
        dt = 2.0        
        nx, ny, nz = 400, 3, 50     
        dx, dy, dz = 200.0, 200.0, 400.0
        
    elif experiment == 'wave_breaking':
        # 1000m Agnesi mountain, moderate wind, high stability
        # Hits the non-linear downslope wind regime without blowing up
        h_func = lambda x, y: 1000.0 / (1.0 + (x / 2000.0)**2)
        u_bg = 12.0
        N_bv = 0.015
        t_end = 2400.0  # Reduced to 40 minutes to capture the wave establishment
        dt = 4.0
        nx, ny, nz = 300, 3, 60
        dx, dy, dz = 500.0, 500.0, 400.0
    else:
        raise ValueError("Unknown experiment type")

    # ---------------------------------------------------------
    # GRID & PHYSICS INITIALIZATION
    # ---------------------------------------------------------
    transform_op = StretchedSleveSimple(stretch_kappa=3.5) if use_stretched else SleveSimple()
    
    grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_center=45.0, lon_center=0.0, 
                          h_func=h_func, transform=transform_op)
    op = CGridOperator3D(grid)
    constants = {'g': 9.81, 'cp': 1004.0, 'Rd': 287.0, 'cvd': 717.0, 'p0': 100000.0}

    physics = Euler3D(grid, op, constants, dt=dt, N_bv=N_bv, damp_height=12000.0, 
                      max_damp=0.5, nu_div_factor=0.5, nu_h_factor=0.2, physics_suite=None)

    bg_ref = {
        'rho': physics.c['p0'] / (physics.c['Rd'] * physics.theta_bg) * \
               (physics.pi_bg ** (physics.c['cvd'] / physics.c['Rd'])),
        'pi': physics.pi_bg,
        'th_v': physics.theta_bg
    }

    state = {
        'u': jnp.ones((nx+1, ny, nz)) * u_bg,
        'v': jnp.zeros((nx, ny+1, nz)),
        'w': jnp.zeros((nx, ny, nz+1)),
        'pi': bg_ref['pi'],
        'eta_dot': jnp.zeros((nx, ny, nz+1)),
        'rho': bg_ref['rho'],
        'th_v': bg_ref['th_v']
    }

    # ---------------------------------------------------------
    # ADD COLD POOL PERTURBATION
    # ---------------------------------------------------------
    if experiment == 'density_current':
        x_c, z_c = 0.0, 3000.0
        x_r, z_r = 4000.0, 2000.0
        
        X_3d = grid.x_m[:, None, None]
        r = jnp.sqrt(((X_3d - x_c) / x_r)**2 + ((grid.Z_m - z_c) / z_r)**2)
        
        theta_prime = jnp.where(r <= 1.0, -15.0 * jnp.cos(0.5 * jnp.pi * r)**2, 0.0)
        state['th_v'] += theta_prime
        
        state['rho'] = constants['p0'] / (constants['Rd'] * state['th_v']) * \
                       (state['pi'] ** (constants['cvd'] / constants['Rd']))

    # ---------------------------------------------------------
    # BOUNDARIES & RUNNER
    # ---------------------------------------------------------
    sponge_depth = 12
    x_idx = jnp.arange(nx, dtype=jnp.float32)
    dist_x = jnp.minimum(x_idx, nx - x_idx)
    weight_x = jnp.where(dist_x < sponge_depth, jnp.cos(0.5 * jnp.pi * dist_x / sponge_depth)**2, 0.0)
    mask_x = weight_x[:, None, None] 

    def bc_fn(state_in, forcing):
        ext_state = {
            'u': jnp.ones_like(state_in['u']) * u_bg,
            'v': jnp.zeros_like(state_in['v']),
            'th_v': bg_ref['th_v'],
            'rho': bg_ref['rho'],
            'pi': bg_ref['pi']
        }
        blended = {}
        for k in state_in.keys():
            if k in ['u', 'v', 'th_v', 'pi', 'rho'] and k in ext_state:
                m = jnp.pad(mask_x, ((0, 1), (0, 0), (0, 0)), mode='edge') if k == 'u' else mask_x
                blended[k] = (1.0 - m) * state_in[k] + m * ext_state[k]
            else:
                blended[k] = state_in[k]
        return blended

    stepper = SISLStepper3D(physics, dt)

    def step_fn(curr_state, step_idx):
        t_curr = step_idx * dt
        next_state = stepper.step(curr_state, t_curr, forcing=None, bc_fn=bc_fn)
        # Calculate and return the maximum vertical velocity metric
        max_w = jnp.max(jnp.abs(next_state['w']))
        return next_state, max_w

    sim = Simulation(step_fn=step_fn, dt=dt)
    final_state = sim.run(state, t_start=0.0, t_end=t_end, chunk_steps=20)

    # ---------------------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------------------
    is_w_plot = (experiment == 'wave_breaking')
    var_to_plot = final_state['w'][:, 1, :] if is_w_plot else final_state['th_v'][:, 1, :]
    cmap_str = 'RdBu_r' if is_w_plot else 'coolwarm'
    
    nz_plot = nz + 1 if is_w_plot else nz
    Z_3d = grid.Z_w if is_w_plot else grid.Z_m
    
    x_plot_1d = grid.x_m / 1000.0 
    X_plot, _ = jnp.meshgrid(x_plot_1d, jnp.arange(nz_plot), indexing='ij')
    Z_plot = Z_3d[:, 1, :] / 1000.0 

    plt.figure(figsize=(14, 6))
    contour = plt.contourf(X_plot, Z_plot, var_to_plot, levels=50, cmap=cmap_str, extend='both')
    plt.colorbar(contour, label='W (m/s)' if is_w_plot else 'Theta-V (K)')
    
    # Plot grid levels
    for k in range(grid.nz + 1):
        z_line = grid.Z_w[:, 1, k] / 1000.0
        plt.plot(x_plot_1d, z_line, color='black', alpha=0.15, linewidth=0.5)

    plt.title(f'{name} at T = {t_end}s')
    plt.xlabel('Distance (km)')
    plt.ylabel('Altitude (km)')
    plt.ylim(0, 10.0) # Zoom in to bottom 10km

    terrain = h_func(grid.x_m, 0.0) / 1000.0
    artifact = xr.Dataset(
        data_vars={
            "field": (("x", "z"), np.asarray(var_to_plot)),
            "physical_height": (("x", "z"), np.asarray(Z_plot) * 1000.0),
            "terrain_height": ("x", np.asarray(terrain) * 1000.0),
        },
        coords={
            "x": np.asarray(grid.x_m),
            "z": np.arange(nz_plot),
        },
        attrs={
            "experiment": experiment,
            "coordinate": "stretched" if use_stretched else "standard",
            "variable": "w" if is_w_plot else "th_v",
            "dt_s": dt,
            "t_end_s": t_end,
        },
    )
    artifact_path = save_plot_dataset(
        artifact, data_dir / f"{name}.nc",
        experiment="stretched_sleve_stress_test",
    )
    print(f"Saved plot-ready artifact to '{artifact_path}'")
    if not render:
        plt.close()
        return
    plt.fill_between(x_plot_1d, 0, terrain, color='black')

    filename = figure_dir / f"{name}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to '{filename}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare stretched and standard SLEVE coordinates")
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--name", default="default")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        layout = ArtifactLayout(
            kind="benchmarks", case="stretched_sleve",
            execution=args.name, output_root=args.output_root,
        ).create()
        data_dir, figure_dir = layout.data, layout.figures
    else:
        data_dir = figure_dir = args.output_dir
        data_dir.mkdir(parents=True, exist_ok=True)
    for exp in ['density_current', 'wave_breaking']:
        for stretched in [True, False]:
            run_stress_test(
                experiment=exp, use_stretched=stretched,
                data_dir=data_dir, figure_dir=figure_dir,
                render=not args.no_render,
            )
