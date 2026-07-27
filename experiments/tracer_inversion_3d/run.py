# Case runner.
import argparse
from pathlib import Path
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from suetes.regional3d.geometry import RegionalGrid3D
from suetes.regional3d.euler import Euler3D
from suetes.regional3d.operators import CGridOperator3D
from suetes.regional3d.steppers import build_dynamical_core
from suetes.physics.base import PhysicsSuite
from suetes.regional3d.boundaries import BenchmarkSponge

from suetes.shared.driver import Simulation
from suetes.shared.optimization import OptaxSolver
from suetes.shared.experiment import save_plot_dataset, add_experiment_args, setup_experiment_directories

parser = argparse.ArgumentParser(description="Run the 3-D tracer inversion experiment")
add_experiment_args(parser)
parser.add_argument("--no-render", action="store_true")
args = parser.parse_args()
data_dir, figure_dir = setup_experiment_directories(
    args, kind="experiments", case="tracer_inversion_3d"
)

# =====================================================================
# CONFIGURATION SWITCHES
# =====================================================================
CORE_TYPE = "split-explicit"  # Toggle to "sisl" or "split-explicit"

# --- 1. SETUP DOMAIN & PHYSICS ---
nx, ny, nz = 200, 3, 40
dx, dy, dz = 400.0, 400.0, 250.0
dt = 5.0
t_end = 2400.0  # 40 mins of advection
num_steps = int(t_end / dt)
u_bg = 10.0

constants = {"g": 9.81, "cp": 1004.0, "Rd": 287.0, "cvd": 717.0, "p0": 100000.0}


def terrain_profile(x, y):
    h0 = 2000.0  # 2 km height
    a = 4000.0  # 4 km width spread
    center = -5000.0  # Positioned at X = -5 km
    return h0 * jnp.exp(-(((x - center) / a) ** 2))


grid = RegionalGrid3D(nx, ny, nz, dx, dy, dz, lat_center=45.0, lon_center=0.0, h_func=terrain_profile)
op = CGridOperator3D(grid)

suite = PhysicsSuite()
suite.register_tracer("q_tr")

physics = Euler3D(
    grid,
    op,
    constants,
    dt=dt,
    N_bv=0.01,
    damp_height=8000.0,
    max_damp=0.5,
    nu_div_factor=0.0,
    nu_h_factor=0.0,
    physics_suite=suite,
)

# --- 2. STATE GENERATOR & BOUNDARY CONDITIONS ---
bg_ref = {
    "rho": physics.c["p0"]
    / (physics.c["Rd"] * physics.theta_bg)
    * (physics.pi_bg ** (physics.c["cvd"] / physics.c["Rd"])),
    "pi": physics.pi_bg,
    "th_v": physics.theta_bg,
}

x_sponge = BenchmarkSponge(nx=nx, sponge_depth=10, axes=("x",))


def bc_fn(state_in, forcing=None):
    ext_state = {
        "u": jnp.ones_like(state_in["u"]) * u_bg,
        "v": jnp.zeros_like(state_in["v"]),
        "th_v": bg_ref["th_v"],
        "rho": bg_ref["rho"],
        "pi": bg_ref["pi"],
        "q_tr": jnp.zeros_like(state_in.get("q_tr", jnp.zeros_like(state_in["rho"]))),
    }
    return x_sponge.blend(state_in, ext_state)


def create_state_with_tracer(x_km, z_km, amplitude):
    X_km = grid.x_m[:, None, None] / 1000.0
    Z_km = grid.Z_m / 1000.0

    sigma_x, sigma_z = 2.0, 1.0
    q_tr = amplitude * jnp.exp(-((X_km - x_km) ** 2 / (2 * sigma_x**2) + (Z_km - z_km) ** 2 / (2 * sigma_z**2)))

    return {
        "u": jnp.ones((nx + 1, ny, nz)) * u_bg,
        "v": jnp.zeros((nx, ny + 1, nz)),
        "w": jnp.zeros((nx, ny, nz + 1)),
        "pi": bg_ref["pi"],
        "eta_dot": jnp.zeros((nx, ny, nz + 1)),
        "rho": bg_ref["rho"],
        "th_v": bg_ref["th_v"],
        "q_tr": q_tr,
    }


# --- 3. GENERATE "TRUE" TARGET DATA ---
print("[SIMULATION] Generating true target observations...")
true_params = {"x": -15.0, "z": 3.5, "A": 10.0}
true_state = create_state_with_tracer(true_params["x"], true_params["z"], true_params["A"])

# Configure dynamic core properties
if CORE_TYPE.lower() == "sisl":
    core_kwargs = {
        "dt": dt,
        "nu_div_factor": 0.0,
        "nu_h_factor": 0.0,
        "damp_height": 8000.0,
        "max_damp": 0.5,
        "N_bv": 0.01,
    }
elif CORE_TYPE.lower() == "split-explicit":
    core_kwargs = {
        "dt": dt,
        "ns": 6,
        "nu_div_factor": 0.0,
        "nu_h_factor": 0.0,
        "damp_height": 8000.0,
        "max_damp": 0.5,
        "N_bv": 0.01,
    }

# Build forward stepper
stepper_fwd, _ = build_dynamical_core(
    core_type=CORE_TYPE,
    grid=grid,
    operators=op,
    constants=constants,
    initial_state=true_state,
    physics_suite=suite,
    **core_kwargs,
)
if hasattr(stepper_fwd, "use_checkpointing"):
    stepper_fwd.use_checkpointing = False


@jax.jit
def generate_target_data(init_state):
    def fast_forward(s, _):
        next_s = stepper_fwd.step(s, 0.0, None, bc_fn)
        return next_s, next_s["q_tr"][:, 1, :]

    return jax.lax.scan(fast_forward, init_state, jnp.arange(num_steps))


true_final_state, q_tr_history = generate_target_data(true_state)

sensor_idx_x = int((10.0 - (grid.x_m[0] / 1000.0)) / (dx / 1000.0))
target_sensor_profile = true_final_state["q_tr"][sensor_idx_x, 1, :]

snapshot_indices = [num_steps // 4, num_steps // 2, 3 * num_steps // 4, num_steps - 1]
snapshots = [true_state["q_tr"][:, 1, :]] + [q_tr_history[i] for i in snapshot_indices]
times_mins = [0.0] + [(i * dt) / 60.0 for i in snapshot_indices]


# --- 4. THE INVERSE PROBLEM ---
# Build adjoint stepper
stepper_adj, _ = build_dynamical_core(
    core_type=CORE_TYPE,
    grid=grid,
    operators=op,
    constants=constants,
    initial_state=true_state,
    physics_suite=suite,
    **core_kwargs,
)
if hasattr(stepper_adj, "use_checkpointing"):
    stepper_adj.use_checkpointing = True

sim_adj = Simulation(step_fn=stepper_adj.step, dt=dt)


def objective_fn(params):
    x_km, z_km, A = params[0], params[1], params[2]
    state = create_state_with_tracer(x_km, z_km, A)

    final_state = sim_adj.run_differentiable(state, 0.0, t_end, bc_fn=bc_fn, chunk_steps=40)

    simulated_sensor_profile = final_state["q_tr"][sensor_idx_x, 1, :]
    mse_loss = jnp.mean((simulated_sensor_profile - target_sensor_profile) ** 2)
    return mse_loss, final_state


# Setup Solver
guess_params = jnp.array([-5.0, 7.0, 2.0])
total_opt_steps = 60
lr_schedule = optax.cosine_decay_schedule(init_value=0.5, decay_steps=total_opt_steps, alpha=0.02)
optimizer = optax.adam(learning_rate=lr_schedule)


def constrain_bounds(p):
    p = p.at[1].set(jnp.maximum(p[1], 0.1))
    p = p.at[2].set(jnp.maximum(p[2], 0.0))
    return p


solver = OptaxSolver(objective_fn, optimizer, has_aux=True)

# This single line handles the entire forward-backward gradient descent loop!
optimal_params, history = solver.fit(guess_params, total_steps=total_opt_steps, bounds_fn=constrain_bounds)


# --- 5. VISUALIZATION ---
print("\n[PLOT] Generating visualizations...")

x_plot_1d = grid.x_m / 1000.0
Z_plot = grid.Z_m[:, 1, :] / 1000.0
X_plot, _ = jnp.meshgrid(x_plot_1d, jnp.arange(nz), indexing="ij")
terrain_plot = terrain_profile(grid.x_m, 0) / 1000.0

# 5a. Plot Forward Snapshots
fig1, axs1 = plt.subplots(5, 1, figsize=(12, 18), sharex=True)
fig1.suptitle("Tracer plume evolution over topography", fontsize=16)

for idx, ax in enumerate(axs1):
    levels = jnp.linspace(0.01, 10.0, 100)
    contour = ax.contourf(X_plot, Z_plot, snapshots[idx], levels=levels, cmap="Blues", extend="max")
    ax.set_facecolor("white")
    ax.fill_between(x_plot_1d, 0, terrain_plot, color="black")
    ax.axvline(x=10.0, color="blue", linestyle="--", linewidth=1.5)
    ax.set_title(f"T = {times_mins[idx]:.1f} mins")
    ax.set_ylabel("Altitude (km)")
    ax.set_ylim([0, 10])

axs1[-1].set_xlabel("Distance (km)")
plt.tight_layout()
if not args.no_render:
    plt.savefig(figure_dir / "tracer_snapshots.png", dpi=150)
plt.close(fig1)

# 5b. Plot Optimization Trajectory
fig2, axs2 = plt.subplots(2, 1, figsize=(12, 10))

# Trajectory Plot
axs2[0].contourf(X_plot, Z_plot, true_state["q_tr"][:, 1, :], levels=20, cmap="Greys", alpha=0.3)
axs2[0].fill_between(x_plot_1d, 0, terrain_plot, color="black", alpha=0.7)
axs2[0].axvline(x=10.0, color="blue", linestyle="--", linewidth=2, label="Sensor array")

# Extract history correctly from the solver's output dictionary
hx = [p[0] for p in history["params"]]
hz = [p[1] for p in history["params"]]
axs2[0].plot(hx, hz, marker="o", color="purple", linestyle="-", linewidth=2, markersize=5, label="Optimizer trajectory")
axs2[0].scatter([hx[0]], [hz[0]], color="orange", s=100, label="Initial guess")
axs2[0].scatter([hx[-1]], [hz[-1]], color="blue", s=100, zorder=5, label="Inferred source location")
axs2[0].scatter(
    [true_params["x"]], [true_params["z"]], color="red", marker="x", s=100, zorder=6, label="True source location"
)

axs2[0].set_title("Source location optimization")
axs2[0].set_xlim([-30, 30])
axs2[0].set_ylim([0, 10])
axs2[0].set_ylabel("Altitude (km)")
axs2[0].legend()

# Sensor Profile Plot (Using the optimal_params found by the solver)
final_guessed_state = create_state_with_tracer(optimal_params[0], optimal_params[1], optimal_params[2])


# Fast forward to get the final recovered state
@jax.jit
def fast_fwd_eval(s):
    def scan_fwd(state, _):
        return stepper_fwd.step(state, 0.0, None, bc_fn), None

    return jax.lax.scan(scan_fwd, s, jnp.arange(num_steps))[0]


final_sim = fast_fwd_eval(final_guessed_state)
recovered_sensor_profile = final_sim["q_tr"][sensor_idx_x, 1, :]
artifact = xr.Dataset(
    data_vars={
        "tracer_snapshot": (("time", "x", "z"), np.asarray(snapshots)),
        "terrain_height": ("x", np.asarray(terrain_plot) * 1000.0),
        "physical_height": (("x", "z"), np.asarray(grid.Z_m[:, 1, :])),
        "optimization_parameters": (("optimization_step", "parameter"), np.asarray(history["params"])),
        "target_sensor_profile": ("z", np.asarray(target_sensor_profile)),
        "recovered_sensor_profile": ("z", np.asarray(recovered_sensor_profile)),
    },
    coords={
        "time": np.asarray(times_mins) * 60.0,
        "x": np.asarray(grid.x_m),
        "z": np.arange(nz),
        "optimization_step": np.arange(len(history["params"])),
        "parameter": ["x", "z", "amplitude"],
    },
    attrs={"core": CORE_TYPE, "dt_s": dt, "t_end_s": t_end},
)
artifact_path = save_plot_dataset(artifact, data_dir / "artifact.nc", experiment="tracer_inversion_3d")
print(f"[OUTPUT] Saved {artifact_path}")
if args.no_render:
    plt.close(fig2)
    raise SystemExit(0)

axs2[1].plot(
    target_sensor_profile, grid.Z_m[sensor_idx_x, 1, :] / 1000.0, "k-", linewidth=3, label="True target profile"
)
axs2[1].plot(
    recovered_sensor_profile, grid.Z_m[sensor_idx_x, 1, :] / 1000.0, "r--", linewidth=2, label="Recovered profile"
)
axs2[1].set_title("Vertical tracer concentration at sensor array (T=40 mins)")
axs2[1].set_xlabel("Tracer concentration")
axs2[1].set_ylabel("Altitude (km)")
axs2[1].legend()

plt.tight_layout()
plt.savefig(figure_dir / "tracer_source_inversion.png", dpi=150)
print(
    f"Done! Check tracer_snapshots_{CORE_TYPE.lower()}.png and tracer_source_inversion_terrain_{CORE_TYPE.lower()}.png"
)
