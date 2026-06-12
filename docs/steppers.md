# Time Integration and Advection

The `steppers` module manages the core temporal evolution of the 3D non-hydrostatic dynamical core. To accommodate different spatial scales and computational objectives, the module implements two fundamentally distinct numerical architectures: a **Semi-Implicit Semi-Lagrangian (SISL)** scheme and a **Split-Explicit (Eulerian) Runge-Kutta 3 (RK3)** scheme.

---

## Architecture Comparison

Choosing the right stepper depends heavily on the horizontal resolution, domain size, and the scale of the physical phenomena being simulated.

| Feature | Semi-Implicit Semi-Lagrangian (SISL) | Split-Explicit (Eulerian RK3) |
| :--- | :--- | :--- |
| **Suggested Scale** | Large-scale / Synoptic dynamics | Small-scale / High-resolution convective dynamics |
| **Advection Treatment** | Geometric trajectory tracing (Backward tracking) | Flux-form 3rd-order upwind biased Eulerian flux |
| **Acoustic/Gravity Waves**| Implicitly treated via 3D GMRES solver | Explicitly sub-cycled via fast acoustic steps ($\Delta \tau$) |
| **CFL Limitation** | Bypasses traditional Eulerian CFL limits; allows large $\Delta t$ | Restrained by explicit horizontal CFL; requires smaller $\Delta t$ |
| **Mass Conservation** | Uses Flux-Form SL (FFSL) for densities/tracers | Inherently conservative via Eulerian flux divergence |

---

## Dynamical Core Factory

The module exposes a unified factory function (`build_dynamical_core`) to safely instantiate the dynamical core physics configuration alongside its paired time-stepper. It applies safe, architecture-specific default stability parameters (such as raising the sponge layer height for explicit architectures to absorb explicit gravity waves) while accepting clean user overrides via `**kwargs`.

```python
from suetes.regional3d.steppers import build_dynamical_core

# For synoptic scales with large time steps
sisl_stepper, dt = build_dynamical_core(
    core_type="sisl", 
    grid=grid, operators=ops, constants=c, initial_state=init_state
)

# For fine-scale, cloud-resolving simulations
explicit_stepper, dt = build_dynamical_core(
    core_type="split-explicit", 
    grid=grid, operators=ops, constants=c, initial_state=init_state,
    ns=6  # Number of acoustic steps per RK3 stage
)
```

## Mathematical Formulations

### Semi-Implicit Semi-Lagrangian (SISL)

The SISL architecture targets large-scale flows where executing massive time steps outweighs the absolute localization of Eulerian grids. The core governing equation discretizes momentum and continuity implicitly:
$$ \frac{\Phi^{n+1} - \Phi_d}{\Delta t} + \alpha \mathcal{T}^{n+1} = (1-\alpha)\mathcal{T}^n,$$
where $\Phi_d$ is the prognostic state evaluated at the upstream departure point $\mathbf{x}_d$, $\mathcal{T}$ represents the linear wave tendencies, and $\alpha = 0.55$ provides a slight semi-implicit off-centering to damp high-frequency wave noise. Stiff acoustic modes are isolated into a vertical 1D Helmholtz equation that serves as an exact preconditioner $M^{-1}$ for a global 3D GMRES solver.

### Split-Explicit Eulerian (RK3)

The Split-Explicit architecture is optimized for fine-scale, high-resolution dynamics where capturing exact local gradients is essential. Slow advective and physical processes ($\mathcal{S}$) are isolated and integrated using a 3rd-order Runge-Kutta cycle:
$$ \Phi^{*} = \Phi^t + \frac{\Delta t}{3} \mathcal{S}(\Phi^t) $$

$$ \Phi^{**} = \Phi^t + \frac{\Delta t}{2} \mathcal{S}(\Phi^{*}) $$

$$ \Phi^{t+\Delta t} = \Phi^t + \Delta t \mathcal{S}(\Phi^{**}) $$.

To maintain computational efficiency without triggering acoustic CFL violations, high-frequency acoustic perturbation terms ($\text{Acoustic}(\Phi'', \Delta \tau)$) are advanced inside each RK3 stage using a significantly smaller, sub-cycled acoustic time step $\Delta \tau$. Vertical components are integrated implicitly within this sub-loop to bypass tight vertical grid constraints.

::: suetes.regional3d.steppers