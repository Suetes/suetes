# Diffusion and damping

The regional core distinguishes several stabilization mechanisms:

| Mechanism | Target |
|---|---|
| Fourth-order horizontal hyperdiffusion | Grid-scale variance |
| Divergence damping | High-frequency divergent motion |
| Rayleigh upper damping | Waves approaching the rigid lid |
| Optional physical turbulence | Unresolved turbulent transport |

`HyperFilter` and `DivergenceDamper` implement the numerical spatial filters.
The core scales their coefficients from explicit stability limits and user
factors. The upper sponge coefficient increases smoothly above its configured
start height.

These choices affect both forward solutions and gradients. Report their
coefficients in resolution studies, and do not interpret a more heavily damped
solution as intrinsically more converged.

Source: `suetes/regional3d/diffusion.py` and `suetes/regional3d/euler.py`.
