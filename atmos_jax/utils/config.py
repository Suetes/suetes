import jax

def setup_jax(use_x64=False):
    """
    Configures JAX precision.
    Must be called BEFORE creating any arrays or grids.
    """
    if use_x64:
        jax.config.update("jax_enable_x64", True)
        print("[Config] Running in 64-bit mode (CPU/High-Precision)")
    else:
        jax.config.update("jax_enable_x64", False)
        print("[Config] Running in 32-bit mode (GPU/Fast)")