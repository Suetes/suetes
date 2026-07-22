import time
import jax
import jax.numpy as jnp

class Simulation:
    """
    Manages the time integration loop, JIT compilation, and chunking.
    """
    def __init__(self, step_fn, dt):
        """
        Args:
            step_fn (callable): A function `fn(state, step_idx)` that returns 
                                `(next_state, metrics_array)`.
            dt (float): Timestep in seconds.
        """
        self.step_fn = step_fn
        self.dt = dt

    def run(self, initial_state, t_start, t_end, chunk_steps=120):
        total_steps = int((t_end - t_start) / self.dt)
        n_chunks = total_steps // chunk_steps
        remainder = total_steps % chunk_steps
        
        print(f"[SIMULATION] Starting: T={t_start} -> T={t_end}")
        print(f"[SIMULATION] Total Steps: {total_steps}")
        print(f"[SIMULATION] Chunk Size:  {chunk_steps} steps")
        
        def run_chunk_impl(curr_state, start_step, n_steps):
            def scan_fn(state, step_offset):
                step_idx = start_step + step_offset
                next_state, metrics = self.step_fn(state, step_idx)
                return next_state, metrics
            
            return jax.lax.scan(scan_fn, curr_state, jnp.arange(n_steps))

        # Construct the wrapper explicitly.  This remains stable when benchmark processes clear JAX compilation caches between independent runs.
        run_chunk = jax.jit(
            run_chunk_impl, static_argnames=('n_steps',)
        )

        print("[SIMULATION] Compiling kernel...")
        t0 = time.time()
        start_step = int(t_start / self.dt)
        compiled_full_chunk = None
        compiled_remainder = None
        # Compile without executing the kernel. Executing a discarded warm-up chunk duplicates host callbacks and other side effects. Retain the
        # executables so the timed integration cannot trigger recompilation.
        if n_chunks:
            compiled_full_chunk = run_chunk.lower(
                initial_state, start_step, chunk_steps
            ).compile()
        if remainder:
            compiled_remainder = run_chunk.lower(
                initial_state, start_step, remainder
            ).compile()
        print(f"[SIMULATION] Compilation finished in {time.time() - t0:.2f}s")
        print(f"[SIMULATION] JIT compiled, running simulation...")

        start_time = time.time()
        state = initial_state
        current_step = start_step
        
        for i in range(n_chunks):
            chunk_start = time.time()
            state, metrics = compiled_full_chunk(state, current_step)
            
            leaves = jax.tree_util.tree_leaves(state)
            if leaves:
                leaves[0].block_until_ready()
            
            current_step += chunk_steps
            t_curr = current_step * self.dt
            
            max_w = float(jnp.max(jnp.abs(metrics))) if metrics is not None else 0.0
            print(f"    Progress: {t_curr:.1f}s / {t_end:.1f}s | "
                  f"Max W: {max_w:.4f} m/s | "
                  f"Chunk Wall Time: {time.time() - chunk_start:.2f}s")

        if remainder > 0:
            print(f"    Finishing remaining {remainder} steps...")
            state, _ = compiled_remainder(state, current_step)
            leaves = jax.tree_util.tree_leaves(state)
            if leaves:
                leaves[0].block_until_ready()

        total_time = time.time() - start_time
        print(f"[Simulation] Done in {total_time:.2f}s\n")
        
        return state

    def run_differentiable(self, initial_state, t_start, t_end, bc_fn=None, forcing=None, chunk_steps=50):
        """
        A purely functional, JAX-traceable simulation loop. 
        Safe to use inside jax.grad() or jax.value_and_grad().
        """
        total_steps = int((t_end - t_start) / self.dt)
        num_chunks = total_steps // chunk_steps
        remainder = total_steps % chunk_steps
        start_step = int(t_start / self.dt)
        
        # Define the inner chunk scanner (with checkpointing for memory-efficient autodiff)
        @jax.checkpoint
        def scan_chunk(curr_state, chunk_idx):
            def inner_scan_fn(state, step_offset):
                step_idx = start_step + (chunk_idx * chunk_steps) + step_offset
                t_curr = step_idx * self.dt
                # Differentiable step functions use physical time, matching
                # the dynamical-core ``step(state, t, ...)`` interface.
                next_state = self.step_fn(state, t_curr, forcing=forcing, bc_fn=bc_fn)
                return next_state, None
                
            chunk_final_state, _ = jax.lax.scan(inner_scan_fn, curr_state, jnp.arange(chunk_steps))
            return chunk_final_state, None

        # Execute the outer chunk loop natively in JAX
        final_state, _ = jax.lax.scan(scan_chunk, initial_state, jnp.arange(num_chunks))

        if remainder:
            remainder_start = start_step + num_chunks * chunk_steps

            def remainder_step(state, step_offset):
                step_idx = remainder_start + step_offset
                next_state = self.step_fn(
                    state, step_idx * self.dt,
                    forcing=forcing, bc_fn=bc_fn,
                )
                return next_state, None

            final_state, _ = jax.lax.scan(
                remainder_step, final_state, jnp.arange(remainder)
            )
        return final_state
