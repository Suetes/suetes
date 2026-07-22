import jax
import jax.numpy as jnp

from suetes.shared.driver import Simulation


def test_differentiable_run_uses_absolute_time_and_remainder():
    observed_times = []

    def step(state, time_seconds, forcing=None, bc_fn=None):
        del forcing, bc_fn
        jax.debug.callback(observed_times.append, time_seconds, ordered=True)
        return state + time_seconds

    result = Simulation(step, dt=2.0).run_differentiable(
        jnp.asarray(0.0), t_start=10.0, t_end=16.0, chunk_steps=2
    )
    result.block_until_ready()

    assert float(result) == 10.0 + 12.0 + 14.0
    assert [float(value) for value in observed_times] == [10.0, 12.0, 14.0]


def test_run_compilation_does_not_duplicate_callbacks():
    observed_steps = []

    def step(state, step_index):
        jax.debug.callback(observed_steps.append, step_index, ordered=True)
        return state + 1, state

    result = Simulation(step, dt=1.0).run(
        jnp.asarray(0), t_start=0.0, t_end=3.0, chunk_steps=2
    )
    result.block_until_ready()

    assert int(result) == 3
    assert [int(value) for value in observed_steps] == [0, 1, 2]
