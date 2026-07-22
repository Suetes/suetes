"""
Chunked, disk-backed boundary/initial-condition store (Zarr).

Replaces the monolithic pickle path (``BoundaryProcessor.build_or_load_timeseries``
+ ``TimeManager``) for runs that need to scale to long forecast windows.


The pickle path builds a Python list of *all* boundary states, then
``TimeManager`` stacks the whole timeseries into host RAM as ``(time,X,Y,Z)``
arrays. The per-step GPU footprint is already minimal (only the two bounding
states are streamed to device), but the host stack grows linearly with the
number of states and dominates memory at long lead times (and the runner keeps
two such stacks: a coarse driver + a native reference).

This module instead:
  * writes each state to a Zarr group as it is built (one state in RAM at a
    time -- low preprocessing RAM, no 7 GB pickle), chunked per-timestep so a
    single time-slice is one chunk read; and
  * streams only the two bounding time-slices from disk per ``get_forcing`` call
    (``LazyZarrTimeManager``), with a tiny cache so the many sub-steps inside one
    forcing hour do not re-read the same chunk.

The existing ``era2suetes.TimeManager`` / ``build_or_load_timeseries`` are left
untouched so the other runners keep working.

I think there's a lot more work to be done here to have a consistent and efficient loader...
"""

import os
import shutil

import numpy as np
import jax
import jax.numpy as jnp
import zarr

_DONE_ATTR = "complete"


def _canonical_float_dtype():
    """JAX's current default float dtype (float32, or float64 if x64 is on).

    Used so a store written under one precision can be read under another:
    every float field is cast to this on read, matching the run's precision.
    """
    return np.dtype(jnp.zeros(1, dtype=float).dtype)


def _store_is_complete(store_path):
    if not os.path.exists(store_path):
        return False
    try:
        g = zarr.open(store_path, mode="r")
        return bool(g.attrs.get(_DONE_ATTR, False))
    except Exception:
        return False


def write_timeseries_zarr(store_path, era5_proc, bridge, num_states, times_sec,
                          coarsen_window=None, static=None):
    """Build boundary states one-at-a-time and write them to a chunked Zarr store.

    Mirrors ``BoundaryProcessor.build_or_load_timeseries`` (regrid each ERA5
    timestep via ``era5_proc.get_stitched_state`` -> ``bridge.process``) but
    streams each state straight to disk instead of accumulating a list.

    Args:
        store_path (str): Output ``.zarr`` directory path.
        era5_proc (ERA5Processor): Raw ERA5 reader.
        bridge (BoundaryProcessor): Regridding / thermodynamic transform.
        num_states (int): Number of hourly states to build.
        times_sec (sequence[float]): Time coordinate (seconds), length num_states.
        coarsen_window (int or None): Spatial coarsening passed to get_stitched_state.

    Returns:
        str: ``store_path`` 
    """
    if _store_is_complete(store_path):
        print(f"[BC-ZARR] Using cached store {store_path}")
        return store_path
    # A partial/aborted store would be inconsistent -- rebuild from scratch.
    if os.path.exists(store_path):
        shutil.rmtree(store_path)

    print(f"[BC-ZARR] Building store (N={num_states}) -> {store_path}")
    root = zarr.open(store_path, mode="w")
    arrays = {}
    keys = None
    for i in range(num_states):
        if i % 6 == 0:
            print(f"[BC-ZARR] -> regridding state for T={i}h")
        raw_state = era5_proc.get_stitched_state(time_idx=i, coarsen_window=coarsen_window)
        bc_state = bridge.process(raw_state)
        # Pull to host (one state) and drop JAX tracers.
        bc_state = {k: np.asarray(v) for k, v in bc_state.items()}

        if keys is None:
            keys = sorted(bc_state.keys())
            for k in keys:
                a = bc_state[k]
                # chunk=(1, *spatial): one time-slice == one chunk read.
                arrays[k] = root.create_array(
                    k, shape=(num_states, *a.shape),
                    chunks=(1, *a.shape), dtype=a.dtype)
        for k in keys:
            arrays[k][i] = bc_state[k]

    # Time-invariant static inputs the SIMULATION needs (so the runner never
    # touches ERA5/GEBCO): topography 'h' and 'land_fraction'.
    static_keys = []
    if static:
        for k, v in static.items():
            a = np.asarray(v)
            root.create_array(f"static_{k}", shape=a.shape, chunks=a.shape, dtype=a.dtype)[:] = a
            static_keys.append(k)

    root.attrs["times_sec"] = [float(t) for t in times_sec]
    root.attrs["keys"] = keys
    root.attrs["static_keys"] = static_keys
    root.attrs["num_states"] = int(num_states)
    root.attrs[_DONE_ATTR] = True
    print(f"[BC-ZARR] Wrote {num_states} states ({len(keys)} fields"
          f"{', static: ' + ','.join(static_keys) if static_keys else ''}) to {store_path}")
    return store_path


def read_static(store_path):
    """Read the time-invariant static fields (topography, land_fraction) from a store."""
    root = zarr.open(store_path, mode="r")
    return {k: np.asarray(root[f"static_{k}"]) for k in root.attrs.get("static_keys", [])}


def read_state(store_path, idx):
    """Read a single boundary state (all fields) at integer ``idx`` (supports -1).

    Returns a dict of JAX arrays cast to the run's precision. Reads only that
    time-slice from disk -- never the full timeseries.
    """
    root = zarr.open(store_path, mode="r")
    keys = list(root.attrs["keys"])
    n = int(root.attrs["num_states"])
    i = int(idx) % n
    return {k: jnp.asarray(np.asarray(root[k][i])) for k in keys}


def read_columns(store_path, indices, ij_list, keys=("th_v", "pi"), z=0):
    """Read point columns over time without materializing the full timeseries.

    For each time index in ``indices``, reads one slice per key (freed before the
    next), extracting the value at every ``(i, j, z)`` in ``ij_list``.

    Returns:
        dict[str, np.ndarray]: key -> array of shape (len(indices), len(ij_list)).
    """
    root = zarr.open(store_path, mode="r")
    out = {k: np.zeros((len(indices), len(ij_list))) for k in keys}
    for n, idx in enumerate(indices):
        for k in keys:
            sl = np.asarray(root[k][int(idx)])
            for m, (i, j) in enumerate(ij_list):
                out[k][n, m] = sl[i, j, z]
    return out


class LazyZarrTimeManager:
    """Streaming drop-in for ``era2suetes.TimeManager`` backed by a Zarr store.

    Public surface matches ``TimeManager``: construct with
    ``(store_path, times_sec_list, grid)`` and call ``get_forcing(t)`` to get the
    linearly-interpolated state at time ``t``. Only the two bounding time-slices
    are ever resident on the host (plus a tiny LRU-ish cache), so host memory is
    O(1) in the number of states rather than O(n_times).
    """

    def __init__(self, store_path, times_sec_list, grid, cache_size=3):
        self.grid = grid
        self.store_path = store_path
        self._root = zarr.open(store_path, mode="r")  # lazy -- no data read
        self.keys = list(self._root.attrs["keys"])
        self.n_times = int(self._root.attrs["num_states"])

        self.times_sec_np = np.asarray(times_sec_list, dtype=np.float32)
        self.times_sec_jax = jnp.asarray(self.times_sec_np)

        # Read in the run's precision so a float64 store works in a float32 run.
        self._fdtype = _canonical_float_dtype()
        self._slice_shape_dtypes = {
            k: jax.ShapeDtypeStruct(self._root[k].shape[1:], self._fdtype)
            for k in self.keys
        }
        self._return_shape_dtypes = (self._slice_shape_dtypes,
                                     self._slice_shape_dtypes)
        self._cache = {}            # idx -> dict of host slices
        self._cache_size = cache_size

    def _slice(self, i):
        s = self._cache.get(i)
        if s is None:
            s = {k: np.asarray(self._root[k][i], dtype=self._fdtype) for k in self.keys}
            if len(self._cache) >= self._cache_size:
                self._cache.pop(next(iter(self._cache)))  # evict oldest
            self._cache[i] = s
        return s

    def _fetch_two_states_host(self, idx_arr):
        i = int(idx_arr)
        i_next = min(i + 1, self.n_times - 1)
        return self._slice(i), self._slice(i_next)

    def bounding_pair(self, t):
        """Host-side bounding pair for time ``t`` -- call ONCE per chunk, interpolate
        on-device downstream. Returns ``(s0, s1, t0, t1)`` with s0/s1 host numpy dicts
        (cached). Avoids the per-step ``pure_callback`` in get_forcing the
        bounding pair is constant across an hour, so fetching it per hour instead of
        per step removes ~all the host<->device sync that caps GPU utilization."""
        i = int(np.searchsorted(self.times_sec_np, np.float32(t), side="right") - 1)
        i = int(np.clip(i, 0, self.n_times - 2))
        s0, s1 = self._fetch_two_states_host(i)
        return s0, s1, float(self.times_sec_np[i]), float(self.times_sec_np[i + 1])

    def get_forcing(self, t):
        idx = jnp.searchsorted(self.times_sec_jax, t, side="right") - 1
        idx = jnp.clip(idx, 0, self.n_times - 2)

        state_t0, state_t1 = jax.pure_callback(
            self._fetch_two_states_host,
            self._return_shape_dtypes,
            idx,
        )

        t0 = self.times_sec_jax[idx]
        t1 = self.times_sec_jax[idx + 1]
        alpha = jnp.clip((t - t0) / (t1 - t0), 0.0, 1.0)
        return {
            k: (1.0 - alpha) * state_t0[k] + alpha * state_t1[k]
            for k in state_t0.keys()
        }
