from functools import partial

import google_benchmark as bm
import jax
import jax.numpy as jnp
import numpy as np
from absl import flags
from jax_stsc_ops import conv_preprocessing as cp_ops

flags.DEFINE_integer("seed", 0, "random seed")
flags.DEFINE_integer("n", 1024**2, "number of events")
flags.DEFINE_integer("g", 32**2, "grid size")
flags.DEFINE_integer("b", 32, "batch size")
flags.DEFINE_integer("k", 25, "kernel_size")

FLAGS = flags.FLAGS


def ids_to_splits_np(ids: np.ndarray) -> np.ndarray:
    return np.pad(np.cumsum(np.bincount(ids)), [[1, 0]])


def inverse_perm(permutation: np.ndarray, dtype=None) -> np.ndarray:
    assert len(permutation.shape) == 1
    if dtype is None:
        dtype = permutation.dtype

    (n,) = permutation.shape
    result = np.zeros((n,), dtype=dtype)
    result[permutation] = np.arange(n, dtype=dtype)
    return result


def get_args(rng: np.random.Generator):
    num_events = FLAGS.n
    grid_size = FLAGS.g
    batch_size = FLAGS.b
    kernel_size = FLAGS.k
    batch_ids = (rng.uniform(size=num_events) * batch_size).astype("int32")
    batch_ids.sort()
    batch_splits = ids_to_splits_np(batch_ids)
    times = rng.uniform(size=num_events).astype(np.float32)
    times.sort()
    pixel_ids = (rng.uniform(size=num_events) * grid_size).astype("int32")
    perm_in = rng.permutation(num_events).astype("int32")
    perm_out = rng.permutation(num_events).astype("int32")

    pad_left = (kernel_size - 1) // 2
    pixel_ids += pad_left
    kernel_offsets = np.arange(-pad_left, -pad_left + kernel_size)

    grid_size = grid_size + kernel_size - 1

    return (
        pixel_ids,
        times,
        batch_splits,
        kernel_offsets,
        perm_in,
        perm_out,
        grid_size,
        num_events,
    )


def run_benchmark(state, func, args, **static_kwargs):
    func = jax.jit(partial(func, **static_kwargs))
    args = tuple(jnp.asarray(x) for x in args)
    jax.block_until_ready(func(*args))
    while state:
        jax.block_until_ready(func(*args))


@bm.register
def get_stationary_predecessor_ids(state):
    with jax.default_device(jax.devices("cpu")[0]):
        rng = np.random.default_rng(FLAGS.seed)
        (
            pixel_ids,
            times,
            batch_splits,
            kernel_offsets,
            perm_in,
            perm_out,
            grid_size,
            num_events,
        ) = get_args(rng)

        inv_perm_in = jnp.pad(
            inverse_perm(perm_in), [[0, 1]], constant_values=num_events
        )

        def func(
            pixel_ids, batch_splits, kernel_offsets, inv_perm_in, perm_out, *, grid_size
        ):
            x = cp_ops.get_stationary_predecessor_ids(
                pixel_ids, batch_splits, kernel_offsets, grid_size=grid_size
            )
            x = jnp.take(inv_perm_in, x, axis=0)
            x = jnp.take(x, perm_out)
            return x

        run_benchmark(
            state,
            func,
            (pixel_ids, batch_splits, kernel_offsets, inv_perm_in, perm_out),
            grid_size=grid_size,
        )


@bm.register
def get_permuted_stationary_predecessor_ids(state):
    with jax.default_device(jax.devices("cpu")[0]):
        rng = np.random.default_rng(FLAGS.seed)
        (
            pixel_ids,
            times,
            batch_splits,
            kernel_offsets,
            perm_in,
            perm_out,
            grid_size,
            num_events,
        ) = get_args(rng)

        inv_perm_in = inverse_perm(perm_in)
        inv_perm_out = inverse_perm(perm_out)

        def func(
            pixel_ids,
            batch_splits,
            kernel_offsets,
            inv_perm_in,
            inv_perm_out,
            *,
            grid_size,
        ):
            return cp_ops.get_permuted_stationary_predecessor_ids(
                pixel_ids,
                batch_splits,
                kernel_offsets,
                inv_perm_in,
                inv_perm_out,
                grid_size=grid_size,
            )

        run_benchmark(
            state,
            func,
            (pixel_ids, batch_splits, kernel_offsets, inv_perm_in, inv_perm_out),
            grid_size=grid_size,
        )


if __name__ == "__main__":
    bm.main()
