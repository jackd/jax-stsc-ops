import functools

import google_benchmark as bm
import jax
import jax.numpy as jnp
from absl import flags
from jax_stsc_ops import cumulative_ema as cema

flags.DEFINE_bool("grad", False, help="Calculate gradients")
flags.DEFINE_bool("r", False, help="Reverse")
flags.DEFINE_integer("n", 2**19, help="Number of elements")
flags.DEFINE_integer("s", 32**3, help="Number of segments")
flags.DEFINE_integer("c", 32, help="Number of channels")
flags.DEFINE_integer("seed", 0, help="random seed")

FLAGS = flags.FLAGS


def run_benchmark(state, func, from_splits=False):
    seeds = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 3)
    values = jax.random.uniform(seeds[0], (FLAGS.n, FLAGS.c))
    factors = jax.random.uniform(seeds[1], (FLAGS.n, FLAGS.c))
    segment_ids = jax.random.uniform(seeds[2], (FLAGS.n,), maxval=FLAGS.s).astype(
        jnp.int32
    )
    if from_splits:
        lengths = jnp.bincount(segment_ids, length=FLAGS.s).astype(jnp.int32)
        splits = jnp.pad(jnp.cumsum(lengths), [[1, 0]])
        args = (values, factors, splits)
    else:
        args = (values, factors, jnp.sort(segment_ids))
    func = functools.partial(func, reverse=FLAGS.r)
    if FLAGS.grad:
        base_func = func
        func = jax.grad(lambda *args: jnp.sum(base_func(*args) ** 2), argnums=(0, 1))

    func = jax.jit(func)
    jax.block_until_ready(func(*args))
    while state:
        jax.block_until_ready(func(*args))


@bm.register
def serial_segment_cumulative_ema(state):
    return run_benchmark(state, cema.serial_segment_cumulative_ema, from_splits=True)


@bm.register
def tiled_segment_cumulative_ema(state):
    return run_benchmark(state, cema.tiled_segment_cumulative_ema)


@bm.register
def associative_scan_segment_cumulative_ema(state):
    def func(values, factors, segment_ids, reverse: bool):
        return cema.associative_scan_segment_cumulative_ema(
            values,
            factors,
            segment_ids=jnp.expand_dims(segment_ids, axis=-1),
            reverse=reverse,
            axis=0,
        )

    return run_benchmark(state, func)


@bm.register
def segment_cumulative_ema(state):
    def func(values, factors, segment_ids, reverse: bool):
        c = values.shape[1]
        segment_ids = jnp.expand_dims(segment_ids, -1) * c + jnp.arange(c)
        out = cema.segment_cumulative_ema(
            values.T.flatten(),
            factors.T.flatten(),
            segment_ids.T.flatten(),
            reverse=reverse,
        )
        return out.T

    return run_benchmark(state, func)


@bm.register
def tiled_segment_cumulative_ema_basic(state):
    return run_benchmark(state, cema.tiled_segment_cumulative_ema_basic)


@bm.register
def tiled_segment_cumulative_ema_basic_from_splits(state):
    return run_benchmark(
        state, cema.tiled_segment_cumulative_ema_basic_from_splits, from_splits=True
    )


if __name__ == "__main__":
    bm.main()
