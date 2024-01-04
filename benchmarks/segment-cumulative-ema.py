import functools

import google_benchmark as bm
import jax
import jax.numpy as jnp
from absl import flags
from jax_stsc_ops import cumulative_ema as cema

flags.DEFINE_bool("grad", False, help="Calculate gradients")
flags.DEFINE_bool("r", False, help="Reverse")
flags.DEFINE_integer("n", 2**18, help="Number of elements")
flags.DEFINE_integer("s", 32**2, help="Number of segments")
flags.DEFINE_integer("seed", 0, help="random seed")

FLAGS = flags.FLAGS


def run_benchmark(
    state, func, from_splits: bool = False, expand_channels: bool = False
):
    seeds = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 3)
    values = jax.random.uniform(seeds[0], (FLAGS.n,))
    factors = jax.random.uniform(seeds[1], (FLAGS.n,))
    segment_ids = jax.random.uniform(seeds[2], (FLAGS.n,), maxval=FLAGS.s).astype(
        jnp.int32
    )
    if expand_channels:
        values = jnp.expand_dims(values, axis=-1)
        factors = jnp.expand_dims(factors, axis=-1)
    if from_splits:
        lengths = jnp.bincount(segment_ids, length=FLAGS.s)
        splits = jnp.pad(jnp.cumsum(lengths), [[1, 0]])
        args = (values, factors, splits)
    else:
        segment_ids = jnp.sort(segment_ids)
        args = (values, factors, segment_ids)

    func = functools.partial(func, reverse=FLAGS.r)
    if FLAGS.grad:
        base_func = func
        func = jax.grad(lambda *args: jnp.sum(base_func(*args) ** 2), argnums=(0, 1))

    func = jax.jit(func)
    jax.block_until_ready(func(*args))
    while state:
        jax.block_until_ready(func(*args))


@bm.register
def segment_cumulative_ema(state):
    return run_benchmark(state, cema.segment_cumulative_ema)


@bm.register
def segment_cumulative_ema_basic(state):
    return run_benchmark(state, cema.segment_cumulative_ema_basic)


@bm.register
def segment_cumulative_ema_basic_from_splits(state):
    return run_benchmark(
        state, cema.segment_cumulative_ema_basic_from_splits, from_splits=True
    )


@bm.register
def serial_segment_cumulative_ema(state):
    return run_benchmark(
        state,
        cema.serial_segment_cumulative_ema,
        from_splits=True,
        expand_channels=True,
    )


@bm.register
def associative_scan_segment_cumulative_ema(state):
    return run_benchmark(state, cema.associative_scan_segment_cumulative_ema)


@bm.register
def associative_scan_segment_cumulative_ema_v0(state):
    return run_benchmark(state, cema.associative_scan_segment_cumulative_ema_v0)


@bm.register
def associative_scan_segment_cumulative_ema_from_splits(state):
    return run_benchmark(
        state,
        cema.associative_scan_segment_cumulative_ema_from_splits,
        from_splits=True,
    )


if __name__ == "__main__":
    bm.main()
