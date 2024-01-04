import functools

import google_benchmark as bm
import jax
import jax.numpy as jnp
from absl import flags
from jax_stsc_ops import cumulative_ema as cema

flags.DEFINE_bool("grad", False, help="Calculate gradients")
flags.DEFINE_bool("r", False, help="Reverse")
flags.DEFINE_bool("complex", False, help="Complex values")
flags.DEFINE_integer("n", 2**18, help="Number of elements")
flags.DEFINE_integer("seed", 0, help="random seed")

FLAGS = flags.FLAGS


def run_benchmark(state, func):
    seeds = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 3)
    if FLAGS.complex:
        values = jax.random.uniform(seeds[0], (2, FLAGS.n))
        values = values[0] + values[1] * 1j
        factors = jax.random.uniform(seeds[1], (2, FLAGS.n))
        factors = factors[0] + factors[1] * 1j
    else:
        values = jax.random.uniform(seeds[0], (FLAGS.n,))
        factors = jax.random.uniform(seeds[1], (FLAGS.n,))

    args = (values, factors)

    func = functools.partial(func, reverse=FLAGS.r)
    if FLAGS.grad:
        base_func = func
        func = jax.grad(
            lambda *args: jnp.sum(jnp.real(base_func(*args)) ** 2), argnums=(0, 1)
        )

    func = jax.jit(func)
    jax.block_until_ready(func(*args))
    while state:
        jax.block_until_ready(func(*args))


@bm.register
def cumulative_ema(state):
    return run_benchmark(state, cema.cumulative_ema)


@bm.register
def associative_scan_cumulative_ema(state):
    return run_benchmark(state, cema.associative_scan_cumulative_ema)


if __name__ == "__main__":
    bm.main()
