"""Tests for throttled_sample ops."""
import unittest
from functools import partial

import jax
import jax.numpy as jnp
import jax.test_util as jtu
import numpy as np
from absl.testing import parameterized
from jax_stsc_ops.cumulative_ema import (
    associative_scan_cumulative_ema,
    associative_scan_segment_cumulative_ema,
    cumulative_ema,
    segment_cumulative_ema,
)

jax.config.update("jax_enable_x64", True)


def get_random_args(size: int, num_segments: int, seed: int, dtype=np.float32):
    seeds = jax.random.split(jax.random.PRNGKey(seed), 3)
    values = jax.random.uniform(seeds[0], (size,))
    factors = jax.random.uniform(seeds[1], (size,))
    segment_ids = jax.random.uniform(seeds[2], (size,), maxval=num_segments).astype(
        jnp.int32
    )
    segment_ids = jnp.sort(segment_ids)
    return values.astype(dtype), factors.astype(dtype), segment_ids


class CumulativeSegmentEmaTest(parameterized.TestCase):
    @parameterized.product(
        # reverse=(False, True),
        # jit=(False, True),
        backend=("cpu", "gpu"),
        dtype=(jnp.float32, jnp.float64, jnp.complex64, jnp.complex128),
    )
    def test_cumulative_ema(
        self,
        reverse: bool = False,
        jit: bool = False,
        backend: str = "cpu",
        size: int = 11,
        dtype=jnp.float32,
        seed: int = 0,
    ):
        func = partial(cumulative_ema, reverse=reverse)
        if jit:
            func = jax.jit(func)
        with jax.default_device(jax.devices(backend)[0]):
            seeds = jax.random.split(jax.random.PRNGKey(seed), 2)
            if "complex" in str(dtype):
                real_dtype = jnp.float32 if dtype == jnp.complex64 else jnp.float64
                values = jax.random.uniform(seeds[0], (2, size), dtype=real_dtype)
                values = values[0] + values[1] * 1j
                factors = jax.random.uniform(seeds[1], (2, size), dtype=real_dtype)
                factors = factors[0] + factors[1] * 1j
            else:
                values = jax.random.uniform(seeds[0], (size,), dtype=dtype)
                factors = jax.random.uniform(seeds[1], (size,), dtype=dtype)
            actual = func(values, factors)
            expected = associative_scan_cumulative_ema(values, factors, reverse=reverse)
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    @parameterized.product(
        reverse=(False, True),
        jit=(False, True),
        backend=("cpu", "gpu"),
    )
    def test_segment_cumulative_ema(
        self,
        reverse: bool = False,
        jit: bool = False,
        backend: str = "cpu",
        size: int = 11,
        num_segments: int = 5,
        seed: int = 0,
    ):
        args = get_random_args(size, num_segments, seed, dtype=np.float32)
        kwargs = {"reverse": reverse}
        func = partial(segment_cumulative_ema, **kwargs)
        if jit:
            func = jax.jit(func)
        with jax.default_device(jax.devices(backend)[0]):
            actual = func(*args)
            expected = associative_scan_segment_cumulative_ema(*args, **kwargs)
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    @parameterized.product(reverse=(False, True))
    def test_segment_cumulative_ema_vjp(
        self,
        reverse: bool = False,
        size: int = 11,
        num_segments: int = 5,
        seed: int = 0,
    ):
        values, factors, segment_ids = get_random_args(size, num_segments, seed)
        func = partial(segment_cumulative_ema, segment_ids=segment_ids, reverse=reverse)
        args = values, factors
        # jtu.check_grads(func, args, 1, modes=("fwd",))
        jtu.check_grads(func, args, 1, modes=("rev",))

        primals_out, f = jax.vjp(func, *args)
        cotangent = jax.random.normal(
            jax.random.PRNGKey(seed + 1), primals_out.shape, primals_out.dtype
        )
        grad_values_actual, grad_factors_actual = f(cotangent)

        expected_func = partial(
            associative_scan_segment_cumulative_ema,
            segment_ids=segment_ids,
            reverse=reverse,
        )
        primals_out_expected, f_expected = jax.vjp(expected_func, *args)
        np.testing.assert_allclose(primals_out, primals_out_expected, rtol=1e-6)

        grad_values_expected, grad_factors_expected = f_expected(cotangent)
        np.testing.assert_allclose(grad_values_actual, grad_values_expected, rtol=1e-6)
        np.testing.assert_allclose(
            grad_factors_actual, grad_factors_expected, rtol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
