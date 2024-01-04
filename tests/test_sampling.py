"""Tests for throttled_sample ops."""
import unittest
from functools import partial

import jax
import numpy as np
from absl.testing import parameterized
from jax_stsc_ops.sampling import throttled_sample


class CountingArgsortTest(parameterized.TestCase):
    @parameterized.product(jit=(False, True))
    def test_throttled_sample(self, jit: bool):
        pixel_ids = [0, 2, 1, 1, 0, 2, 1, 1, 1]
        times = np.array([0, 1, 2, 2, 2, 2, 2, 2, 5], np.float32)
        batch_splits = [0, 9]
        min_dt = 0.5
        sample_rate = 2
        grid_size = 3

        expected = np.array([3, 4, 5, 8])
        func = partial(
            throttled_sample,
            min_dt=min_dt,
            sample_rate=sample_rate,
            grid_size=grid_size,
        )
        if jit:
            func = jax.jit(func)
        with jax.default_device(jax.devices("cpu")[0]):
            actual, _ = func(pixel_ids, times, batch_splits)
        np.testing.assert_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
