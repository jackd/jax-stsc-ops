"""Tests for counting_argsort ops."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from jax_stsc_ops import counting_argsort


class CountingArgsortTest(parameterized.TestCase):
    @parameterized.product(jit=(False, True))
    def test_counting_argsort(self, jit: bool):
        segment_ids = [0, 2, 1, 1, 0, 2, 1]
        lengths = np.bincount(segment_ids)
        splits = np.pad(np.cumsum(lengths), [[1, 0]])
        expected = np.argsort(segment_ids)

        func = counting_argsort
        if jit:
            func = jax.jit(func)

        with jax.default_device(jax.devices("cpu")[0]):
            segment_ids = jnp.asarray(segment_ids)
            splits = jnp.asarray(splits)
            actual = func(segment_ids, splits)
        np.testing.assert_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
