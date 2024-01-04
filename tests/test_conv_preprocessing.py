"""Tests for throttled_sample ops."""
import unittest
from functools import partial

import jax
import jax.numpy as jnp
import jax_stsc_ops.conv_preprocessing as cp_ops
import numpy as np
from absl.testing import parameterized


def inverse_perm(permutation: jnp.ndarray, dtype=None) -> jnp.ndarray:
    assert len(permutation.shape) == 1
    if dtype is None:
        dtype = permutation.dtype

    return (
        jnp.zeros_like(permutation, dtype=dtype)
        .at[permutation]
        .set(jnp.arange(permutation.shape[0], dtype=dtype))
    )


def get_stationary_predecessor_ids_np(
    pixel_ids: np.ndarray,
    batch_splits: np.ndarray,
    kernel_offsets: np.ndarray,
    grid_size: int,
):
    (N,) = pixel_ids.shape
    (K,) = kernel_offsets.shape

    batch_size = batch_splits.shape[0] - 1
    last = np.empty((grid_size,), dtype=np.int32)
    predecessor_ids = np.full((N, K), N, dtype=np.int32)
    for b in range(batch_size):
        last[:] = N
        for i in range(batch_splits[b], batch_splits[b + 1]):
            pixel = pixel_ids[i]
            last[pixel] = i
            for ik, k in enumerate(kernel_offsets):
                predecessor_ids[i, ik] = last[pixel + k]
    return predecessor_ids


def get_successor_ids_np(
    pixel_ids_in: np.ndarray,
    times_in: np.ndarray,
    batch_splits_in: np.ndarray,
    pixel_ids_out: np.ndarray,
    times_out: np.ndarray,
    batch_splits_out: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    (n_in,) = pixel_ids_in.shape
    assert times_in.shape == (n_in,)
    (n_out,) = pixel_ids_out.shape
    assert times_out.shape == (n_out,)
    assert batch_splits_in.shape == batch_splits_out.shape
    assert len(batch_splits_in.shape) == 1
    predecessor_ids = np.empty((n_in,), "int32")

    batch_size = batch_splits_in.shape[0] - 1
    last = np.empty((grid_size,), "int32")
    # last = np.full((batch_size, grid_size), n_out, "int32")
    predecessor_ids[batch_splits_in[-1] :] = n_out
    # for b in nb.prange(batch_size):
    for b in range(batch_size):
        e_in_start, e_in_end = batch_splits_in[b : b + 2]
        e_out_start, e_out_end = batch_splits_out[b : b + 2]
        last[:] = n_out
        e_out = e_out_end - 1
        for e_in in range(e_in_end - 1, e_in_start - 1, -1):
            t_in = times_in[e_in]
            while e_out >= e_out_start and t_in <= times_out[e_out]:
                last[pixel_ids_out[e_out]] = e_out
                e_out -= 1
            predecessor_ids[e_in] = last[pixel_ids_in[e_in]]

    return predecessor_ids


def ids_to_splits_np(ids: np.ndarray) -> np.ndarray:
    return np.pad(np.cumsum(np.bincount(ids)), [[1, 0]])


def get_random_stream(
    rng: np.random.Generator, num_events: int, grid_size: int, batch_size: int
):
    batch_ids = (rng.uniform(size=num_events) * batch_size).astype("int32")
    batch_ids.sort()
    batch_splits = ids_to_splits_np(batch_ids)
    times = rng.uniform(size=num_events).astype(np.float32)
    times.sort()
    pixel_ids = (rng.uniform(size=num_events) * grid_size).astype("int32")
    return pixel_ids, times, batch_splits


class ConvPreprocessingTest(parameterized.TestCase):
    @parameterized.product(jit=(False, True))
    def test_get_stationary_predecessor_ids(
        self,
        seed: int = 0,
        num_events: int = 71,
        grid_size: int = 11,
        kernel_size: int = 5,
        batch_size: int = 2,
        jit: bool = False,
    ):
        rng = np.random.default_rng(seed)

        pixel_ids, times, batch_splits = get_random_stream(
            rng, num_events, grid_size, batch_size
        )
        del times
        pad_left = (kernel_size - 1) // 2
        pixel_ids += pad_left
        kernel_offsets = np.arange(-pad_left, -pad_left + kernel_size)

        expected = get_stationary_predecessor_ids_np(
            pixel_ids, batch_splits, kernel_offsets, grid_size + kernel_size - 1
        )
        func = partial(
            cp_ops.get_stationary_predecessor_ids, grid_size=grid_size + kernel_size - 1
        )
        if jit:
            func = jax.jit(func)
        with jax.default_device(jax.devices("cpu")[0]):
            actual = func(pixel_ids, batch_splits, kernel_offsets)
        np.testing.assert_equal(actual, expected)

    @parameterized.product(jit=(False, True))
    def test_get_permuted_stationary_predecessor_ids_consistent(
        self,
        seed: int = 0,
        num_events: int = 71,
        grid_size: int = 11,
        kernel_size: int = 5,
        batch_size: int = 2,
        jit: bool = False,
    ):
        rng = np.random.default_rng(seed)

        pixel_ids, times, batch_splits = get_random_stream(
            rng, num_events, grid_size, batch_size
        )
        perm_in = rng.permutation(num_events).astype("int32")
        perm_out = rng.permutation(num_events).astype("int32")
        del times
        pad_left = (kernel_size - 1) // 2
        pixel_ids += pad_left
        kernel_offsets = np.arange(-pad_left, -pad_left + kernel_size)

        grid_size = grid_size + kernel_size - 1

        func = partial(
            cp_ops.get_permuted_stationary_predecessor_ids, grid_size=grid_size
        )
        if jit:
            func = jax.jit(func)
        with jax.default_device(jax.devices("cpu")[0]):
            actual = func(
                pixel_ids,
                batch_splits,
                kernel_offsets,
                inverse_perm(perm_in),
                inverse_perm(perm_out),
            )

            expected = cp_ops.get_stationary_predecessor_ids(
                pixel_ids, batch_splits, kernel_offsets, grid_size=grid_size
            )
            expected = jnp.take(
                jnp.pad(inverse_perm(perm_in), [[0, 1]], constant_values=num_events),
                expected,
                axis=0,
            )
            expected = jnp.take(expected, perm_out, axis=0)
        np.testing.assert_array_equal(actual, expected)

    @parameterized.product(jit=(False, True))
    def test_get_successor_ids(
        self,
        seed: int = 0,
        n_in: int = 53,
        n_out: int = 23,
        grid_size: int = 11,
        batch_size: int = 2,
        jit: bool = False,
    ):
        rng = np.random.default_rng(seed)

        pixel_ids_in, times_in, batch_splits_in = get_random_stream(
            rng, n_in, grid_size, batch_size
        )
        pixel_ids_out, times_out, batch_splits_out = get_random_stream(
            rng, n_out, grid_size, batch_size
        )
        expected = get_successor_ids_np(
            pixel_ids_in,
            times_in,
            batch_splits_in,
            pixel_ids_out,
            times_out,
            batch_splits_out,
            grid_size,
        )
        func = partial(cp_ops.get_successor_ids, grid_size=grid_size)
        if jit:
            func = jax.jit(func)
        with jax.default_device(jax.devices("cpu")[0]):
            actual = func(
                pixel_ids_in,
                times_in,
                batch_splits_in,
                pixel_ids_out,
                times_out,
                batch_splits_out,
            )
        np.testing.assert_equal(actual, expected)

    @parameterized.product(jit=(False, True))
    def test_get_permuted_successor_ids(
        self,
        seed: int = 0,
        n_in: int = 53,
        n_out: int = 23,
        grid_size: int = 11,
        batch_size: int = 2,
        jit: bool = False,
    ):
        rng = np.random.default_rng(seed)

        pixel_ids_in, times_in, batch_splits_in = get_random_stream(
            rng, n_in, grid_size, batch_size
        )
        pixel_ids_out, times_out, batch_splits_out = get_random_stream(
            rng, n_out, grid_size, batch_size
        )
        perm_in = rng.permutation(n_in).astype("int32")
        perm_out = rng.permutation(n_out).astype("int32")

        func = partial(cp_ops.get_permuted_successor_ids, grid_size=grid_size)
        if jit:
            func = jax.jit(func)

        with jax.default_device(jax.devices("cpu")[0]):
            actual = func(
                pixel_ids_in,
                times_in,
                batch_splits_in,
                inverse_perm(perm_in),
                pixel_ids_out,
                times_out,
                batch_splits_out,
                inverse_perm(perm_out),
            )
            expected = cp_ops.get_successor_ids(
                pixel_ids_in,
                times_in,
                batch_splits_in,
                pixel_ids_out,
                times_out,
                batch_splits_out,
                grid_size=grid_size,
            )
            expected = jnp.take(
                jnp.pad(inverse_perm(perm_out), [[0, 1]], constant_values=n_out),
                expected,
                axis=0,
            )
            expected = jnp.take(expected, perm_in, axis=0)
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
