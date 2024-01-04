"""Use conv_preprocessing ops in python."""
__all__ = [
    "get_stationary_predecessor_ids",
    "get_successor_ids",
]

from functools import partial

from jax import core
from jax import numpy as jnp
from jax.interpreters import mlir, xla
from jaxlib.hlo_helpers import custom_call


def get_stationary_predecessor_ids(
    pixel_ids: jnp.ndarray,
    batch_splits: jnp.ndarray,
    kernel_offsets: jnp.ndarray,
    grid_size: int,
) -> jnp.ndarray:
    """
    Get predecessor indices for a stationary chronological event stream.

    This is an optimization of `get_predecessor_ids` for the case where the input stream
    is the same as the output stream. The results should be identical so long as times
    are unique.

    Args:
        pixel_ids: [E] in [0, grid_size)
        batch_splits: [B+1] in [0, E]
        kernel_offsets: [K]
        grid_size:

    Returns:
        [E, K] values in [0, E)
    """
    return _get_stationary_predecessor_ids_p.bind(
        jnp.asarray(pixel_ids, jnp.int32),
        jnp.asarray(batch_splits, jnp.int32),
        jnp.asarray(kernel_offsets, jnp.int32),
        grid_size=grid_size,
    )


# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _get_stationary_predecessor_ids_abstract(
    pixel_ids: core.ShapedArray,
    batch_splits: core.ShapedArray,
    kernel_offsets: core.ShapedArray,
    *,
    grid_size: int,
) -> core.ShapedArray:
    return core.ShapedArray(
        (pixel_ids.shape[0], kernel_offsets.shape[0]), dtype=pixel_ids.dtype
    )


def _get_stationary_predecessor_ids_lowering(
    ctx, pixel_ids, batch_splits, kernel_offsets, *, grid_size: int
):
    (num_events,) = pixel_ids.type.shape
    batch_size = batch_splits.type.shape[0] - 1
    (kernel_size,) = kernel_offsets.type.shape

    op_name = "cpu_get_stationary_predecessor_ids"
    return custom_call(
        op_name,
        # Output types
        result_types=[
            mlir.aval_to_ir_type(core.ShapedArray((num_events, kernel_size), jnp.int32))
        ],
        # The inputs:
        operands=[
            mlir.ir_constant(grid_size),
            mlir.ir_constant(batch_size),
            mlir.ir_constant(num_events),
            mlir.ir_constant(kernel_size),
            pixel_ids,
            batch_splits,
            kernel_offsets,
        ],
        # Layout specification:
        operand_layouts=[(), (), (), (), (0,), (0,), (0,)],
        result_layouts=[(1, 0)],
    ).results


_get_stationary_predecessor_ids_p = core.Primitive("get_stationary_predecessor_ids")
_get_stationary_predecessor_ids_p.def_impl(
    partial(xla.apply_primitive, _get_stationary_predecessor_ids_p)
)
_get_stationary_predecessor_ids_p.def_abstract_eval(
    _get_stationary_predecessor_ids_abstract
)

# Connect the XLA translation rules for JIT compilation
mlir.register_lowering(
    _get_stationary_predecessor_ids_p,
    _get_stationary_predecessor_ids_lowering,
    platform="cpu",
)


def get_permuted_stationary_predecessor_ids(
    pixel_ids: jnp.ndarray,
    batch_splits: jnp.ndarray,
    kernel_offsets: jnp.ndarray,
    perm_in: jnp.ndarray,
    perm_out: jnp.ndarray,
    grid_size: int,
) -> jnp.ndarray:
    """
    Get predecessor indices for a stationary chronological event stream.

    This is an optimization of `get_predecessor_ids` for the case where the input stream
    is the same as the output stream. The results should be identical so long as times
    are unique.

    Args:
        pixel_ids: [E] in [0, grid_size)
        batch_splits: [B+1] in [0, E]
        kernel_offsets: [K]
        grid_size:

    Returns:
        [E, K] values in [0, E)
    """
    return _get_permuted_stationary_predecessor_ids_p.bind(
        jnp.asarray(pixel_ids, jnp.int32),
        jnp.asarray(batch_splits, jnp.int32),
        jnp.asarray(kernel_offsets, jnp.int32),
        jnp.asarray(perm_in, jnp.int32),
        jnp.asarray(perm_out, jnp.int32),
        grid_size=grid_size,
    )


# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _get_permuted_stationary_predecessor_ids_abstract(
    pixel_ids: core.ShapedArray,
    batch_splits: core.ShapedArray,
    kernel_offsets: core.ShapedArray,
    perm_in: core.ShapedArray,
    perm_out: core.ShapedArray,
    *,
    grid_size: int,
) -> core.ShapedArray:
    return core.ShapedArray(
        (pixel_ids.shape[0], kernel_offsets.shape[0]), dtype=pixel_ids.dtype
    )


def _get_permuted_stationary_predecessor_ids_lowering(
    ctx, pixel_ids, batch_splits, kernel_offsets, perm_in, perm_out, *, grid_size: int
):
    (num_events,) = pixel_ids.type.shape
    batch_size = batch_splits.type.shape[0] - 1
    (kernel_size,) = kernel_offsets.type.shape

    op_name = "cpu_get_permuted_stationary_predecessor_ids"
    return custom_call(
        op_name,
        # Output types
        result_types=[
            mlir.aval_to_ir_type(core.ShapedArray((num_events, kernel_size), jnp.int32))
        ],
        # The inputs:
        operands=[
            mlir.ir_constant(grid_size),
            mlir.ir_constant(batch_size),
            mlir.ir_constant(num_events),
            mlir.ir_constant(kernel_size),
            pixel_ids,
            batch_splits,
            kernel_offsets,
            perm_in,
            perm_out,
        ],
        # Layout specification:
        operand_layouts=[(), (), (), (), (0,), (0,), (0,), (0,), (0,)],
        result_layouts=[(1, 0)],
    ).results


_get_permuted_stationary_predecessor_ids_p = core.Primitive(
    "get_permuted_stationary_predecessor_ids"
)
_get_permuted_stationary_predecessor_ids_p.def_impl(
    partial(xla.apply_primitive, _get_permuted_stationary_predecessor_ids_p)
)
_get_permuted_stationary_predecessor_ids_p.def_abstract_eval(
    _get_permuted_stationary_predecessor_ids_abstract
)

# Connect the XLA translation rules for JIT compilation
mlir.register_lowering(
    _get_permuted_stationary_predecessor_ids_p,
    _get_permuted_stationary_predecessor_ids_lowering,
    platform="cpu",
)


def get_successor_ids(
    pixel_ids_in: jnp.ndarray,
    times_in: jnp.ndarray,
    batch_splits_in: jnp.ndarray,
    pixel_ids_out: jnp.ndarray,
    times_out: jnp.ndarray,
    batch_splits_out: jnp.ndarray,
    grid_size: int,
) -> jnp.ndarray:
    """
    Get successor_ids as used in e.g. `exclusive_conv`.

    If `successor_ids[e_in] == e_out`, it means that output event `e_out`
    is the earliest event at or after input event `e_in` with the same pixel_id.

    Args:
        pixel_ids_in: [E_in] in [0, grid_size)
        times_in: [E_in]
        batch_splits_in: [B + 1] in [0, E_in]
        pixel_ids_out: [E_out] in [0, grid_size)
        times_out: [E_out]
        batch_splits_out: [B + 1] in [0, E_out]
        grid_size:

    Returns:
        [E_in] successor_ids in [0, E_out]
    """
    return _get_successor_ids_p.bind(
        jnp.asarray(pixel_ids_in, jnp.int32),
        jnp.asarray(times_in, jnp.float32),
        jnp.asarray(batch_splits_in, jnp.int32),
        jnp.asarray(pixel_ids_out, jnp.int32),
        jnp.asarray(times_out, jnp.float32),
        jnp.asarray(batch_splits_out, jnp.int32),
        grid_size=grid_size,
    )


# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _get_successor_ids_abstract(
    pixel_ids_in: core.ShapedArray,
    times_in: core.ShapedArray,
    batch_splits_in: core.ShapedArray,
    pixel_ids_out: core.ShapedArray,
    times_out: core.ShapedArray,
    batch_splits_out: core.ShapedArray,
    *,
    grid_size: int,
) -> core.ShapedArray:
    return core.ShapedArray((pixel_ids_in.shape[0],), dtype=jnp.int32)


def _get_successor_ids_lowering(
    ctx,
    pixel_ids_in: core.ShapedArray,
    times_in,
    batch_splits_in,
    pixel_ids_out,
    times_out,
    batch_splits_out,
    *,
    grid_size: int,
):
    batch_size = batch_splits_in.type.shape[0] - 1
    (num_events_in,) = pixel_ids_in.type.shape
    (num_events_out,) = pixel_ids_out.type.shape

    op_name = "cpu_get_successor_ids"
    return custom_call(
        op_name,
        result_types=[pixel_ids_in.type],
        operands=[
            mlir.ir_constant(grid_size),
            mlir.ir_constant(batch_size),
            mlir.ir_constant(num_events_in),
            mlir.ir_constant(num_events_out),
            pixel_ids_in,
            times_in,
            batch_splits_in,
            pixel_ids_out,
            times_out,
            batch_splits_out,
        ],
        operand_layouts=[(), (), (), (), (0,), (0,), (0,), (0,), (0,), (0,)],
        result_layouts=[(0,)],
    ).results


_get_successor_ids_p = core.Primitive("get_successor_ids")
_get_successor_ids_p.def_impl(partial(xla.apply_primitive, _get_successor_ids_p))
_get_successor_ids_p.def_abstract_eval(_get_successor_ids_abstract)

# Connect the XLA translation rules for JIT compilation
mlir.register_lowering(
    _get_successor_ids_p, _get_successor_ids_lowering, platform="cpu"
)


def get_permuted_successor_ids(
    pixel_ids_in: jnp.ndarray,
    times_in: jnp.ndarray,
    batch_splits_in: jnp.ndarray,
    perm_in: jnp.ndarray,
    pixel_ids_out: jnp.ndarray,
    times_out: jnp.ndarray,
    batch_splits_out: jnp.ndarray,
    perm_out: jnp.ndarray,
    grid_size: int,
) -> jnp.ndarray:
    """
    Get successor_ids as used in e.g. `exclusive_conv`.

    If `successor_ids[e_in] == e_out`, it means that output event `e_out`
    is the earliest event at or after input event `e_in` with the same pixel_id.

    Args:
        pixel_ids_in: [E_in] in [0, grid_size)
        times_in: [E_in]
        batch_splits_in: [B + 1] in [0, E_in]
        pixel_ids_out: [E_out] in [0, grid_size)
        times_out: [E_out]
        batch_splits_out: [B + 1] in [0, E_out]
        grid_size:

    Returns:
        [E_in] successor_ids in [0, E_out]
    """
    return _get_permuted_successor_ids_p.bind(
        jnp.asarray(pixel_ids_in, jnp.int32),
        jnp.asarray(times_in, jnp.float32),
        jnp.asarray(batch_splits_in, jnp.int32),
        jnp.asarray(perm_in, jnp.int32),
        jnp.asarray(pixel_ids_out, jnp.int32),
        jnp.asarray(times_out, jnp.float32),
        jnp.asarray(batch_splits_out, jnp.int32),
        jnp.asarray(perm_out, jnp.int32),
        grid_size=grid_size,
    )


# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _get_permuted_successor_ids_abstract(
    pixel_ids_in: core.ShapedArray,
    times_in: core.ShapedArray,
    batch_splits_in: core.ShapedArray,
    perm_in: core.ShapedArray,
    pixel_ids_out: core.ShapedArray,
    times_out: core.ShapedArray,
    batch_splits_out: core.ShapedArray,
    perm_out: core.ShapedArray,
    *,
    grid_size: int,
) -> core.ShapedArray:
    return core.ShapedArray((pixel_ids_in.shape[0],), dtype=jnp.int32)


def _get_permuted_successor_ids_lowering(
    ctx,
    pixel_ids_in: core.ShapedArray,
    times_in,
    batch_splits_in,
    perm_in,
    pixel_ids_out,
    times_out,
    batch_splits_out,
    perm_out,
    *,
    grid_size: int,
):
    batch_size = batch_splits_in.type.shape[0] - 1
    (num_events_in,) = pixel_ids_in.type.shape
    (num_events_out,) = pixel_ids_out.type.shape

    op_name = "cpu_get_permuted_successor_ids"
    return custom_call(
        op_name,
        result_types=[pixel_ids_in.type],
        operands=[
            mlir.ir_constant(grid_size),
            mlir.ir_constant(batch_size),
            mlir.ir_constant(num_events_in),
            mlir.ir_constant(num_events_out),
            pixel_ids_in,
            times_in,
            batch_splits_in,
            perm_in,
            pixel_ids_out,
            times_out,
            batch_splits_out,
            perm_out,
        ],
        operand_layouts=[
            (),
            (),
            (),
            (),
            (0,),
            (0,),
            (0,),
            (0,),
            (0,),
            (0,),
            (0,),
            (0,),
        ],
        result_layouts=[(0,)],
    ).results


_get_permuted_successor_ids_p = core.Primitive("get_permuted_successor_ids")
_get_permuted_successor_ids_p.def_impl(
    partial(xla.apply_primitive, _get_permuted_successor_ids_p)
)
_get_permuted_successor_ids_p.def_abstract_eval(_get_permuted_successor_ids_abstract)

# Connect the XLA translation rules for JIT compilation
mlir.register_lowering(
    _get_permuted_successor_ids_p, _get_permuted_successor_ids_lowering, platform="cpu"
)
