__all__ = ["throttled_sample"]

import typing as tp
from functools import partial

import numpy as np
from jax import core
from jax import numpy as jnp
from jax.interpreters import mlir, xla
from jaxlib.hlo_helpers import custom_call

# Register the CPU XLA custom calls
from . import registrations  # pylint:disable=unused-import


def throttled_sample(
    pixel_ids: jnp.ndarray,
    times: jnp.ndarray,
    batch_splits: jnp.ndarray,
    grid_size: int,
    sample_rate: int,
    min_dt: float = 0.0,
    size_out: int | None = None,
) -> tp.Tuple[jnp.ndarray, jnp.ndarray]:
    pixel_ids = jnp.asarray(pixel_ids, jnp.int32)
    times = jnp.asarray(times, jnp.float32)
    batch_splits = jnp.asarray(batch_splits, jnp.int32)
    if size_out is None:
        size_out = pixel_ids.shape[0] // sample_rate
    return _throttled_sample_prim.bind(
        pixel_ids,
        times,
        batch_splits,
        grid_size=grid_size,
        sample_rate=sample_rate,
        min_dt=min_dt,
        size_out=size_out,
    )


# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _throttled_sample_abstract(
    pixel_ids: core.ShapedArray,
    times: core.ShapedArray,
    batch_splits: core.ShapedArray,
    *,
    grid_size: int,
    sample_rate: int,
    min_dt: float,
    size_out: int,
) -> core.ShapedArray:
    sample_ids = core.ShapedArray((size_out,), pixel_ids.dtype)
    batch_splits_out = core.ShapedArray(batch_splits.shape, batch_splits.dtype)
    return sample_ids, batch_splits_out


def _throttled_sample_lowering(
    ctx,
    pixel_ids,
    times,
    batch_splits,
    *,
    grid_size: int,
    sample_rate: int,
    min_dt: float,
    size_out: int,
):
    (size_in,) = pixel_ids.type.shape

    splits_type = batch_splits.type
    batch_size = splits_type.shape[0] - 1

    sample_ids_type = mlir.aval_to_ir_type(core.ShapedArray((size_out,), jnp.int32))

    op_name = "cpu_throttled_sample"
    return custom_call(
        op_name,
        result_types=[sample_ids_type, splits_type],
        operands=[
            mlir.ir_constant(grid_size),
            mlir.ir_constant(batch_size),
            mlir.ir_constant(size_in),
            mlir.ir_constant(size_out),
            mlir.ir_constant(sample_rate),
            mlir.ir_constant(np.asarray(min_dt, np.float32)),
            pixel_ids,
            times,
            batch_splits,
        ],
        operand_layouts=[(), (), (), (), (), (), (0,), (0,), (0,)],
        result_layouts=[(0,), (0,)],
    ).results


_throttled_sample_prim = core.Primitive("throttled_sample")
_throttled_sample_prim.multiple_results = True
_throttled_sample_prim.def_impl(partial(xla.apply_primitive, _throttled_sample_prim))
_throttled_sample_prim.def_abstract_eval(_throttled_sample_abstract)

# Connect the XLA translation rules for JIT compilation
mlir.register_lowering(
    _throttled_sample_prim, _throttled_sample_lowering, platform="cpu"
)
