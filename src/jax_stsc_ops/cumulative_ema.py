__all__ = ["segment_cumulative_ema"]

from functools import partial

import jax
import numpy as np
from jax import core
from jax import numpy as jnp
from jax.interpreters import mlir, xla
from jaxlib.hlo_helpers import custom_call

try:
    from . import gpu_ops
except ImportError:
    gpu_ops = None
# Register the CPU XLA custom calls
from . import registrations  # pylint:disable=unused-import

# def segment_cumulative_ema(
#     values: jnp.ndarray,
#     factors: jnp.ndarray,
#     segment_ids: jnp.ndarray,
#     reverse: bool = False,
# ) -> jnp.ndarray:
#     return _segment_cumulative_ema_p.bind(
#         values, factors, segment_ids, reverse=reverse
#     )


def _segment_cumulative_ema(
    values: jnp.ndarray,
    factors: jnp.ndarray,
    segment_ids: jnp.ndarray,
    reverse: bool = False,
) -> jnp.ndarray:
    return _segment_cumulative_ema_p.bind(values, factors, segment_ids, reverse=reverse)


segment_cumulative_ema = jax.custom_vjp(_segment_cumulative_ema, nondiff_argnums=(3,))


def _fwd(values, factors, segment_ids, reverse: bool = False):
    output = _segment_cumulative_ema_p.bind(
        values, factors, segment_ids, reverse=reverse
    )
    return output, (output, factors, segment_ids)


def _bwd(reverse, aux, grad_output):
    output, factors, segment_ids = aux

    if reverse:
        shifted_factors = jnp.pad(factors[:-1], [[1, 0]])
    else:
        shifted_factors = jnp.pad(factors[1:], [[0, 1]])

    grad_values = segment_cumulative_ema(
        grad_output, shifted_factors, segment_ids, reverse=not reverse
    )

    if reverse:
        lagged_output = jnp.pad(output[1:], [[0, 1]])
    else:
        lagged_output = jnp.pad(output[:-1], [[1, 0]])
    grad_factors = grad_values * lagged_output
    grad_factors = jnp.where(
        jnp.pad(segment_ids[:-1] == segment_ids[1:], [[0, 1]] if reverse else [[1, 0]]),
        grad_factors,
        jnp.zeros_like(grad_factors),
    )
    return grad_values, grad_factors, None


segment_cumulative_ema.defvjp(_fwd, _bwd)


# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _segment_cumulative_ema_abstract(
    values: core.ShapedArray,
    factors: core.ShapedArray,
    segment_ids: core.ShapedArray,
    *,
    reverse: bool,
) -> core.ShapedArray:
    return core.ShapedArray(values.shape, values.dtype)


def _segment_cumulative_ema_lowering(
    ctx, values, factors, segment_ids, *, reverse: bool, platform: str
):
    (size,) = values.type.shape
    result_types = [values.type]

    op_name = platform + "_segment_cumulative_ema"
    if platform == "cpu":
        return custom_call(
            op_name,
            result_types=result_types,
            operands=[
                mlir.ir_constant(size),
                mlir.ir_constant(reverse),
                values,
                factors,
                segment_ids,
            ],
            operand_layouts=[(), (), (0,), (0,), (0,)],
            result_layouts=[(0,)],
        ).results
    else:
        assert platform == "gpu", platform
        dtype = np.dtype(ctx.avals_in[0].dtype)
        if dtype == np.float32:
            op_name = op_name + "_f32"
        else:
            assert dtype == np.float64, dtype
            op_name = op_name + "_f64"
        if gpu_ops is None:
            raise ValueError(
                "The 'jax_stsc_ops' module was not compiled with CUDA support"
            )
        opaque = gpu_ops.build_cumulative_ema_descriptor(size, reverse)
        return custom_call(
            op_name,
            result_types=result_types,
            operands=[
                values,
                factors,
                segment_ids,
            ],
            operand_layouts=[(0,), (0,), (0,)],
            result_layouts=[(0,)],
            backend_config=opaque,
        ).results


# def _normalize_axis(axis, ndims):
#     if axis < 0:
#         axis += ndims
#     assert axis < ndims, (axis, ndims)
#     return axis


# def _segment_cumulative_ema_p_batch(args, axes, reverse: bool):
#     values, factors, segment_ids = args
#     values_axis, factors_axis, segment_ids_axis = axes

#     if values_axis is not None:
#         size = values.shape[values_axis]
#     elif factors_axis is not None:
#         size = factors.shape[factors_axis]
#     elif segment_ids_axis is not None:
#         size = segment_ids.shape[segment_ids_axis]
#     else:
#         raise ValueError("At least one axis must be non-None")

#     if values_axis is None:
#         values = jnp.expand_dims(values, axis=-1)
#     else:
#         ndims = len(values.shape)
#         values_axis = _normalize_axis(values_axis, ndims)
#         if values_axis != ndims - 1:
#             values = np.swapaxes(values, values_axis, ndims - 1)
#     if factors_axis is None:
#         factors = jnp.expand_dims(factors, axis=-1)
#     else:
#         ndims = len(factors.shape)
#         factors_axis = _normalize_axis(factors_axis, ndims)
#         if factors_axis != ndims - 1:
#             factors = np.swapaxes(factors, factors_axis, ndims - 1)

#     if values_axis != 0:
#         values = jnp.swapaxes(values, 0, values_axis)
#     if factors_axis != 0:
#         factors = jnp.swapaxes(factors, 0, factors_axis)
#     if segment_ids_axis != 0:
#         segment_ids = jnp.swapaxes(segment_ids, 0, segment_ids_axis)

#     assert values.shape == factors.shape == segment_ids.shape
#     raise Exception("TODO")


# def _segment_cumulative_ema_jvp(primals, tangents, *, reverse: bool):
#     values, factors, segment_ids = primals
#     d_values, d_factors, _ = tangents

#     output = segment_cumulative_ema(values, factors, segment_ids, reverse=reverse)
#     if reverse:
#         lagged_output = jnp.pad(output[1:], [[0, 1]])
#     else:
#         lagged_output = jnp.pad(output[:-1], [[1, 0]])

#     new_values = lagged_output * d_factors + d_values
#     tangent_out = segment_cumulative_ema(
#         new_values, factors, segment_ids, reverse=reverse
#     )
#     return output, tangent_out


# def _segment_cumulative_ema_transpose_rule(
#     cotangent: jnp.ndarray,
#     values: jnp.ndarray,
#     factors: jnp.ndarray,
#     segment_ids: jnp.ndarray,
#     *,
#     reverse: bool,
# ) -> tp.Sequence[jnp.ndarray]:
#     output = segment_cumulative_ema(values, factors, segment_ids, reverse=reverse)
#     grad_output = cotangent

#     if reverse:
#         shifted_factors = jnp.pad(factors[:-1], [[1, 0]])
#     else:
#         shifted_factors = jnp.pad(factors[1:], [[0, 1]])

#     grad_values = segment_cumulative_ema(
#         grad_output, shifted_factors, segment_ids, reverse=not reverse
#     )

#     if reverse:
#         lagged_output = jnp.pad(output[1:], [[0, 1]])
#     else:
#         lagged_output = jnp.pad(output[:-1], [[1, 0]])
#     grad_factors = grad_values * lagged_output
#     grad_factors = jnp.where(
#         jnp.pad(segment_ids[:-1] == segment_ids[1:], [[0, 1]] if reverse else [[1, 0]]),
#         grad_factors,
#         jnp.zeros_like(grad_factors),
#     )
#     return grad_values, grad_factors, None


# def _segment_cumulative_ema_transpose_rule(x, reverse):
#     # The logic here should be similar to the logic in _bwd
#     # Compute grad_values and grad_factors based on output_tangent
#     raise Exception(x)
#     values, factors, segment_ids = args
#     output = segment_cumulative_ema(values, factors, segment_ids, reverse=reverse)

#     if reverse:
#         shifted_factors = jnp.pad(factors[:-1], [[1, 0]])
#     else:
#         shifted_factors = jnp.pad(factors[1:], [[0, 1]])

#     grad_values = segment_cumulative_ema(
#         output_tangent, shifted_factors, segment_ids, reverse=not reverse
#     )

#     if reverse:
#         lagged_output = jnp.pad(output[1:], [[0, 1]])
#     else:
#         lagged_output = jnp.pad(output[:-1], [[1, 0]])
#     grad_factors = grad_values * lagged_output
#     grad_factors = jnp.where(
#         jnp.pad(segment_ids[:-1] == segment_ids[1:], [[0, 1]] if reverse else [[1, 0]]),
#         grad_factors,
#         jnp.zeros_like(grad_factors),
#     )

#     # Return gradients w.r.t. each differentiable input
#     return grad_values, grad_factors


_segment_cumulative_ema_p = core.Primitive("segment_cumulative_ema")
_segment_cumulative_ema_p.def_impl(
    partial(xla.apply_primitive, _segment_cumulative_ema_p)
)
_segment_cumulative_ema_p.def_abstract_eval(_segment_cumulative_ema_abstract)


# Connect the XLA translation rules for JIT compilation
for platform in ("cpu", "gpu"):
    mlir.register_lowering(
        _segment_cumulative_ema_p,
        partial(_segment_cumulative_ema_lowering, platform=platform),
        platform=platform,
    )


def _cumulative_ema(
    values: jnp.ndarray, factors: jnp.ndarray, reverse: bool = False
) -> jnp.ndarray:
    return _cumulative_ema_p.bind(values, factors, reverse=reverse)


cumulative_ema = jax.custom_vjp(_cumulative_ema, nondiff_argnums=(2,))


def _fwd(values, factors, reverse: bool = False):
    output = _cumulative_ema_p.bind(values, factors, reverse=reverse)
    return output, (output, factors)


def _bwd(reverse, aux, grad_output):
    output, factors = aux

    if reverse:
        shifted_factors = jnp.pad(factors[:-1], [[1, 0]])
    else:
        shifted_factors = jnp.pad(factors[1:], [[0, 1]])

    grad_values = cumulative_ema(grad_output, shifted_factors, reverse=not reverse)

    if reverse:
        lagged_output = jnp.pad(output[1:], [[0, 1]])
    else:
        lagged_output = jnp.pad(output[:-1], [[1, 0]])
    grad_factors = grad_values * lagged_output
    return grad_values, grad_factors


cumulative_ema.defvjp(_fwd, _bwd)


# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _cumulative_ema_abstract(
    values: core.ShapedArray,
    factors: core.ShapedArray,
    *,
    reverse: bool,
) -> core.ShapedArray:
    return core.ShapedArray(values.shape, values.dtype)


def _cumulative_ema_lowering(ctx, values, factors, *, reverse: bool, platform: str):
    (size,) = values.type.shape
    result_types = [values.type]

    dtype = np.dtype(ctx.avals_in[0].dtype)
    assert np.dtype(ctx.avals_in[1].dtype) == dtype
    if dtype == np.float32:
        dtype_suffix = "f32"
    elif dtype == np.float64:
        dtype_suffix = "f64"
    elif dtype == np.complex64:
        dtype_suffix = "c64"
    elif dtype == np.complex128:
        dtype_suffix = "c128"
    else:
        raise TypeError(f"Unsupported dtype {dtype}")

    op_name = platform + "_cumulative_ema_" + dtype_suffix
    if platform == "cpu":
        return custom_call(
            op_name,
            result_types=result_types,
            operands=[
                mlir.ir_constant(size),
                mlir.ir_constant(reverse),
                values,
                factors,
            ],
            operand_layouts=[(), (), (0,), (0,)],
            result_layouts=[(0,)],
        ).results
    else:
        assert platform == "gpu", platform
        if gpu_ops is None:
            raise ValueError(
                "The 'jax_stsc_ops' module was not compiled with CUDA support"
            )
        opaque = gpu_ops.build_cumulative_ema_descriptor(size, reverse)
        return custom_call(
            op_name,
            result_types=result_types,
            operands=[values, factors],
            operand_layouts=[(0,), (0,)],
            result_layouts=[(0,)],
            backend_config=opaque,
        ).results


_cumulative_ema_p = core.Primitive("cumulative_ema")
_cumulative_ema_p.def_impl(partial(xla.apply_primitive, _cumulative_ema_p))
_cumulative_ema_p.def_abstract_eval(_cumulative_ema_abstract)


# Connect the XLA translation rules for JIT compilation
for platform in ("cpu", "gpu"):
    mlir.register_lowering(
        _cumulative_ema_p,
        partial(_cumulative_ema_lowering, platform=platform),
        platform=platform,
    )


def segment_cumulative_ema_basic(values, factors, segment_ids, reverse: bool = False):
    padding = [[0, 0] for _ in segment_ids.shape]
    if reverse:
        padding[0][1] = 1
    else:
        padding[0][0] = 1
    scale_factor = jnp.pad(segment_ids[:-1] == segment_ids[1:], padding)
    factors = factors * scale_factor.astype(factors.dtype)
    values, factors = jnp.broadcast_arrays(values, factors)
    if len(values.shape) == 1:
        return cumulative_ema(values, factors, reverse=reverse)
    assert all(s == 1 for s in segment_ids.shape[1:]), segment_ids.shape
    n = values.shape[0]
    return cumulative_ema(
        values.reshape(n, -1).T.reshape(-1),
        factors.reshape(n, -1).T.reshape(-1),
        reverse=reverse,
    ).T.reshape(values.shape)


def segment_cumulative_ema_basic_from_splits(
    values: jnp.ndarray,
    factors: jnp.ndarray,
    splits: jnp.ndarray,
    reverse: bool = False,
) -> jnp.ndarray:
    factors = factors.at[splits[:-1] if reverse else splits[1:]].set(0)
    return cumulative_ema(values, factors, reverse=reverse)


def associative_scan_cumulative_ema(
    values: jnp.ndarray, factors: jnp.ndarray, reverse: bool = False, axis: int = 0
) -> jnp.ndarray:
    def add(a, b):
        v_a, f_a = a
        v_b, f_b = b
        return v_a * f_b + v_b, f_a * f_b

    output = jax.lax.associative_scan(
        add, (values, factors), reverse=reverse, axis=axis
    )
    return output[0]


def associative_scan_segment_cumulative_ema_v0(
    values: jnp.ndarray,
    factors: jnp.ndarray,
    segment_ids: jnp.ndarray,
    reverse: bool = False,
    axis: int = 0,
) -> jnp.ndarray:
    def add(a, b):
        v_a, f_a, s_a = a
        v_b, f_b, s_b = b
        same = s_a == s_b
        v_out = jnp.where(same, v_a * f_b + v_b, v_b)
        f_out = jnp.where(same, f_a * f_b, f_b)
        s_out = s_b
        return (v_out, f_out, s_out)

    output = jax.lax.associative_scan(
        add, (values, factors, segment_ids), reverse=reverse, axis=axis
    )
    return output[0]


def associative_scan_segment_cumulative_ema(
    values: jnp.ndarray,
    factors: jnp.ndarray,
    segment_ids: jnp.ndarray,
    reverse: bool = False,
    axis: int = 0,
) -> jnp.ndarray:
    padding = [[0, 0] for _ in segment_ids.shape]
    if reverse:
        padding[0][1] = 1
    else:
        padding[0][0] = 1
    scale_factor = jnp.pad(
        (segment_ids[1:] == segment_ids[:-1]).astype(factors.dtype), padding
    )
    factors = factors * scale_factor

    return associative_scan_cumulative_ema(values, factors, reverse=reverse, axis=axis)


def associative_scan_segment_cumulative_ema_from_splits(
    values: jnp.ndarray,
    factors: jnp.ndarray,
    splits: jnp.ndarray,
    reverse: bool = False,
    axis: int = 0,
) -> jnp.ndarray:
    factors = factors.at[splits[:-1] if reverse else splits[1:]].set(0)

    return associative_scan_cumulative_ema(values, factors, reverse=reverse, axis=axis)


def tiled_segment_cumulative_ema(
    values: jnp.ndarray,
    factors: jnp.ndarray,
    segment_ids: jnp.ndarray,
    reverse: bool = False,
) -> jnp.ndarray:
    return _tiled_segment_cumulative_ema_p.bind(
        values, factors, segment_ids, reverse=reverse
    )


def _tiled_segment_cumulative_ema_abstract(
    values: core.ShapedArray,
    factors: core.ShapedArray,
    segment_ids: core.ShapedArray,
    *,
    reverse: bool,
) -> core.ShapedArray:
    return core.ShapedArray(values.shape, values.dtype)


def _tiled_segment_cumulative_ema_lowering(
    ctx, values, factors, segment_ids, *, reverse: bool, platform: str
):
    (size, num_channels) = values.type.shape
    result_types = [values.type]

    op_name = platform + "_tiled_segment_cumulative_ema"
    if platform == "cpu":
        return custom_call(
            op_name,
            result_types=result_types,
            operands=[
                mlir.ir_constant(num_channels),
                mlir.ir_constant(size),
                mlir.ir_constant(reverse),
                values,
                factors,
                segment_ids,
            ],
            operand_layouts=[(), (), (), (0, 1), (0, 1), (0,)],
            result_layouts=[(0, 1)],
        ).results
    else:
        assert platform == "gpu", platform
        dtype = np.dtype(ctx.avals_in[0].dtype)
        if dtype == np.float32:
            op_name = op_name + "_f32"
        else:
            assert dtype == np.float64, dtype
            op_name = op_name + "_f64"
        if gpu_ops is None:
            raise ValueError(
                "The 'jax_stsc_ops' module was not compiled with CUDA support"
            )
        opaque = gpu_ops.build_tiled_cumulative_ema_descriptor(
            num_channels, size, reverse
        )
        return custom_call(
            op_name,
            result_types=result_types,
            operands=[
                values,
                factors,
                segment_ids,
            ],
            operand_layouts=[(0, 1), (0, 1), (0,)],
            result_layouts=[(0, 1)],
            backend_config=opaque,
        ).results


_tiled_segment_cumulative_ema_p = core.Primitive("segment_cumulative_ema")
_tiled_segment_cumulative_ema_p.def_impl(
    partial(xla.apply_primitive, _tiled_segment_cumulative_ema_p)
)
_tiled_segment_cumulative_ema_p.def_abstract_eval(
    _tiled_segment_cumulative_ema_abstract
)

# Connect the XLA translation rules for JIT compilation
for platform in ("cpu", "gpu"):
    mlir.register_lowering(
        _tiled_segment_cumulative_ema_p,
        partial(_tiled_segment_cumulative_ema_lowering, platform=platform),
        platform=platform,
    )


def tiled_segment_cumulative_ema_basic(
    values: jnp.ndarray,
    factors: jnp.ndarray,
    segment_ids: jnp.ndarray,
    reverse: bool = False,
) -> jnp.ndarray:
    scale_factor = jnp.pad(
        segment_ids[:-1] == segment_ids[1:], [[0, 1]] if reverse else [[1, 0]]
    )
    factors = factors * jnp.expand_dims(scale_factor, 1).astype(factors.dtype)
    return cumulative_ema(values.flatten(), factors.flatten(), reverse=reverse).reshape(
        values.shape
    )


def tiled_segment_cumulative_ema_basic_from_splits(
    values: jnp.ndarray,
    factors: jnp.ndarray,
    splits: jnp.ndarray,
    reverse: bool = False,
) -> jnp.ndarray:
    factors = factors.at[splits[:-1] if reverse else splits[1:]].set(0)
    return cumulative_ema(values.flatten(), factors.flatten(), reverse=reverse).reshape(
        values.shape
    )


def serial_segment_cumulative_ema(
    values: jnp.ndarray,
    factors: jnp.ndarray,
    splits: jnp.ndarray,
    *,
    reverse: bool = False,
):
    return _serial_segment_cumulative_ema_p.bind(
        values, factors, splits, reverse=reverse
    )


def _serial_segment_cumulative_ema_abstract(
    values: core.ShapedArray,
    factors: core.ShapedArray,
    splits: core.ShapedArray,
    *,
    reverse: bool,
) -> core.ShapedArray:
    return core.ShapedArray(values.shape, values.dtype)


def _serial_segment_cumulative_ema_lowering(
    ctx, values, factors, splits, *, reverse: bool, platform: str
):
    (num_events, num_channels) = values.type.shape
    (num_segments,) = splits.type.shape
    num_segments -= 1
    result_types = [values.type]

    dtype = np.dtype(ctx.avals_in[0].dtype)
    if dtype == np.float32:
        dtype_str = "f32"
    else:
        assert dtype == np.float64, dtype
        dtype_str = "f64"
    op_name = platform + "_serial_segment_cumulative_ema_" + dtype_str

    if platform == "cpu":
        return custom_call(
            op_name,
            result_types=result_types,
            operands=[
                mlir.ir_constant(num_channels),
                mlir.ir_constant(num_events),
                mlir.ir_constant(num_segments),
                mlir.ir_constant(reverse),
                values,
                factors,
                splits,
            ],
            operand_layouts=[(), (), (0, 1), (0, 1), (0,)],
            result_layouts=[(0, 1)],
        ).results
    else:
        assert platform == "gpu", platform
        if gpu_ops is None:
            raise ValueError(
                "The 'jax_stsc_ops' module was not compiled with CUDA support"
            )
        opaque = gpu_ops.build_tiled_segment_cumulative_ema_descriptor(
            num_channels, num_events, num_segments, reverse
        )
        return custom_call(
            op_name,
            result_types=result_types,
            operands=[
                values,
                factors,
                splits,
            ],
            operand_layouts=[(0, 1), (0, 1), (0,)],
            result_layouts=[(0, 1)],
            backend_config=opaque,
        ).results


_serial_segment_cumulative_ema_p = core.Primitive("serial_segment_cumulative_ema_p")
_serial_segment_cumulative_ema_p.def_impl(
    partial(xla.apply_primitive, _serial_segment_cumulative_ema_p)
)
_serial_segment_cumulative_ema_p.def_abstract_eval(
    _serial_segment_cumulative_ema_abstract
)


# Connect the XLA translation rules for JIT compilation
for platform in ("cpu", "gpu"):
    mlir.register_lowering(
        _serial_segment_cumulative_ema_p,
        partial(_serial_segment_cumulative_ema_lowering, platform=platform),
        platform=platform,
    )
