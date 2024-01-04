__all__ = ["counting_argsort"]

from functools import partial

from jax import core
from jax import numpy as jnp
from jax.interpreters import mlir, xla
from jaxlib.hlo_helpers import custom_call

# Register the CPU XLA custom calls
from . import registrations  # pylint:disable=unused-import


def counting_argsort(ids: jnp.ndarray, splits: jnp.ndarray) -> jnp.ndarray:
    return _counting_argsort_prim.bind(ids, splits)


# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _counting_argsort_abstract(
    ids: core.ShapedArray, splits: core.ShapedArray
) -> core.ShapedArray:
    del splits
    return core.ShapedArray(ids.shape, ids.dtype)


def _counting_argsort_lowering(ctx, ids, splits):
    ids_type = ids.type
    (num_ids,) = ids_type.shape

    num_segments = splits.type.shape[0] - 1

    op_name = "cpu_counting_argsort"
    return custom_call(
        op_name,
        result_types=[ids_type],
        operands=[
            mlir.ir_constant(num_ids),
            mlir.ir_constant(num_segments),
            ids,
            splits,
        ],
        operand_layouts=[(), (), (0,), (0,)],
        result_layouts=[(0,)],
    ).results


_counting_argsort_prim = core.Primitive("counting_argsort")
_counting_argsort_prim.def_impl(partial(xla.apply_primitive, _counting_argsort_prim))
_counting_argsort_prim.def_abstract_eval(_counting_argsort_abstract)

# Connect the XLA translation rules for JIT compilation
mlir.register_lowering(
    _counting_argsort_prim, _counting_argsort_lowering, platform="cpu"
)
