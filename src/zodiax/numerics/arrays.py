"""Array conversion shared by Zodiax numerical definitions."""

from __future__ import annotations

from typing import Any

import jax.numpy as np
from jax import Array

from .expressions import Expression

__all__ = ["as_array", "broadcast_flatten"]


def as_array(value: Any, *, dtype: type | None = None) -> Array | Expression | None:
    """Convert numerical values while preserving Expressions and None.

    With dtype=None, infer the numerical type from the input, retaining existing
    array dtypes subject to JAX's configuration. Use float, int, complex, or bool
    to request a numerical type; JAX's startup configuration chooses its default
    precision. Callers supply floating or complex values for trainable parameters.

    Expressions and None pass through unchanged, even when dtype is supplied.
    No cast is attached to an expression for later evaluation.
    """
    if value is None or isinstance(value, Expression):
        return value

    array = np.asarray(value, dtype=dtype)

    # Python scalars can produce weakly typed arrays, whose type adapts to other
    # operands. Supplying the inferred dtype explicitly removes that weak_type
    # flag without choosing a different precision for the stored array.
    return np.asarray(array, dtype=array.dtype)


def broadcast_flatten(
    value: Any, shape: tuple[int, ...], *, dtype: type | None = None
) -> Array:
    """Broadcast an array-like value to ``shape`` and flatten in row-major order.

    Preserves the inferred dtype unless ``dtype`` is supplied. Unlike ``as_array``,
    this operation requires realised numerical values, not deferred expressions.
    The target shape must be static when used inside JIT.
    """
    value = np.asarray(value, dtype=dtype)
    try:
        return np.broadcast_to(value, shape).reshape(-1)
    except ValueError as error:
        raise ValueError(f"value must broadcast to output shape {shape}.") from error
