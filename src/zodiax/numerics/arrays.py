"""Array conversion shared by Zodiax numerical definitions."""

from __future__ import annotations

from typing import Any

import jax.numpy as np
from jax import Array

from .expressions import Expression

__all__ = ["as_array"]


def _as_array(value: Any, name: str = "value", *, dtype: Any = None) -> Array:
    """Return a strongly typed JAX array while preserving its inferred dtype."""
    try:
        array = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be array-like.") from error
    return np.asarray(array, dtype=array.dtype)


def _as_inexact_array(
    value: Any,
    name: str = "value",
    *,
    dtype: Any = None,
) -> Array:
    """Return a strongly typed floating or complex JAX array."""
    array = _as_array(value, name, dtype=dtype)

    if not np.issubdtype(array.dtype, np.inexact):
        if dtype is not None:
            raise TypeError("dtype must be floating or complex.")
        array = np.asarray(array, dtype=float)

    return array


def _as_float_array(value: Any, name: str = "value") -> Array:
    """Return a strongly typed floating JAX array without accepting complex data."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be floating and array-like.") from error
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must have a floating dtype.")
    return np.asarray(array, dtype=array.dtype)


def _validate_inexact_array(value: Any, name: str) -> None:
    """Validate constructor-free storage of an inexact array."""
    if not isinstance(value, Array):
        raise TypeError(f"{name} must be a JAX array.")
    if bool(value.weak_type) or not np.issubdtype(value.dtype, np.inexact):
        raise TypeError(f"{name} must be a strongly typed inexact JAX array.")


def _validate_float_array(value: Any, name: str) -> None:
    """Validate constructor-free storage of a floating array."""
    if not isinstance(value, Array):
        raise TypeError(f"{name} must be a JAX array.")
    if bool(value.weak_type) or not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"{name} must be a strongly typed floating JAX array.")


def as_array(value: Any, *, dtype: Any = None) -> Array | Expression | None:
    """Coerce an array value while preserving expressions and absent values.

    Python scalars, NumPy values, and array-like sequences become strongly typed JAX
    arrays with their inferred dtype. ``None`` and :class:`Expression` definitions pass
    through unchanged, which makes this suitable at public model-constructor
    boundaries without every domain library reimplementing the same dispatch.

    Parameters
    ----------
    value
        An array-like value, ``None``, or a numerical expression.
    dtype
        Optional dtype. Existing or inferred dtype is retained when omitted.
    """
    if value is None or isinstance(value, Expression):
        return value
    return _as_array(value, dtype=dtype)
