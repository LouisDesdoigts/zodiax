"""Multi-axis explicit basis expansions."""

from __future__ import annotations

from math import prod
from operator import index as integer_index
from typing import Any

import equinox as eqx
import jax.numpy as np
from jax import Array

from ..arrays import _as_inexact_array
from ..expressions import Expression, resolve
from ..transforms import (
    Transform,
    _initialise_transform,
    _resolved,
    _validate_operand,
)

__all__ = [
    "basis",
    "Basis",
]


def _axis_count(axes: Any) -> int:
    """Return one positive static contraction-axis count."""
    if isinstance(axes, bool):
        raise TypeError("axes must be a positive integer.")
    try:
        axes = integer_index(axes)
    except TypeError as error:
        raise TypeError("axes must be a positive integer.") from error
    if axes < 1:
        raise ValueError("axes must be a positive integer.")
    return axes


def _validate_M(M: Array, axes: int | None) -> None:
    """Validate a concrete basis tensor independently of its coefficients."""
    if M.ndim == 0:
        raise ValueError("M must have at least one dimension.")
    if axes is not None and M.ndim < axes:
        raise ValueError(f"M must have at least {axes} dimensions.")
    if 0 in M.shape:
        raise ValueError("M must have nonzero dimensions.")


def _infer_axes(x: Array, M: Array) -> int:
    """Match one unique trailing coefficient shape to leading basis dimensions."""
    candidates = [
        axes
        for axes in range(1, min(x.ndim, M.ndim) + 1)
        if x.shape[-axes:] == M.shape[:axes]
    ]
    if not candidates:
        raise ValueError("No trailing x dimensions match the leading dimensions of M.")
    if len(candidates) > 1:
        raise ValueError(
            "Coefficient dimensions are ambiguous; provide the advanced axes "
            f"override. Matching contraction counts are {candidates}."
        )
    return candidates[0]


def _contraction_axes(x: Array, M: Array, axes: int | None) -> int:
    """Return explicit or shape-inferred contraction dimensionality."""
    return _infer_axes(x, M) if axes is None else _axis_count(axes)


def _validate_x(x: Array, M: Array, axes: int | None) -> int:
    """Validate concrete coefficient axes against a concrete basis tensor."""
    axes = _contraction_axes(x, M, axes)
    if x.ndim < axes or x.shape[-axes:] != M.shape[:axes]:
        shape = M.shape[:axes]
        raise ValueError(f"x must end in the coefficient shape {shape}.")
    return axes


def _contract(x: Any, M: Any, axes: int | None) -> Array:
    """Contract trailing coefficient axes against leading basis axes."""
    x = _as_inexact_array(x, "x")
    M = _as_inexact_array(M, "M")
    _validate_M(M, axes)
    axes = _validate_x(x, M, axes)
    x_axes = tuple(range(x.ndim - axes, x.ndim))
    M_axes = tuple(range(axes))
    return np.tensordot(x, M, axes=(x_axes, M_axes))


def basis(x: Any, M: Any, axes: int | None = None, **context: Any) -> Array:
    """Expand coefficients over one explicit or contextual basis tensor.

    By default, one unique trailing shape of ``x`` is matched against the leading
    dimensions of ``M``. ``axes`` is an advanced override for ambiguous shapes.
    Remaining leading ``x`` dimensions are batches and remaining trailing ``M``
    dimensions are the output shape.
    """
    x = resolve(x, **context)
    M = resolve(M, **context)
    if M is None:
        raise ValueError("M must not resolve to None.")
    return _contract(x, M, axes)


def _infer_solve_axes(y: Array, M: Array) -> int:
    """Infer coefficient dimensionality from one uniquely matching output shape."""
    candidates = []
    # A fully contracted multi-axis M has no output suffix from which to distinguish
    # coefficient axes from batches; that uncommon case requires explicit axes.
    stop = M.ndim + 1 if M.ndim == 1 else M.ndim
    for axes in range(1, stop):
        output_shape = M.shape[axes:]
        if not output_shape or (
            y.ndim >= len(output_shape)
            and y.shape[-len(output_shape) :] == output_shape
        ):
            candidates.append(axes)
    if not candidates:
        raise ValueError("No trailing y dimensions match an output shape of M.")
    if len(candidates) > 1:
        raise ValueError(
            "Coefficient dimensions are ambiguous while solving; provide the "
            f"advanced axes override. Matching contraction counts are {candidates}."
        )
    return candidates[0]


def _solve(y: Any, M: Any, axes: int | None) -> Array:
    """Return minimum-norm coefficients for a multi-axis basis expansion."""
    y = _as_inexact_array(y, "y")
    M = _as_inexact_array(M, "M")
    _validate_M(M, axes)
    axes = _infer_solve_axes(y, M) if axes is None else _axis_count(axes)

    coefficient_shape = M.shape[:axes]
    output_shape = M.shape[axes:]
    output_ndim = len(output_shape)
    if output_ndim and (y.ndim < output_ndim or y.shape[-output_ndim:] != output_shape):
        raise ValueError(f"y must end in the output shape {output_shape}.")

    batch_shape = y.shape if output_ndim == 0 else y.shape[:-output_ndim]
    coefficient_size = prod(coefficient_shape)
    output_size = prod(output_shape)
    dtype = np.result_type(y.dtype, M.dtype)
    solve_dtype = np.result_type(dtype, np.float32)
    matrix = M.reshape((coefficient_size, output_size)).astype(solve_dtype)
    values = y.reshape((-1, output_size)).astype(solve_dtype)
    x = np.linalg.lstsq(matrix.T, values.T, rcond=None)[0].T
    return x.reshape(batch_shape + coefficient_shape).astype(dtype)


class Basis(Transform):
    """Expand coefficient axes over an explicit or contextual basis tensor.

    By default, trailing coefficient dimensions of ``x`` are matched uniquely to
    leading dimensions of ``M``. Leading ``x`` dimensions are retained as batch
    dimensions, and remaining ``M`` dimensions form the sampled output shape.

    ``M`` may itself be an :class:`Expression`. Generated-and-materialised bases can
    therefore be composed directly, while genuinely matrix-free operations should
    remain specialised :class:`Transform` implementations.
    """

    M: Any = None
    axes: int | None = eqx.field(default=None, static=True)

    def __init__(
        self,
        x: Any = None,
        M: Any = None,
        axes: int | None = None,
        *,
        alias: Any = None,
    ):
        _initialise_transform(self, x, alias, M=M)
        self.axes = None if axes is None else _axis_count(axes)

        if self.M is not None and not isinstance(self.M, Expression):
            _validate_M(self.M, self.axes)
            if self.x is not None and not isinstance(self.x, Expression):
                _validate_x(self.x, self.M, self.axes)

    def fwd(self, x: Any, **context: Any) -> Array:
        """Expand explicitly supplied coefficients over the resolved basis."""
        return basis(x, self.M, self.axes, **context)

    def solve(self, y: Any, **context: Any) -> Array:
        """Return minimum-norm coefficient values representing ``y``."""
        y = _resolved(y, context)
        M = _resolved(self.M, context)
        if M is None:
            raise ValueError("Basis M must not resolve to None.")
        return _solve(y, M, self.axes)

    def __zodiax_validate__(self) -> None:
        """Validate constructor-free archive reconstruction."""
        _validate_operand(self.M, "M", optional=True)
        if self.axes is not None:
            axes = _axis_count(self.axes)
            if type(self.axes) is not int or self.axes != axes:
                raise TypeError("axes must be None or a positive Python integer.")
        if self.M is not None and not isinstance(self.M, Expression):
            _validate_M(self.M, self.axes)
            if self.x is not None and not isinstance(self.x, Expression):
                _validate_x(self.x, self.M, self.axes)
