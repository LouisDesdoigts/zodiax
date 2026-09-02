"""Composable array normalisation transforms."""

from __future__ import annotations

from operator import index as integer_index
from typing import Any

import equinox as eqx
import jax.numpy as np
from jax import Array

from .arrays import _as_inexact_array
from .transforms import Transform, _operand, _pdoc, _resolved, _validate_operand

__all__ = ["Norm", "MeanNorm", "RMSNorm", "SumNorm"]

_MODES = ("mean", "rms", "sum")


def _mode(value: Any) -> str:
    """Return one supported normalisation measure name."""
    if not isinstance(value, str):
        raise TypeError("mode must be a string.")
    value = value.strip().lower()
    if value not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}.")
    return value


def _axis(value: Any) -> tuple[int, ...] | None:
    """Return canonical static reduction axes or ``None`` for the whole array."""
    if value is None:
        return None
    values = value if isinstance(value, (tuple, list)) else (value,)
    if not values:
        raise ValueError("axis must contain at least one index.")

    axes = []
    for item in values:
        if isinstance(item, bool):
            raise TypeError("axis must contain integers.")
        try:
            item = integer_index(item)
        except TypeError as error:
            raise TypeError("axis must contain integers.") from error
        axes.append(item)
    if len(axes) != len(set(axes)):
        raise ValueError("axis must not contain duplicate indices.")
    return tuple(axes)


def _axes(axis: tuple[int, ...] | None, ndim: int) -> tuple[int, ...] | None:
    """Validate and canonicalise negative axes for one realised array rank."""
    if axis is None:
        return None
    axes = tuple(value + ndim if value < 0 else value for value in axis)
    if any(value < 0 or value >= ndim for value in axes):
        raise ValueError(f"axis entries must lie in [-{ndim}, {ndim - 1}].")
    if len(axes) != len(set(axes)):
        raise ValueError("axis entries must identify unique dimensions.")
    return axes


def _weights(x: Array, value: Any, context: dict[str, Any]) -> Array | None:
    """Resolve nonnegative real weights broadcastable to ``x.shape``."""
    value = _resolved(value, context)
    if value is None:
        return None
    value = _as_inexact_array(value, "w")
    if np.iscomplexobj(value):
        raise TypeError("w must be real.")
    try:
        value = np.broadcast_to(value, x.shape)
    except ValueError as error:
        raise ValueError("w must broadcast to x.shape.") from error
    return eqx.error_if(value, np.any(value < 0), "w must be nonnegative.")


def _measure(
    x: Array,
    mode: str,
    w: Array | None,
    axis: tuple[int, ...] | None,
) -> Array:
    """Return a keep-dimensional weighted normalisation measure."""
    axes = _axes(axis, x.ndim)
    values = x if w is None else x * w

    if mode == "sum":
        return np.sum(values, axis=axes, keepdims=True)

    if w is None:
        shape = x.shape if axes is None else tuple(x.shape[i] for i in axes)
        count = np.prod(np.asarray(shape))
    else:
        count = np.sum(w, axis=axes, keepdims=True)
        count = eqx.error_if(
            count,
            np.any(count == 0),
            "Normalisation weights must have positive total weight.",
        )

    if mode == "mean":
        return np.sum(values, axis=axes, keepdims=True) / count

    squares = np.abs(x) ** 2
    if w is not None:
        squares = squares * w
    return np.sqrt(np.sum(squares, axis=axes, keepdims=True) / count)


class Norm(Transform):
    """Scale ``x`` so its selected mean, RMS, or sum equals ``s``.

    ``axis=None`` measures the complete array and is omitted from the printed
    representation. ``w`` supplies optional nonnegative weights or a support mask.
    ``s=None`` means a target value of one.
    """

    s: Any = None
    w: Any = None
    mode: str | None = eqx.field(default=None, static=True)
    axis: tuple[int, ...] | None = eqx.field(default=None, static=True)

    def __init__(
        self,
        mode: str,
        x: Any = None,
        s: Any = None,
        w: Any = None,
        axis: Any = None,
        *,
        alias: Any = None,
    ):
        self.alias = alias
        self.x = _operand(x, "x", optional=True)
        self.s = _operand(s, "s", optional=True)
        self.w = _operand(w, "w", optional=True)
        self.mode = _mode(mode)
        self.axis = _axis(axis)

    def fwd(self, x: Any, **context: Any) -> Array:
        """Normalise an explicitly supplied array."""
        x = _as_inexact_array(_resolved(x, context), "x")
        w = _weights(x, self.w, context)
        measure = _measure(x, self.mode, w, self.axis)
        measure = eqx.error_if(
            measure,
            np.any(np.abs(measure) == 0),
            f"{self.mode} normalisation measure must be nonzero.",
        )
        s = _resolved(self.s, context)
        s = np.asarray(1, dtype=x.dtype) if s is None else _as_inexact_array(s, "s")
        y = x * s / measure
        if y.shape != x.shape:
            raise ValueError("s must broadcast without changing x.shape.")
        return y

    def __pdoc__(self, **kwargs: Any) -> Any:
        fields = [
            (name, value)
            for name, value in (("x", self.x), ("s", self.s), ("w", self.w))
            if value is not None
        ]
        if type(self) is Norm:
            fields.append(("mode", self.mode))
        if self.axis is not None:
            fields.append(("axis", self.axis))
        return _pdoc(self, type(self).__name__, fields, **kwargs)

    def __zodiax_validate__(self) -> None:
        _validate_operand(self.x, "x", optional=True)
        _validate_operand(self.s, "s", optional=True)
        _validate_operand(self.w, "w", optional=True)
        if self.mode != _mode(self.mode):
            raise ValueError("mode must use its canonical spelling.")
        if self.axis != _axis(self.axis):
            raise ValueError("axis must use its canonical tuple representation.")


class MeanNorm(Norm):
    """Scale ``x`` to a target arithmetic mean."""

    def __init__(
        self,
        x: Any = None,
        s: Any = None,
        w: Any = None,
        axis: Any = None,
        *,
        alias: Any = None,
    ):
        super().__init__("mean", x=x, s=s, w=w, axis=axis, alias=alias)


class RMSNorm(Norm):
    """Scale ``x`` to a target root-mean-square value."""

    def __init__(
        self,
        x: Any = None,
        s: Any = None,
        w: Any = None,
        axis: Any = None,
        *,
        alias: Any = None,
    ):
        super().__init__("rms", x=x, s=s, w=w, axis=axis, alias=alias)


class SumNorm(Norm):
    """Scale ``x`` to a target sum."""

    def __init__(
        self,
        x: Any = None,
        s: Any = None,
        w: Any = None,
        axis: Any = None,
        *,
        alias: Any = None,
    ):
        super().__init__("sum", x=x, s=s, w=w, axis=axis, alias=alias)
