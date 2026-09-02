"""Composable one-dimensional interpolation backed by Interpax."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from numbers import Real
from typing import Any

import equinox as eqx
import interpax as ipx
import jax.numpy as np
from jax import Array

from ..arrays import _as_inexact_array
from ..expressions import Expression, resolve
from ..transforms import Transform, _operand, _validate_operand

__all__ = ["interpolate", "Interpolation"]


def _method(value: Any) -> str:
    """Return a non-empty interpolation method name."""
    if not isinstance(value, str):
        raise TypeError("method must be a string.")
    value = value.strip().lower()
    if not value:
        raise ValueError("method must not be empty.")
    return value


def _extrap_value(value: Any) -> bool | float:
    """Return one scalar Interpax extrapolation setting."""
    if isinstance(value, bool):
        return value
    if not isinstance(value, Real):
        raise TypeError("extrap values must be booleans or real numbers.")
    value = float(value)
    if not isfinite(value):
        raise ValueError("numeric extrap values must be finite.")
    return value


def _extrap(value: Any) -> bool | float | tuple[bool | float, bool | float]:
    """Normalise scalar or lower/upper Interpax extrapolation behaviour."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        value = tuple(value)
        if len(value) != 2:
            raise ValueError("extrap sequences must contain lower and upper values.")
        return (_extrap_value(value[0]), _extrap_value(value[1]))
    return _extrap_value(value)


def _period(value: Any) -> float | None:
    """Return one positive periodic interval or ``None``."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("period must be a positive real number or None.")
    value = float(value)
    if not isfinite(value) or value <= 0:
        raise ValueError("period must be finite and positive.")
    return value


def _real_array(value: Any, name: str) -> Array:
    """Return a strongly typed real inexact array."""
    value = _as_inexact_array(value, name)
    if np.iscomplexobj(value):
        raise TypeError(f"{name} must be real.")
    return value


def _knots(value: Any) -> Array:
    """Return one validated, strictly increasing knot vector."""
    value = _real_array(value, "knots")
    if value.ndim != 1:
        raise ValueError("knots must be one-dimensional.")
    if value.size < 2:
        raise ValueError("knots must contain at least two values.")
    return eqx.error_if(
        value,
        np.any(~np.isfinite(value)) | np.any(np.diff(value) <= 0),
        "knots must be finite and strictly increasing.",
    )


def _data(knots: Any, values: Any) -> tuple[Array, Array]:
    """Resolve the physical Interpax data contract."""
    knots = _knots(knots)
    values = _as_inexact_array(values, "values")
    if values.ndim == 0 or values.shape[0] != knots.size:
        raise ValueError("values leading axis must match knots.")
    return knots, values


def _zero_extrap(value: Any) -> bool:
    """Return whether one canonical extrapolation policy is zero fill."""
    values = value if isinstance(value, tuple) else (value,)
    return all(not isinstance(item, bool) and item == 0.0 for item in values)


def interpolate(
    x: Any,
    knots: Any,
    values: Any,
    *,
    method: str = "linear",
    extrap: Any = 0.0,
    period: Any = None,
    **context: Any,
) -> Array:
    """Interpolate leading-axis samples at arbitrary-shaped query coordinates.

    Interpax consumes a one-dimensional query. Zodiax flattens ``x`` for that call
    and restores ``x.shape + values.shape[1:]`` afterwards. Knots are strictly
    increasing and values may retain any trailing value shape.
    """
    method = _method(method)
    extrap = _extrap(extrap)
    period = _period(period)
    x = _real_array(resolve(x, **context), "x")
    knots = resolve(knots, **context)
    values = resolve(values, **context)
    knots, values = _data(knots, values)

    shape = x.shape + values.shape[1:]
    output = ipx.interp1d(
        x.reshape(-1),
        knots,
        values,
        method=method,
        extrap=extrap,
        period=period,
    )
    return output.reshape(shape)


class Interpolation(Transform):
    """Interpolate sampled values along one strictly increasing coordinate axis.

    ``x`` is the query coordinate and may be stored, supplied explicitly, or
    produced by another :class:`~zodiax.numerics.Expression`. A ``StateRef`` stored
    in ``x`` therefore reads a time, index, or other query from explicit Zodiax
    state without giving interpolation a separate context-selection protocol.

    The leading axis of ``values`` corresponds to ``knots``. Remaining value axes
    are appended after the arbitrary query shape. Interpolation is forward-only;
    no general inverse or representative solve is defined. ``extrap`` follows the
    portable scalar subset of Interpax: ``False`` produces NaN outside the knots,
    ``True`` extrapolates, a finite number fills, and a pair configures the lower and
    upper sides independently.
    """

    knots: Any = None
    values: Any = None
    method: str = eqx.field(default="linear", static=True)
    extrap: bool | float | tuple[bool | float, bool | float] = eqx.field(
        default=0.0, static=True
    )
    period: float | None = eqx.field(default=None, static=True)

    def __init__(
        self,
        knots: Any,
        values: Any,
        x: Any = None,
        *,
        method: str = "linear",
        extrap: Any = 0.0,
        period: Any = None,
        alias: Any = None,
    ):
        self.alias = alias
        self.x = _operand(x, "x", optional=True)
        self.knots = _operand(knots, "knots")
        self.values = _operand(values, "values")
        self.method = _method(method)
        self.extrap = _extrap(extrap)
        self.period = _period(period)

        knots_expression = isinstance(self.knots, Expression)
        values_expression = isinstance(self.values, Expression)
        if not knots_expression and not values_expression:
            self.knots, self.values = _data(self.knots, self.values)
        else:
            if not knots_expression:
                self.knots = _knots(self.knots)
            if not values_expression and self.values.ndim == 0:
                raise ValueError("values must have a leading knot axis.")
        if self.x is not None and not isinstance(self.x, Expression):
            _real_array(self.x, "x")

    def fwd(self, x: Any, **context: Any) -> Array:
        """Interpolate at explicitly supplied query coordinates."""
        return interpolate(
            x,
            self.knots,
            self.values,
            method=self.method,
            extrap=self.extrap,
            period=self.period,
            **context,
        )

    def integrate(self, lower: Any, upper: Any, **context: Any) -> Array:
        """Exactly integrate scalar piecewise-linear values with zero fill.

        Bounds may be broadcast arrays. Higher-order methods, vector-valued samples,
        nonzero extrapolation, and periodic interpolation intentionally remain
        outside this exact integration contract.
        """
        if self.method != "linear":
            raise NotImplementedError(
                "Exact integration currently requires method='linear'."
            )
        if not _zero_extrap(self.extrap):
            raise NotImplementedError(
                "Exact integration currently requires zero extrapolation."
            )
        if self.period is not None:
            raise NotImplementedError(
                "Exact integration does not currently support a period."
            )

        knots = resolve(self.knots, **context)
        values = resolve(self.values, **context)
        knots, values = _data(knots, values)
        if values.ndim != 1:
            raise NotImplementedError(
                "Exact integration currently requires scalar sampled values."
            )

        lower = _real_array(resolve(lower, **context), "lower")
        upper = _real_array(resolve(upper, **context), "upper")
        lower = eqx.error_if(lower, np.any(np.isnan(lower)), "lower must not be NaN.")
        upper = eqx.error_if(upper, np.any(np.isnan(upper)), "upper must not be NaN.")
        lower, upper = np.broadcast_arrays(lower, upper)
        sign = np.where(upper < lower, -1, 1)
        lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
        x0, x1 = knots[:-1], knots[1:]
        y0, y1 = values[:-1], values[1:]
        lower = lower[..., None]
        upper = upper[..., None]
        start = np.maximum(lower, x0)
        stop = np.minimum(upper, x1)
        active = stop > start
        start = np.where(active, start, x0)
        stop = np.where(active, stop, x0)
        slope = (y1 - y0) / (x1 - x0)
        integral = y0 * (stop - start)
        integral += slope * ((stop - x0) ** 2 - (start - x0) ** 2) / 2
        return sign * np.where(active, integral, 0.0).sum(-1)

    def __zodiax_validate__(self) -> None:
        """Validate constructor-free archive reconstruction."""
        _validate_operand(self.x, "x", optional=True)
        _validate_operand(self.knots, "knots")
        _validate_operand(self.values, "values")
        if self.x is not None and not isinstance(self.x, Expression):
            _real_array(self.x, "x")
        if not isinstance(self.knots, Expression) and not isinstance(
            self.values, Expression
        ):
            _data(self.knots, self.values)
        elif not isinstance(self.knots, Expression):
            _knots(self.knots)
        elif not isinstance(self.values, Expression) and self.values.ndim == 0:
            raise ValueError("values must have a leading knot axis.")
        if self.method != _method(self.method):
            raise ValueError("method must be stored canonically.")
        if self.extrap != _extrap(self.extrap):
            raise ValueError("extrap must be stored canonically.")
        if self.period != _period(self.period):
            raise ValueError("period must be stored canonically.")
