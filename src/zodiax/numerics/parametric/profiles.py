"""Coordinate-dependent numerical profiles."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as np
import jax.scipy.linalg as jspl
from jax import Array

from ..arrays import _as_inexact_array
from ..expressions import Expression, resolve
from ..transforms import (
    Transform,
    _initialise_transform,
    _resolved,
    _validate_operand,
)
from ._context import coordinate_axis, coordinate_context

__all__ = ["gaussian", "Gaussian"]


def _axis(value: Any, ndim: int, name: str) -> Array:
    """Return a scalar or trailing physical-axis vector of length ``ndim``."""
    value = _as_inexact_array(value, name)
    if value.ndim == 0:
        return np.broadcast_to(value, (ndim,))
    if value.shape[-1] == 1:
        return np.broadcast_to(value, value.shape[:-1] + (ndim,))
    if value.shape[-1] != ndim:
        raise ValueError(f"{name} must be scalar or end in an axis of size {ndim}.")
    return value


def _covariance(value: Any) -> tuple[Array, int]:
    """Return a square covariance and its physical dimensionality."""
    value = _as_inexact_array(value, "cov")
    if np.iscomplexobj(value):
        raise TypeError("cov must be real.")
    if value.ndim < 2 or value.shape[-2] != value.shape[-1] or value.shape[-1] < 1:
        raise ValueError("cov must end in a non-empty square matrix.")
    return value, value.shape[-1]


def gaussian(
    coords: Any,
    cov: Any,
    mean: Any = None,
    **context: Any,
) -> Array:
    """Evaluate a sampled-peak-normalised covariance Gaussian.

    The covariance determines the dimensionality ``D``. Coordinates must follow the
    same convention as :class:`~zodiax.numerics.grids.Grid`, with shape
    ``(..., D, *spatial_shape)`` and exactly ``D`` final spatial axes. Leading axes
    of the coordinate array, covariance, and mean broadcast normally. The returned
    profile has a sampled peak of one; it is not sum- or integral-normalised.
    """
    coords = _as_inexact_array(resolve(coords, **context), "coords")
    if np.iscomplexobj(coords):
        raise TypeError("coords must be real.")
    cov, ndim = _covariance(resolve(cov, **context))
    component_axis = coordinate_axis(coords, ndim)

    cov = eqx.error_if(
        cov,
        np.any(~np.isfinite(cov)) | np.any(~np.isclose(cov, np.swapaxes(cov, -1, -2))),
        "cov must be finite and symmetric.",
    )
    eigenvalues = np.linalg.eigvalsh(cov)
    cov = eqx.error_if(
        cov,
        np.any(eigenvalues <= 0),
        "cov must be positive definite.",
    )

    mean = resolve(mean, **context)
    mean = np.zeros((ndim,), dtype=coords.dtype) if mean is None else mean
    mean = _axis(mean, ndim, "mean")
    if np.iscomplexobj(mean):
        raise TypeError("mean must be real.")

    spatial_shape = coords.shape[-ndim:]
    sample_shape = (1,) * ndim
    mean = mean.reshape(mean.shape + sample_shape)
    offset = coords - mean
    points = np.moveaxis(offset, component_axis, -1)

    leading = np.broadcast_shapes(points.shape[:component_axis], cov.shape[:-2])
    points = np.broadcast_to(points, leading + spatial_shape + (ndim,))
    cholesky = np.linalg.cholesky(cov)
    cholesky = np.broadcast_to(cholesky, leading + (ndim, ndim))
    rhs = np.moveaxis(points, -1, -ndim - 1).reshape(leading + (ndim, -1))

    whitened = jspl.solve_triangular(cholesky, rhs, lower=True)
    exponent = np.square(whitened).sum(-2).reshape(leading + spatial_shape)
    axes = tuple(range(-ndim, 0))
    reference = exponent.min(axes, keepdims=True)
    return np.exp(0.5 * (reference - exponent))


class Gaussian(Transform):
    """A general covariance-defined Gaussian profile.

    ``x`` is an optional stored coordinate array. When it is absent, resolution
    reads ``coords`` from context. ``cov`` is the canonical representation;
    axis-aligned widths, correlation coefficients, and ellipse parameters belong in
    downstream convenience constructors or wrappers.
    """

    cov: Any = None
    mean: Any = None

    def __init__(
        self,
        x: Any = None,
        cov: Any = None,
        mean: Any = None,
        *,
        alias: Any = None,
    ):
        _initialise_transform(self, x, alias, cov=cov, mean=mean)

        if self.cov is not None and not isinstance(self.cov, Expression):
            _, ndim = _covariance(self.cov)
            if self.mean is not None and not isinstance(self.mean, Expression):
                _axis(self.mean, ndim, "mean")

    def evaluate(
        self,
        *,
        coords: Any = None,
        coordinates: Any = None,
        **context: Any,
    ) -> Array:
        """Evaluate stored coordinates or read ``coords`` from context."""
        coords, context = coordinate_context(
            coords, coordinates, context, required=self.x is None
        )
        if self.x is not None:
            return self.fwd(_resolved(self.x, context), **context)
        return self.fwd(coords, **context)

    def fwd(self, x: Any, **context: Any) -> Array:
        """Evaluate the profile at explicit coordinates ``x``."""
        x = _resolved(x, context)
        cov = _resolved(self.cov, context)
        if cov is None:
            raise ValueError("Gaussian cov must not resolve to None.")
        mean = _resolved(self.mean, context)
        return gaussian(x, cov, mean)

    def __zodiax_validate__(self) -> None:
        """Validate constructor-free archive reconstruction."""
        _validate_operand(self.cov, "cov", optional=True)
        _validate_operand(self.mean, "mean", optional=True)
        if self.cov is not None and not isinstance(self.cov, Expression):
            _, ndim = _covariance(self.cov)
            if self.mean is not None and not isinstance(self.mean, Expression):
                _axis(self.mean, ndim, "mean")
