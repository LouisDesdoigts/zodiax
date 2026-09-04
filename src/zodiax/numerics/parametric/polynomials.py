"""Polynomial basis generators and contextual polynomial transforms."""

from __future__ import annotations

from itertools import product
from math import prod as shape_product
from operator import index as integer_index
from typing import Any

import equinox as eqx
import jax.numpy as np
from jax import Array, vmap

from ..arrays import _as_array, _as_inexact_array
from ..expressions import Expression, resolve
from ..transforms import (
    Transform,
    _initialise_transform,
    _resolved,
)
from ._context import coordinate_axis, coordinate_context

__all__ = ["polynomial_powers", "Polynomial"]


def _integer(value: Any, name: str, *, minimum: int) -> int:
    """Return one bounded Python integer without silently truncating values."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    try:
        value = integer_index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer.") from error
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def polynomial_powers(degree: int, ndim: int = 1) -> Array:
    """Return multivariate exponents through one maximum total degree.

    Columns enumerate terms in increasing total degree. Within each total degree,
    the first variable's exponent decreases first, matching the established dLux
    polynomial convention.
    """
    degree = _integer(degree, "degree", minimum=0)
    ndim = _integer(ndim, "ndim", minimum=1)
    powers = [
        values
        for total in range(degree + 1)
        for values in product(range(total, -1, -1), repeat=ndim)
        if sum(values) == total
    ]
    return np.asarray(powers, dtype=int).T


def _powers_array(p: Any) -> Array:
    """Return a nonnegative integer ``(n_variables, n_terms)`` exponent array."""
    p = _as_array(p, "p")
    if not np.issubdtype(p.dtype, np.integer):
        raise TypeError("p must have an integer dtype.")
    p = p[None, :] if p.ndim == 1 else p
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] == 0:
        raise ValueError("p must have shape (n_variables, n_terms).")
    return eqx.error_if(p, np.any(p < 0), "p must be nonnegative.")


def _validate_powers_array(p: Any) -> None:
    """Validate constructor-free storage of a monomial exponent array."""
    if not isinstance(p, Array):
        raise TypeError("p must be a JAX array.")
    if bool(p.weak_type) or not np.issubdtype(p.dtype, np.integer):
        raise TypeError("p must be a strongly typed integer JAX array.")
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] == 0:
        raise ValueError("p must have shape (n_variables, n_terms).")
    if bool(np.any(p < 0)):
        raise ValueError("p must be nonnegative.")


def _polynomial_powers(
    degree: Any,
    ndim: Any,
    p: Any,
    degrees: Any,
) -> Array:
    """Resolve explicit, maximum-degree, or selected-degree term definitions."""
    if p is not None:
        if degree is not None or degrees is not None:
            raise ValueError("p is mutually exclusive with degree and degrees.")
        return _powers_array(p)
    if degree is not None and degrees is not None:
        raise ValueError("Provide only one of degree or degrees.")
    if degree is None and degrees is None:
        raise ValueError("Provide one of degree, degrees, or p.")

    ndim = _integer(ndim, "ndim", minimum=1)
    if degrees is None:
        return polynomial_powers(degree, ndim)

    degrees = _as_array(degrees, "degrees")
    if not np.issubdtype(degrees.dtype, np.integer):
        raise TypeError("degrees must have an integer dtype.")
    degrees = np.atleast_1d(degrees)
    if degrees.ndim != 1 or degrees.size == 0:
        raise ValueError("degrees must contain at least one total degree.")
    if bool(np.any(degrees < 0)):
        raise ValueError("degrees must be nonnegative.")
    powers = polynomial_powers(int(degrees.max()), ndim)
    return powers[:, np.isin(powers.sum(0), degrees)]


def _term_layout(x: Any, p: Any, **context: Any) -> tuple[Array, int]:
    """Evaluate one or more multivariate monomial terms on coordinates.

    Sampled coordinates use ``(..., D, *spatial_shape)`` with exactly ``D`` final
    spatial axes. The inferred component axis is replaced by a term axis in the
    result. A bare array remains a shorthand for arbitrary one-variable samples.
    """
    x = _as_inexact_array(resolve(x, **context), "x")
    p = _powers_array(p)
    n_variables = p.shape[0]

    # Preserve the established scalar-polynomial shorthand used by spectral
    # consumers. A Grid-style 1-D field is unambiguous from its ``(..., 1, n)``
    # shape and follows the standard paired-batch convention below.
    if n_variables == 1 and (x.ndim < 2 or x.shape[-2] != 1):
        shape = p.shape + (1,) * x.ndim
        return np.prod(x[None, None, ...] ** p.reshape(shape), axis=0), x.ndim

    coordinate_axis(x, n_variables)
    leading_ndim = x.ndim - n_variables - 1
    shape = (1,) * leading_ndim + p.shape + (1,) * n_variables
    terms = np.prod(
        np.expand_dims(x, leading_ndim + 1) ** p.reshape(shape),
        axis=leading_ndim,
    )
    return terms, n_variables


def _polynomial_terms(x: Any, p: Any, **context: Any) -> Array:
    """Return sampled monomial terms using the shared coordinate convention."""
    return _term_layout(x, p, **context)[0]


def _expand(x: Any, terms: Array, spatial_ndim: int, context: dict[str, Any]) -> Array:
    """Contract terms with coefficients using paired leading-axis broadcasting."""
    x = _as_inexact_array(resolve(x, **context), "x")
    term_axis = terms.ndim - spatial_ndim - 1
    n_terms = terms.shape[term_axis]
    if x.ndim == 0 or x.shape[-1] != n_terms:
        raise ValueError(f"x must have a final coefficient axis of size {n_terms}.")

    leading = np.broadcast_shapes(x.shape[:-1], terms.shape[:term_axis])
    spatial_shape = terms.shape[-spatial_ndim:] if spatial_ndim else ()
    x = np.broadcast_to(x, leading + (n_terms,))
    x = x.reshape(leading + (n_terms,) + (1,) * spatial_ndim)
    terms = np.broadcast_to(terms, leading + (n_terms,) + spatial_shape)
    return np.sum(x * terms, axis=len(leading))


def _solve_terms(y: Any, terms: Array, spatial_ndim: int) -> Array:
    """Solve paired batches of sampled polynomial terms for coefficients."""
    y = _as_inexact_array(y, "y")
    term_axis = terms.ndim - spatial_ndim - 1
    n_terms = terms.shape[term_axis]
    spatial_shape = terms.shape[-spatial_ndim:] if spatial_ndim else ()
    if spatial_ndim and (
        y.ndim < spatial_ndim or y.shape[-spatial_ndim:] != spatial_shape
    ):
        raise ValueError(f"y must end in the coordinate shape {spatial_shape}.")

    y_leading = y.shape[:-spatial_ndim] if spatial_ndim else y.shape
    leading = np.broadcast_shapes(y_leading, terms.shape[:term_axis])
    terms = np.broadcast_to(terms, leading + (n_terms,) + spatial_shape)
    y = np.broadcast_to(y, leading + spatial_shape)

    sample_size = shape_product(spatial_shape)
    dtype = np.result_type(y.dtype, terms.dtype)
    solve_dtype = np.result_type(dtype, np.float32)
    matrices = terms.reshape((-1, n_terms, sample_size)).astype(solve_dtype)
    values = y.reshape((-1, sample_size)).astype(solve_dtype)

    def solve_one(matrix: Array, value: Array) -> Array:
        return np.linalg.lstsq(matrix.T, value, rcond=None)[0]

    coefficients = vmap(solve_one)(matrices, values)
    return coefficients.reshape(leading + (n_terms,)).astype(dtype)


class Polynomial(Transform):
    """Expand coefficients over polynomial terms generated from ``coords``.

    A ``D``-variable sampled coordinate field follows ``Grid`` shape
    ``(..., D, *spatial_shape)`` with ``D`` trailing spatial axes. Leading
    coordinate and coefficient axes use paired broadcasting. Bare sample arrays are
    also accepted for one-variable polynomials.
    """

    p: Array | None = None

    def __init__(
        self,
        x: Any = None,
        degree: Any = None,
        ndim: int = 1,
        p: Any = None,
        degrees: Any = None,
        *,
        alias: Any = None,
    ):
        p = _polynomial_powers(degree, ndim, p, degrees)
        _initialise_transform(self, x, alias)
        self.p = p

        if self.x is not None and not isinstance(self.x, Expression):
            if self.x.ndim == 0 or self.x.shape[-1] != p.shape[1]:
                raise ValueError(
                    f"x must have a final coefficient axis of size {p.shape[1]}."
                )

    def fwd(
        self,
        x: Any,
        *,
        coords: Any = None,
        coordinates: Any = None,
        **context: Any,
    ) -> Array:
        """Evaluate polynomial terms and expand explicitly supplied coefficients."""
        coords, context = coordinate_context(
            coords, coordinates, context, required=True
        )
        terms, spatial_ndim = _term_layout(coords, self.p, **context)
        return _expand(x, terms, spatial_ndim, context)

    def solve(
        self,
        y: Any,
        *,
        coords: Any = None,
        coordinates: Any = None,
        **context: Any,
    ) -> Array:
        """Return minimum-norm coefficients for values at ``coords``."""
        coords, context = coordinate_context(
            coords, coordinates, context, required=True
        )
        terms, spatial_ndim = _term_layout(coords, self.p, **context)
        return _solve_terms(_resolved(y, context), terms, spatial_ndim)

    def __zodiax_validate__(self) -> None:
        """Validate stored polynomial powers and optional coefficients."""
        _validate_powers_array(self.p)
        if self.x is not None and not isinstance(self.x, Expression):
            if self.x.ndim == 0 or self.x.shape[-1] != self.p.shape[1]:
                raise ValueError(
                    "Polynomial x must end in the stored monomial term count."
                )
