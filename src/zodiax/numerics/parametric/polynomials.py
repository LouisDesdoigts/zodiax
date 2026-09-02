"""Polynomial basis generators and contextual polynomial transforms."""

from __future__ import annotations

from itertools import product
from operator import index as integer_index
from typing import Any

import equinox as eqx
import jax.numpy as np
from jax import Array

from ..arrays import _as_array, _as_inexact_array
from ..expressions import Expression, resolve
from ..transforms import Transform, _operand, _resolved, _validate_operand
from ._context import coordinate_context
from .bases import _solve, basis

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


def _polynomial_terms(x: Any, p: Any, **context: Any) -> Array:
    """Evaluate one or more multivariate monomial terms on coordinates.

    For one variable, ``x`` has arbitrary sample shape. For several variables it
    has a leading variable axis followed by arbitrary sample axes. The returned
    leading axis enumerates terms.
    """
    x = _as_inexact_array(resolve(x, **context), "x")
    p = _powers_array(p)
    n_variables = p.shape[0]

    if n_variables == 1:
        x = x[None, ...]
    if x.ndim == 0 or x.shape[0] != n_variables:
        raise ValueError("coords leading axis must match the number of variables.")

    shape = p.shape + (1,) * (x.ndim - 1)
    return np.prod(x[:, None] ** p.reshape(shape), axis=0)


class Polynomial(Transform):
    """Expand coefficients over polynomial terms generated from ``coords``."""

    p: Array | None = None

    def __init__(
        self,
        degree: Any = None,
        x: Any = None,
        ndim: int = 1,
        p: Any = None,
        degrees: Any = None,
        *,
        alias: Any = None,
    ):
        p = _polynomial_powers(degree, ndim, p, degrees)
        self.alias = alias
        self.x = _operand(x, "x", optional=True)
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
        terms = _polynomial_terms(coords, self.p, **context)
        return basis(x, terms, axes=1, **context)

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
        terms = _polynomial_terms(coords, self.p, **context)
        return _solve(_resolved(y, context), terms, axes=1)

    def __zodiax_validate__(self) -> None:
        """Validate stored polynomial powers and optional coefficients."""
        _validate_operand(self.x, "x", optional=True)
        _validate_powers_array(self.p)
        if self.x is not None and not isinstance(self.x, Expression):
            if self.x.ndim == 0 or self.x.shape[-1] != self.p.shape[1]:
                raise ValueError(
                    "Polynomial x must end in the stored monomial term count."
                )
