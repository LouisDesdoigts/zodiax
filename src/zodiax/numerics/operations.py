"""Elementwise numerical operations with ordinary JAX broadcasting."""

from __future__ import annotations

from typing import Any

import jax.numpy as np
from jax import Array

from .arrays import as_array
from .expressions import resolve
from .transforms import Transform

__all__ = ["Operation", "Pow", "Exp", "Log"]


class Operation(Transform):
    """An elementwise transform accepting scalar, vector, or grid inputs.

    Subclasses implement their calculation in ``fwd`` using ordinary JAX
    broadcasting. Array-valued operands may expand the output shape; no axis is
    reserved for coefficients or a particular sampling layout. Calling, partial
    resolution, composition, and reverse methods are inherited from Transform.
    Reverse methods invert values elementwise; they do not undo shape expansion
    from broadcasting.
    """


class Pow(Operation):
    """Raise an input to a power: ``y = x**p``.

    General powers are intentionally forward-only because a globally valid real
    inverse requires domain and branch choices.
    """

    p: Any = None

    def __init__(self, x: Any = None, p: Any = None, *, alias: Any = None):
        """Store optional input, required power, and parameter aliases."""
        if p is None:
            raise ValueError("p must not be None.")
        self.alias = alias
        self.x = as_array(x)
        self.p = as_array(p)

    def fwd(self, x: Array, **context: Any) -> Array:
        p = resolve(self.p, **context)
        return np.power(x, p)


class Exp(Operation):
    """Exponentiate an input: ``y = exp(x)`` or ``base**x``.

    ``base=None`` selects the natural exponential. An explicit base must be real,
    positive, and unequal to one. ``inv`` uses the corresponding principal
    logarithm. It is exact for real inputs and positive outputs, and branch-local
    rather than globally one-to-one for complex values.
    """

    base: Any = None

    def __init__(
        self,
        x: Any = None,
        base: Any = None,
        *,
        alias: Any = None,
    ):
        """Store optional input and base arrays or expressions, with aliases."""
        self.alias = alias
        self.x = as_array(x)
        self.base = as_array(base)

    def fwd(self, x: Array, **context: Any) -> Array:
        base = resolve(self.base, **context)
        if base is None:
            return np.exp(x)
        return np.power(base, x)

    def inv(self, y: Array, **context: Any) -> Array:
        base = resolve(self.base, **context)
        if base is None:
            return np.log(y)
        return np.log(y) / np.log(base)


class Log(Operation):
    """Apply a natural or selected-base principal logarithm.

    ``base=None`` selects the natural logarithm. An explicit base must be real,
    positive, and unequal to one. ``inv`` is exact on the selected logarithm branch;
    complex logarithms do not define a globally one-to-one transform.
    """

    base: Any = None

    def __init__(
        self,
        x: Any = None,
        base: Any = None,
        *,
        alias: Any = None,
    ):
        """Store optional input and base arrays or expressions, with aliases."""
        self.alias = alias
        self.x = as_array(x)
        self.base = as_array(base)

    def fwd(self, x: Array, **context: Any) -> Array:
        base = resolve(self.base, **context)
        if base is None:
            return np.log(x)
        return np.log(x) / np.log(base)

    def inv(self, y: Array, **context: Any) -> Array:
        base = resolve(self.base, **context)
        if base is None:
            return np.exp(y)
        return np.power(base, y)
