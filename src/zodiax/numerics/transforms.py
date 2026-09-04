"""Composable numerical expressions and transforms.

The public vocabulary follows the usual numerical-transform convention:

``x``
    Input to a transform and the stored upstream expression.
``y``
    Output of a transform. It appears in equations and reverse-method arguments,
    but is never a stored field.
``b``, ``s``, ``M``, ``p``, ``base``
    Additive bias, multiplicative scale, matrix or sampled basis, power, and
    logarithmic base respectively.

Every transform accepts an explicit input and may store one under ``x``. Leaving
``x=None`` creates an unbound symbolic transform; storing another expression under
``x`` obtains the input from that expression. ``Map`` is the compact fixed-order
composition ``y = matmul(s * x, M) + b`` with each of ``s``, ``M``, and ``b``
optional.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from hashlib import blake2s
from math import prod
from typing import Any

import equinox as eqx
import jax.numpy as np
import numpy as onp
import wadler_lindig as wl
from jax import Array

from .arrays import _as_inexact_array, _validate_inexact_array
from .expressions import Expression, resolve

__all__ = [
    "Transform",
    "add",
    "mul",
    "power",
    "matmul",
    "Add",
    "Mul",
    "Pow",
    "Exp",
    "Log",
    "MatMul",
    "Mask",
    "Map",
]

_NO_INPUT = object()


def _resolved(value: Any, context: dict[str, Any]) -> Any:
    """Resolve a possibly deferred operand in the current evaluation context."""
    return resolve(value, **context)


def _operand(value: Any, name: str, *, optional: bool = False) -> Any:
    """Normalise an array-or-expression operand without realising expressions."""
    if value is None:
        if optional:
            return None
        raise ValueError(f"{name} must not be None.")
    if isinstance(value, Expression):
        return value
    return _as_inexact_array(value, name)


def _initialise_transform(
    owner: Any,
    x: Any,
    alias: Any,
    *,
    required: Sequence[str] = (),
    **operands: Any,
) -> None:
    """Initialise the common transform input, alias, and numerical operands."""
    owner.alias = alias
    owner.x = _operand(x, "x", optional=True)
    required = frozenset(required)
    for name, value in operands.items():
        setattr(owner, name, _operand(value, name, optional=name not in required))


def _validate_operand(value: Any, name: str, *, optional: bool = False) -> None:
    """Validate constructor-free array-or-expression storage."""
    if value is None and optional:
        return
    if isinstance(value, Expression):
        return
    _validate_inexact_array(value, name)


def _preserve_shape(x: Array, y: Array, name: str) -> Array:
    """Require a broadcast operation to retain its input shape."""
    if y.shape != x.shape:
        raise ValueError(
            f"{name} must broadcast without changing x.shape; received "
            f"{x.shape} -> {y.shape}."
        )
    return y


def _nonzero(value: Any, name: str) -> Array:
    """Dynamically enforce a nonzero inverse operand."""
    value = _as_inexact_array(value, name)
    return eqx.error_if(value, np.any(value == 0), f"{name} must be nonzero.")


def _base(value: Any, context: dict[str, Any]) -> Array | None:
    """Resolve one real logarithmic base, where ``None`` denotes Euler's number."""
    value = _resolved(value, context)
    if value is None:
        return None
    value = _as_inexact_array(value, "base")
    if np.iscomplexobj(value):
        raise TypeError("base must be real.")
    return eqx.error_if(
        value,
        np.any(value <= 0) | np.any(value == 1),
        "base must be positive and not equal to one.",
    )


def add(x: Any, b: Any = None, **context: Any) -> Array:
    """Return ``x + b`` while preserving ``x.shape``; ``b=None`` is identity."""
    x = _as_inexact_array(_resolved(x, context), "x")
    b = _resolved(b, context)
    if b is None:
        return x
    y = x + _as_inexact_array(b, "b")
    return _preserve_shape(x, y, "b")


def mul(x: Any, s: Any = None, **context: Any) -> Array:
    """Return ``x * s`` while preserving ``x.shape``; ``s=None`` is identity."""
    x = _as_inexact_array(_resolved(x, context), "x")
    s = _resolved(s, context)
    if s is None:
        return x
    y = x * _as_inexact_array(s, "s")
    return _preserve_shape(x, y, "s")


def power(x: Any, p: Any, **context: Any) -> Array:
    """Return ``x**p`` using ordinary JAX broadcasting."""
    x = _as_inexact_array(_resolved(x, context), "x")
    p = _as_inexact_array(_resolved(p, context), "p")
    return np.power(x, p)


def matmul(x: Any, M: Any = None, **context: Any) -> Array:
    """Contract the final axis of ``x`` with the first axis of ``M``.

    For ``M.shape == (input_size, *output_shape)``, an input with shape
    ``(*batch_shape, input_size)`` produces ``(*batch_shape, *output_shape)``.
    A two-dimensional ``M`` therefore has row-vector ``x @ M`` semantics, while a
    higher-rank ``M`` represents a sampled basis with native output axes. ``None``
    is the identity.
    """
    x = _as_inexact_array(_resolved(x, context), "x")
    M = _resolved(M, context)
    if M is None:
        return x
    M = _as_inexact_array(M, "M")
    if M.ndim == 0:
        raise ValueError("M must have at least one axis; use Mul for scalar scaling.")
    if 0 in M.shape:
        raise ValueError("M must have nonzero dimensions.")
    if x.ndim == 0 or x.shape[-1] != M.shape[0]:
        raise ValueError(f"x must have a final axis of size {M.shape[0]}.")
    return np.tensordot(x, M, axes=((-1,), (0,)))


def _solve_matmul(y: Any, M: Any) -> Array:
    """Return a minimum-norm input for :func:`matmul`."""
    y = _as_inexact_array(y, "y")
    M = _as_inexact_array(M, "M")
    if M.ndim == 0:
        raise ValueError("M must have at least one axis.")
    if 0 in M.shape:
        raise ValueError("M must have nonzero dimensions.")

    output_shape = M.shape[1:]
    ndim = len(output_shape)
    if ndim and (y.ndim < ndim or y.shape[-ndim:] != output_shape):
        raise ValueError(f"y must end in shape {output_shape}.")

    batch_shape = y.shape if ndim == 0 else y.shape[:-ndim]
    width = prod(output_shape)
    dtype = np.result_type(y.dtype, M.dtype)
    solve_dtype = np.result_type(dtype, np.float32)
    matrix = M.reshape((M.shape[0], width)).astype(solve_dtype)
    flat = y.reshape((-1, width)).astype(solve_dtype)
    x = np.linalg.lstsq(matrix.T, flat.T, rcond=None)[0].T
    return x.reshape(batch_shape + (M.shape[0],)).astype(dtype)


def _pdoc(
    owner: Any,
    name: str,
    fields: Sequence[tuple[str, Any]],
    **kwargs: Any,
) -> Any:
    """Build an Equinox-compatible compact document including a real alias."""
    alias = getattr(owner, "alias", None)
    if alias is not None:
        fields = (("alias", alias), *fields)
    return wl.bracketed(
        begin=wl.TextDoc(f"{name}("),
        docs=wl.named_objs(fields, **kwargs),
        sep=wl.comma,
        end=wl.TextDoc(")"),
        indent=kwargs["indent"],
    )


class Transform(Expression):
    """A stored-input numerical transform.

    ``x`` is always the stored upstream input and ``fwd(x)`` always returns the
    output ``y``. A transform with ``x=None`` is unbound, while any Expression may
    provide its stored input. Specialised coordinate transforms may override
    :meth:`evaluate` to consume named coordinates. Exact transforms implement
    :meth:`inv`; projections or rank-deficient transforms implement :meth:`solve`.
    """

    x: Any = None

    @abstractmethod
    def fwd(self, x: Any, **context: Any) -> Array:
        """Map an explicitly supplied input ``x`` to an output ``y``."""

    def __zodiax_validate__(self) -> None:
        """Validate the common optional stored input after archive loading."""
        _validate_operand(self.x, "x", optional=True)

    def evaluate(self, **context: Any) -> Array:
        """Evaluate this transform from its stored upstream input."""
        if self.x is None:
            raise ValueError(f"{type(self).__name__} has no stored x input.")
        y = self.fwd(_resolved(self.x, context), **context)
        return _as_inexact_array(y, "y")

    def __call__(self, x: Any = _NO_INPUT, **context: Any) -> Array:
        """Resolve stored input, or apply the transform to explicit input ``x``."""
        if x is _NO_INPUT:
            return _as_inexact_array(resolve(self, **context), "y")
        return self.apply(x, **context)

    def apply(self, x: Any, **context: Any) -> Array:
        """Apply the complete transform spine to an explicit deepest input."""
        return _as_inexact_array(self.decode(x, **context), "y")

    def inv(self, y: Any, **context: Any) -> Array:
        """Return the exact local inverse when one is defined."""
        del y, context
        raise NotImplementedError(f"{type(self).__name__} does not define inv().")

    def solve(self, y: Any, **context: Any) -> Array:
        """Return a representative local input for an output ``y``."""
        return self.inv(y, **context)

    def decode(self, x: Any, **context: Any) -> Array:
        """Map a deepest input forward through the complete transform spine."""
        if isinstance(self.x, Transform):
            x = self.x.decode(x, **context)
        y = self.fwd(_resolved(x, context), **context)
        return _as_inexact_array(y, "y")

    def encode(self, y: Any, **context: Any) -> Array:
        """Map an output back through a fully reverse-capable transform spine."""
        x = self.solve(_resolved(y, context), **context)
        x = _resolved(x, context)
        if isinstance(self.x, Transform):
            return self.x.encode(x, **context)
        return _as_inexact_array(x, "x")

    def initialise(self, y: Any, **context: Any) -> Transform:
        """Bind a deepest input when every traversed transform supports reverse."""
        if isinstance(self.x, Transform):
            inner = self.x.initialise(self.solve(y, **context), **context)
            return eqx.tree_at(lambda transform: transform.x, self, inner)
        if isinstance(self.x, Expression):
            raise ValueError(
                f"Cannot initialise {type(self).__name__} through an Expression "
                "input; update its canonical definition or Linked owner instead."
            )
        x = _operand(self.encode(y, **context), "x", optional=True)
        return eqx.tree_at(
            lambda transform: transform.x,
            self,
            x,
            is_leaf=lambda leaf: leaf is None,
        )

    def project(self, y: Any, **context: Any) -> Array:
        """Project ``y`` through the available reverse and forward operations."""
        return self.decode(self.encode(y, **context), **context)


class Add(Transform):
    """Add a bias: ``y = x + b``."""

    b: Any = None

    def __init__(self, x: Any = None, b: Any = None, *, alias: Any = None):
        _initialise_transform(self, x, alias, b=b)

    def fwd(self, x: Any, **context: Any) -> Array:
        return add(x, self.b, **context)

    def inv(self, y: Any, **context: Any) -> Array:
        y = _as_inexact_array(_resolved(y, context), "y")
        b = _resolved(self.b, context)
        if b is None:
            return y
        x = y - _as_inexact_array(b, "b")
        return _preserve_shape(y, x, "b")

    def __zodiax_validate__(self) -> None:
        _validate_operand(self.b, "b", optional=True)


class Mul(Transform):
    """Multiply by a scale: ``y = x * s``."""

    s: Any = None

    def __init__(self, x: Any = None, s: Any = None, *, alias: Any = None):
        _initialise_transform(self, x, alias, s=s)

    def fwd(self, x: Any, **context: Any) -> Array:
        return mul(x, self.s, **context)

    def inv(self, y: Any, **context: Any) -> Array:
        y = _as_inexact_array(_resolved(y, context), "y")
        s = _resolved(self.s, context)
        if s is None:
            return y
        x = y / _nonzero(s, "s")
        return _preserve_shape(y, x, "s")

    def __zodiax_validate__(self) -> None:
        _validate_operand(self.s, "s", optional=True)


class Pow(Transform):
    """Raise an input to a power: ``y = x**p``.

    General powers are intentionally forward-only because a globally valid real
    inverse requires domain and branch choices.
    """

    p: Any = None

    def __init__(self, x: Any = None, p: Any = None, *, alias: Any = None):
        _initialise_transform(self, x, alias, required=("p",), p=p)

    def fwd(self, x: Any, **context: Any) -> Array:
        return power(x, self.p, **context)

    def __zodiax_validate__(self) -> None:
        _validate_operand(self.p, "p")


class Exp(Transform):
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
        _initialise_transform(self, x, alias, base=base)

    def fwd(self, x: Any, **context: Any) -> Array:
        x = _as_inexact_array(_resolved(x, context), "x")
        base = _base(self.base, context)
        if base is None:
            return np.exp(x)
        return _preserve_shape(x, np.power(base, x), "base")

    def inv(self, y: Any, **context: Any) -> Array:
        y = _as_inexact_array(_resolved(y, context), "y")
        base = _base(self.base, context)
        if base is None:
            return np.log(y)
        return _preserve_shape(y, np.log(y) / np.log(base), "base")

    def __zodiax_validate__(self) -> None:
        _validate_operand(self.base, "base", optional=True)


class Log(Transform):
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
        _initialise_transform(self, x, alias, base=base)

    def fwd(self, x: Any, **context: Any) -> Array:
        x = _as_inexact_array(_resolved(x, context), "x")
        base = _base(self.base, context)
        if base is None:
            return np.log(x)
        return _preserve_shape(x, np.log(x) / np.log(base), "base")

    def inv(self, y: Any, **context: Any) -> Array:
        y = _as_inexact_array(_resolved(y, context), "y")
        base = _base(self.base, context)
        if base is None:
            return np.exp(y)
        return _preserve_shape(y, np.power(base, y), "base")

    def __zodiax_validate__(self) -> None:
        _validate_operand(self.base, "base", optional=True)


class MatMul(Transform):
    """Apply a matrix or sampled basis: ``y = x @ M``."""

    M: Any = None

    def __init__(self, x: Any = None, M: Any = None, *, alias: Any = None):
        _initialise_transform(self, x, alias, required=("M",), M=M)
        if not isinstance(self.M, Expression):
            self._validate_M(self.M)

    @staticmethod
    def _validate_M(M: Array) -> None:
        if M.ndim == 0:
            raise ValueError("M must have at least one axis; use Mul for scaling.")
        if 0 in M.shape:
            raise ValueError("M must have nonzero dimensions.")

    def fwd(self, x: Any, **context: Any) -> Array:
        M = _resolved(self.M, context)
        if M is None:
            raise ValueError("MatMul M must not resolve to None.")
        return matmul(x, M, **context)

    def solve(self, y: Any, **context: Any) -> Array:
        y = _resolved(y, context)
        M = _resolved(self.M, context)
        if M is None:
            raise ValueError("MatMul M must not resolve to None.")
        return _solve_matmul(y, M)

    def __zodiax_validate__(self) -> None:
        _validate_operand(self.M, "M")
        if not isinstance(self.M, Expression):
            self._validate_M(self.M)


class Mask(Transform):
    """Scatter compact ``x`` values into selected output entries.

    The output ``shape`` is static, while selected flat indices occupy one integer
    JAX-array leaf. Large masks therefore do not become large Python PyTree
    definitions or static compilation keys.
    """

    indices: Array
    shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, x: Any = None, mask: Any = None, *, alias: Any = None):
        mask = onp.asarray(mask)
        if mask.dtype != onp.bool_:
            raise TypeError("mask must have a boolean dtype.")
        if mask.ndim == 0:
            raise ValueError("mask must have at least one axis.")
        indices = onp.flatnonzero(mask)
        if indices.size == 0:
            raise ValueError("mask must select at least one entry.")

        _initialise_transform(self, x, alias)
        self.indices = np.asarray(indices, dtype=np.int32)
        self.shape = tuple(int(size) for size in mask.shape)

    @property
    def size(self) -> int:
        """Number of selected values."""
        return self.indices.size

    def fwd(self, x: Any, **context: Any) -> Array:
        x = _as_inexact_array(_resolved(x, context), "x")
        if x.ndim == 0 or x.shape[-1] != self.size:
            raise ValueError(f"x must have a final axis of size {self.size}.")
        flat = np.zeros(x.shape[:-1] + (prod(self.shape),), dtype=x.dtype)
        flat = flat.at[..., np.asarray(self.indices)].set(x)
        return flat.reshape(x.shape[:-1] + self.shape)

    def solve(self, y: Any, **context: Any) -> Array:
        """Gather representative compact values from a full output."""
        y = _as_inexact_array(_resolved(y, context), "y")
        ndim = len(self.shape)
        if y.ndim < ndim or y.shape[-ndim:] != self.shape:
            raise ValueError(f"y must end in shape {self.shape}.")
        batch = y.shape[:-ndim]
        flat = y.reshape(batch + (prod(self.shape),))
        return flat[..., np.asarray(self.indices)]

    def _index_summary(self) -> tuple[int, ...] | str:
        indices = tuple(int(index) for index in onp.asarray(self.indices))
        if self.size <= 8:
            return indices
        payload = repr((self.shape, indices)).encode()
        digest = blake2s(payload, digest_size=4).hexdigest()
        return f"{self.size}@{digest}"

    def __pdoc__(self, **kwargs: Any) -> Any:
        fields = [] if self.x is None else [("x", self.x)]
        fields.extend([("idx", self._index_summary()), ("shape", self.shape)])
        return _pdoc(
            self,
            "Mask",
            fields,
            **kwargs,
        )

    def __zodiax_validate__(self) -> None:
        if type(self.shape) is not tuple or not self.shape:
            raise ValueError("shape must be a non-empty tuple.")
        if any(type(size) is not int or size <= 0 for size in self.shape):
            raise ValueError("shape dimensions must be positive integers.")
        if not isinstance(self.indices, Array):
            raise TypeError("indices must be a JAX array.")
        if bool(self.indices.weak_type) or not np.issubdtype(
            self.indices.dtype, np.integer
        ):
            raise TypeError("indices must be a strongly typed integer JAX array.")
        if self.indices.ndim != 1 or self.indices.size == 0:
            raise ValueError("indices must be a non-empty one-dimensional array.")
        if bool(np.any(self.indices[:-1] >= self.indices[1:])):
            raise ValueError("indices must be strictly increasing and unique.")
        if bool(self.indices[0] < 0) or bool(self.indices[-1] >= prod(self.shape)):
            raise ValueError("indices must lie within shape.")


class Map(Transform):
    """A compact fixed-order scale, matrix/basis, and bias composition.

    Forward evaluation is ``y = matmul(s * x, M) + b``. Each of ``s``, ``M``, and
    ``b`` may be ``None``, denoting identity, so this includes both the conventional
    elementwise affine map ``s * x + b`` and row-vector dense map ``x @ M + b``.

    If both ``s`` and ``M`` are trainable then the parameterisation is generally
    non-identifiable: scale can move between them without changing the output. The
    combined form is intended for fixed unit/preconditioning scales or cases where
    an explicit constraint gives each field distinct meaning.
    """

    s: Any = None
    M: Any = None
    b: Any = None

    def __init__(
        self,
        x: Any = None,
        s: Any = None,
        M: Any = None,
        b: Any = None,
        *,
        alias: Any = None,
    ):
        _initialise_transform(self, x, alias, s=s, M=M, b=b)
        if self.M is not None and not isinstance(self.M, Expression):
            MatMul._validate_M(self.M)

    def fwd(self, x: Any, **context: Any) -> Array:
        y = mul(x, self.s, **context)
        y = matmul(y, self.M, **context)
        return add(y, self.b, **context)

    def solve(self, y: Any, **context: Any) -> Array:
        y = _as_inexact_array(_resolved(y, context), "y")

        b = _resolved(self.b, context)
        if b is not None:
            shifted = y - _as_inexact_array(b, "b")
            y = _preserve_shape(y, shifted, "b")

        M = _resolved(self.M, context)
        if M is not None:
            y = _solve_matmul(y, M)

        s = _resolved(self.s, context)
        if s is not None:
            unscaled = y / _nonzero(s, "s")
            y = _preserve_shape(y, unscaled, "s")
        return y

    def __zodiax_validate__(self) -> None:
        _validate_operand(self.s, "s", optional=True)
        _validate_operand(self.M, "M", optional=True)
        _validate_operand(self.b, "b", optional=True)
        if self.M is not None and not isinstance(self.M, Expression):
            MatMul._validate_M(self.M)
