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
from math import prod
from typing import Any

import equinox as eqx
import jax.numpy as np
import numpy as onp
from jax import Array

from .arrays import _as_inexact_array
from .expressions import Expression, _missing_context, _Unresolved, resolve

__all__ = [
    "Transform",
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


def _operand(value: Any, name: str, *, optional: bool = False) -> Any:
    """Normalise an array-or-expression operand without realising expressions."""
    if value is None:
        if optional:
            return None
        raise ValueError(f"{name} must not be None.")
    if isinstance(value, Expression):
        return value
    return _as_inexact_array(value, name)


def _preserve_shape(x: Array, y: Array, name: str) -> Array:
    """Require a broadcast operation to retain its input shape."""
    if y.shape != x.shape:
        raise ValueError(
            f"{name} must broadcast without changing x.shape; received "
            f"{x.shape} -> {y.shape}."
        )
    return y


def matmul(x: Any, M: Any = None, **context: Any) -> Array:
    """Contract the final axis of ``x`` with the first axis of ``M``.

    For ``M.shape == (input_size, *output_shape)``, an input with shape
    ``(*batch_shape, input_size)`` produces ``(*batch_shape, *output_shape)``.
    A two-dimensional ``M`` therefore has row-vector ``x @ M`` semantics, while a
    higher-rank ``M`` represents a sampled basis with native output axes. ``None``
    is the identity.
    """
    x = resolve(x, **context)
    x = _as_inexact_array(x, "x")
    M = resolve(M, **context)
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


class Transform(Expression):
    """A stored-input numerical transform.

    ``x`` is always the stored upstream input and ``fwd(x)`` always returns the
    output ``y``. A transform with ``x=None`` is unbound, while any Expression may
    provide its stored input. Subclasses put the local numerical formula in ``fwd``;
    the inherited ``evaluate`` supplies the stored input, while ``apply`` applies
    the complete nested transform chain to an explicit deepest input.
    Declare required named context, such as coordinates, in the ``fwd`` signature.
    The inherited ``evaluate`` checks those requirements, so no forwarding override
    is needed. Exact transforms implement :meth:`inv`; projections or rank-deficient
    transforms implement :meth:`solve`.

    ``evaluate``, ``apply``, and ``encode`` prepare floating or complex array
    inputs. The numerical methods ``fwd``, ``inv``, and ``solve`` accept and return
    arrays; they resolve their stored operands where needed. Expressions used as
    numerical operands must supply numerical scalars or arrays.
    """

    x: Any = None

    @abstractmethod
    def fwd(self, x: Array, **context: Any) -> Array:
        """Apply this transform to explicit ``x``, without following stored ``x``.

        Declare required named context in the signature, for example
        ``fwd(self, x, *, coordinates, **context)``. During resolution, missing
        context retains the transform with its available children prepared.
        """

    def evaluate(self, **context: Any) -> Array:
        """Resolve stored input and pass it to the subclass's ``fwd`` calculation."""
        if self.x is None:
            raise _Unresolved(f"{type(self).__name__} has no stored x input.")

        # fwd receives x positionally; its remaining required inputs come from
        # context. Defer through the usual resolver signal if any are unavailable,
        # allowing child preparation to retain whatever can already resolve.
        missing = _missing_context(self.fwd, context, self.x)
        if missing:
            raise _Unresolved(
                f"{type(self).__name__}.fwd requires context: {', '.join(missing)}."
            )

        x = resolve(self.x, **context)
        x = _as_inexact_array(x, "x")
        return self.fwd(x, **context)

    def __call__(self, x: Any = None, **context: Any) -> Array | Transform:
        """Resolve stored input, or apply the transform to explicit input ``x``."""
        if x is None:
            return self.resolve(**context)
        return self.apply(x, **context)

    def apply(self, x: Any, **context: Any) -> Array:
        """Apply the complete transform chain to an explicit deepest input."""
        if isinstance(self.x, Transform):
            x = self.x.apply(x, **context)
        else:
            x = resolve(x, **context)
            x = _as_inexact_array(x, "x")
        return self.fwd(x, **context)

    def inv(self, y: Array, **context: Any) -> Array:
        """Return the exact local inverse when one is defined."""
        raise NotImplementedError(f"{type(self).__name__} does not define inv().")

    def solve(self, y: Array, **context: Any) -> Array:
        """Return a representative local input for an output ``y``."""
        return self.inv(y, **context)

    def encode(self, y: Any, **context: Any) -> Array:
        """Map an output back through a fully reverse-capable transform spine."""
        value = resolve(y, **context)
        value = _as_inexact_array(value, "y")

        # Undo each transform from the outermost operation to the deepest input.
        transform = self
        while isinstance(transform, Transform):
            value = transform.solve(value, **context)
            transform = transform.x
        return value

    def initialise(self, y: Any, **context: Any) -> Transform:
        """Bind a deepest input when every traversed transform supports reverse."""

        # Locate the stored input without evaluating or replacing the transforms.
        def deepest_input(transform):
            while isinstance(transform.x, Transform):
                transform = transform.x
            return transform.x

        if isinstance(deepest_input(self), Expression):
            raise ValueError(
                f"Cannot initialise {type(self).__name__} through an Expression "
                "input; update its canonical definition or Linked owner instead."
            )
        x = self.encode(y, **context)
        return eqx.tree_at(
            deepest_input,
            self,
            x,
            is_leaf=lambda leaf: leaf is None,
        )

    def project(self, y: Any, **context: Any) -> Array:
        """Project ``y`` through the available reverse and forward operations."""
        return self.apply(self.encode(y, **context), **context)


class Add(Transform):
    """Add a bias: ``y = x + b``, preserving the input shape; ``b=None`` is identity."""

    b: Any = None

    def __init__(self, x: Any = None, b: Any = None, *, alias: Any = None):
        """Store optional input and bias arrays or expressions, with aliases."""
        self.alias = alias
        self.x = _operand(x, "x", optional=True)
        self.b = _operand(b, "b", optional=True)

    def fwd(self, x: Array, **context: Any) -> Array:
        b = resolve(self.b, **context)
        if b is None:
            return x
        y = x + b
        return _preserve_shape(x, y, "b")

    def inv(self, y: Array, **context: Any) -> Array:
        b = resolve(self.b, **context)
        if b is None:
            return y
        x = y - b
        return _preserve_shape(y, x, "b")


class Mul(Transform):
    """Scale an input: ``y = x * s``, preserving its shape; ``s=None`` is identity."""

    s: Any = None

    def __init__(self, x: Any = None, s: Any = None, *, alias: Any = None):
        """Store optional input and scale arrays or expressions, with aliases."""
        self.alias = alias
        self.x = _operand(x, "x", optional=True)
        self.s = _operand(s, "s", optional=True)

    def fwd(self, x: Array, **context: Any) -> Array:
        s = resolve(self.s, **context)
        if s is None:
            return x
        y = x * s
        return _preserve_shape(x, y, "s")

    def inv(self, y: Array, **context: Any) -> Array:
        s = resolve(self.s, **context)
        if s is None:
            return y
        x = y / s
        return _preserve_shape(y, x, "s")


class Pow(Transform):
    """Raise an input to a power: ``y = x**p``.

    General powers are intentionally forward-only because a globally valid real
    inverse requires domain and branch choices.
    """

    p: Any = None

    def __init__(self, x: Any = None, p: Any = None, *, alias: Any = None):
        """Store optional input, required power, and parameter aliases."""
        self.alias = alias
        self.x = _operand(x, "x", optional=True)
        self.p = _operand(p, "p")

    def fwd(self, x: Array, **context: Any) -> Array:
        p = resolve(self.p, **context)
        return np.power(x, p)


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
        """Store optional input and base arrays or expressions, with aliases."""
        self.alias = alias
        self.x = _operand(x, "x", optional=True)
        self.base = _operand(base, "base", optional=True)

    def fwd(self, x: Array, **context: Any) -> Array:
        base = resolve(self.base, **context)
        if base is None:
            return np.exp(x)
        y = np.power(base, x)
        return _preserve_shape(x, y, "base")

    def inv(self, y: Array, **context: Any) -> Array:
        base = resolve(self.base, **context)
        if base is None:
            return np.log(y)
        x = np.log(y) / np.log(base)
        return _preserve_shape(y, x, "base")


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
        """Store optional input and base arrays or expressions, with aliases."""
        self.alias = alias
        self.x = _operand(x, "x", optional=True)
        self.base = _operand(base, "base", optional=True)

    def fwd(self, x: Array, **context: Any) -> Array:
        base = resolve(self.base, **context)
        if base is None:
            return np.log(x)
        y = np.log(x) / np.log(base)
        return _preserve_shape(x, y, "base")

    def inv(self, y: Array, **context: Any) -> Array:
        base = resolve(self.base, **context)
        if base is None:
            return np.exp(y)
        x = np.power(base, y)
        return _preserve_shape(y, x, "base")


class MatMul(Transform):
    """Apply a matrix or sampled basis: ``y = x @ M``."""

    M: Any = None

    def __init__(self, x: Any = None, M: Any = None, *, alias: Any = None):
        """Store optional input, required matrix or basis, and aliases."""
        self.alias = alias
        self.x = _operand(x, "x", optional=True)
        self.M = _operand(M, "M")
        if not isinstance(self.M, Expression):
            self._validate_M(self.M)

    @staticmethod
    def _validate_M(M: Array) -> None:
        if M.ndim == 0:
            raise ValueError("M must have at least one axis; use Mul for scaling.")
        if 0 in M.shape:
            raise ValueError("M must have nonzero dimensions.")

    def fwd(self, x: Array, **context: Any) -> Array:
        return matmul(x, self.M, **context)

    def solve(self, y: Array, **context: Any) -> Array:
        M = resolve(self.M, **context)
        return _solve_matmul(y, M)


class Mask(Transform):
    """Populate selected output entries with compact ``x`` values.

    The final axis of ``x`` contains one value per true mask entry, in flattened
    order. The output replaces that axis with ``mask.shape``, preserving any
    leading batch axes and filling unselected entries with zero.

    Store one boolean mask and a fixed selected count, ``size``, so JIT knows the
    compact array length. Changing mask values must preserve this count.

    Performance note: when the mask is passed through JIT as an array input,
    selected indices are calculated during execution. This adds work proportional
    to the number of mask elements in both ``fwd`` and ``solve``.
    """

    mask: Array
    size: int = eqx.field(static=True)

    def __init__(self, x: Any = None, mask: Any = None, *, alias: Any = None):
        """Store optional input, a boolean mask, its selected count, and aliases."""
        mask = onp.asarray(mask)
        if mask.dtype != onp.bool_:
            raise TypeError("mask must have a boolean dtype.")
        if mask.ndim == 0:
            raise ValueError("mask must have at least one axis.")
        size = int(onp.count_nonzero(mask))
        if size == 0:
            raise ValueError("mask must select at least one entry.")

        self.alias = alias
        self.x = _operand(x, "x", optional=True)
        self.mask = np.asarray(mask)
        self.size = size

    @property
    def shape(self) -> tuple[int, ...]:
        """Output shape for one compact input vector."""
        return self.mask.shape

    def fwd(self, x: Array, **context: Any) -> Array:
        if x.ndim == 0 or x.shape[-1] != self.size:
            raise ValueError(f"x must have a final axis of size {self.size}.")

        indices = np.flatnonzero(self.mask, size=self.size)
        flat = np.zeros(x.shape[:-1] + (self.mask.size,), dtype=x.dtype)
        flat = flat.at[..., indices].set(x)
        return flat.reshape(x.shape[:-1] + self.shape)

    def solve(self, y: Array, **context: Any) -> Array:
        """Gather representative compact values from a full output."""
        ndim = len(self.shape)
        if y.ndim < ndim or y.shape[-ndim:] != self.shape:
            raise ValueError(f"y must end in shape {self.shape}.")

        batch = y.shape[:-ndim]
        flat = y.reshape(batch + (self.mask.size,))
        indices = np.flatnonzero(self.mask, size=self.size)
        return flat[..., indices]


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
        """Store input, scale, matrix or basis, bias, and aliases; all are optional."""
        self.alias = alias
        self.x = _operand(x, "x", optional=True)
        self.s = _operand(s, "s", optional=True)
        self.M = _operand(M, "M", optional=True)
        self.b = _operand(b, "b", optional=True)
        if self.M is not None and not isinstance(self.M, Expression):
            MatMul._validate_M(self.M)

    def fwd(self, x: Array, **context: Any) -> Array:
        # Apply the input scale, then contract with the matrix or sampled basis.
        s = resolve(self.s, **context)
        if s is not None:
            scaled = x * s
            x = _preserve_shape(x, scaled, "s")
        y = matmul(x, self.M, **context)

        # Add the output bias without introducing new axes.
        b = resolve(self.b, **context)
        if b is not None:
            shifted = y + b
            y = _preserve_shape(y, shifted, "b")
        return y

    def solve(self, y: Array, **context: Any) -> Array:
        # Undo the bias, matrix or basis, and input scale in reverse order.
        b = resolve(self.b, **context)
        if b is not None:
            shifted = y - b
            y = _preserve_shape(y, shifted, "b")

        M = resolve(self.M, **context)
        if M is not None:
            y = _solve_matmul(y, M)

        s = resolve(self.s, **context)
        if s is not None:
            unscaled = y / s
            y = _preserve_shape(y, unscaled, "s")
        return y
