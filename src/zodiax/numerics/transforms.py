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
from jax import Array

from .arrays import as_array
from .expressions import Expression, _missing_context, _Unresolved, resolve

__all__ = [
    "Parametric",
    "Transform",
    "matmul",
    "Mask",
    "Map",
]


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
    x = as_array(x)
    M = resolve(M, **context)
    if M is None:
        return x
    M = as_array(M)
    if M.ndim == 0:
        raise ValueError("M must have at least one axis; use Map(s=...) for scaling.")
    if 0 in M.shape:
        raise ValueError("M must have nonzero dimensions.")
    if x.ndim == 0 or x.shape[-1] != M.shape[0]:
        raise ValueError(f"x must have a final axis of size {M.shape[0]}.")
    return np.tensordot(x, M, axes=((-1,), (0,)))


def _solve_matmul(y: Any, M: Any) -> Array:
    """Return a minimum-norm input for :func:`matmul`."""
    y = as_array(y)
    M = as_array(M)
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
    matrix = M.reshape((M.shape[0], width))
    flat = y.reshape((-1, width))
    # The pseudoinverse gives the minimum-norm solution, including zero for a zero
    # matrix. JAX handles rank selection and ordinary numerical type promotion.
    x = flat @ np.linalg.pinv(matrix)
    return x.reshape(batch_shape + (M.shape[0],))


class Parametric(Expression):
    """A numerical expression evaluated from one principal stored input.

    Subclasses choose the field name and expose it through the read-only ``input``
    property. Return the stored value unchanged so immutable updates can locate it
    in the tree. ``None`` leaves the expression unbound.

    Implement the local calculation in ``fwd``, declaring required named context
    in its signature. ``evaluate`` resolves the stored input, while ``apply``
    follows nested Parametric inputs from the deepest one outwards. An optional
    ``inv`` supplies the local reverse; ``initialise`` binds a recovered deepest
    input without replacing the surrounding definitions.
    """

    @property
    @abstractmethod
    def input(self) -> Any:
        """Return the principal stored field unchanged, without resolving it."""

    @abstractmethod
    def fwd(self, x: Array, **context: Any) -> Array:
        """Apply this calculation to explicit ``x``, without following stored input.

        Declare required named context in the signature, for example
        ``fwd(self, x, *, coordinates, **context)``. During resolution, missing
        context retains the transform with its available children prepared.
        """

    def evaluate(self, **context: Any) -> Array:
        """Resolve stored input and pass it to the subclass's ``fwd`` calculation."""
        value = self.input
        if value is None:
            raise _Unresolved(f"{type(self).__name__} has no stored input.")

        # fwd receives x positionally; its remaining required inputs come from
        # context. Defer through the usual resolver signal if any are unavailable,
        # allowing child preparation to retain whatever can already resolve.
        missing = _missing_context(self.fwd, context, value)
        if missing:
            raise _Unresolved(
                f"{type(self).__name__}.fwd requires context: {', '.join(missing)}."
            )

        value = resolve(value, **context)
        value = as_array(value)
        return self.fwd(value, **context)

    def __call__(
        self, x: Any = None, *, inverse: bool = False, **context: Any
    ) -> Array | Parametric:
        """Resolve stored input, or apply the transform to explicit input ``x``."""
        if x is None:
            if inverse:
                raise ValueError("An inverse call requires an explicit input value.")
            return self.resolve(**context)
        return self.apply(x, inverse=inverse, **context)

    def apply(self, x: Any, *, inverse: bool = False, **context: Any) -> Array:
        """Apply the complete stored Parametric chain to an explicit input.

        Forward application supplies the deepest input and works outwards.
        ``inverse=True`` reverses the chain from the outermost transform inwards,
        using each subclass's ``inv``. A missing reverse raises NotImplementedError.
        Representative reverses need not recover every input or output exactly.
        Keep ``inverse`` a Python bool fixed during JAX tracing.
        """
        value = resolve(x, **context)
        value = as_array(value)

        # Follow each principal input from the outermost owner to the deepest one.
        chain = []
        parametric = self
        while isinstance(parametric, Parametric):
            chain.append(parametric)
            parametric = parametric.input

        # Forward calculations run outwards; reverse calculations run inwards.
        if not inverse:
            chain.reverse()
        for parametric in chain:
            if inverse:
                value = parametric.inv(value, **context)
            else:
                value = parametric.fwd(value, **context)
        return value

    def inv(self, y: Array, **context: Any) -> Array:
        """Reverse this local calculation, when a subclass supplies a reverse."""
        raise NotImplementedError(f"{type(self).__name__} does not define inv().")

    def initialise(self, y: Any, **context: Any) -> Parametric:
        """Bind a deepest input when every traversed transform supports reverse."""

        # Locate the stored input without evaluating or replacing the transforms.
        def deepest_input(parametric):
            while isinstance(parametric.input, Parametric):
                parametric = parametric.input
            return parametric.input

        if isinstance(deepest_input(self), Expression):
            raise ValueError(
                f"Cannot initialise {type(self).__name__} through an Expression "
                "input; update its canonical definition or Linked owner instead."
            )
        x = self.apply(y, inverse=True, **context)
        return eqx.tree_at(
            deepest_input,
            self,
            x,
            is_leaf=lambda leaf: leaf is None,
        )


class Transform(Parametric):
    """A stored-input numerical transform.

    ``x`` is always the stored upstream input and ``fwd(x)`` always returns the
    output ``y``. A transform with ``x=None`` is unbound, while any Expression may
    provide its stored input. Subclasses put the local numerical formula in ``fwd``;
    the inherited ``evaluate`` supplies the stored input, while ``apply`` applies
    the complete nested transform chain to an explicit deepest input.
    Declare required named context, such as coordinates, in the ``fwd`` signature.
    The inherited ``evaluate`` checks those requirements, so no forwarding override
    is needed. Subclasses implement their local reverse in ``inv``. This may be an
    exact inverse or a representative reverse, such as a matrix pseudoinverse.

    Construction, ``evaluate``, and ``apply`` convert numerical inputs
    to arrays while preserving their inferred dtype. Supply floating or complex
    values for trainable parameters; integer inputs remain integers. The numerical
    methods ``fwd`` and ``inv`` follow ordinary JAX type promotion and
    resolve their stored operands where needed. Expressions used as numerical
    operands must supply numerical scalars or arrays.
    """

    x: Any = None

    @property
    def input(self) -> Any:
        """Return the stored transform input."""
        return self.x


class Mask(Transform):
    """Populate selected output entries with compact ``x`` values.

    The final axis of ``x`` contains one value per true mask entry, in flattened
    order. The output replaces that axis with ``mask.shape``, preserving any
    leading batch axes and filling unselected entries with zero.

    Store one boolean mask and a fixed selected count, ``size``, so JIT knows the
    compact array length. Construct the Mask before entering JIT; changing its
    mask values later must preserve this count.

    Performance note: when the mask is passed through JIT as an array input,
    selected indices are calculated during execution. This adds work proportional
    to the number of mask elements in both ``fwd`` and ``inv``.
    """

    mask: Array
    size: int = eqx.field(static=True)

    def __init__(self, x: Any = None, mask: Any = None, *, alias: Any = None):
        """Store optional input, a boolean mask, its selected count, and aliases."""
        mask = as_array(mask)
        if not isinstance(mask, Array) or mask.dtype != np.bool_:
            raise TypeError("mask must have a boolean dtype.")
        if mask.ndim == 0:
            raise ValueError("mask must have at least one axis.")
        # Count during eager construction so the compact length is static under JIT.
        size = int(np.count_nonzero(mask))
        if size == 0:
            raise ValueError("mask must select at least one entry.")

        self.alias = alias
        self.x = as_array(x)
        self.mask = mask
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

    def inv(self, y: Array, **context: Any) -> Array:
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
        self.x = as_array(x)
        self.s = as_array(s)
        self.M = as_array(M)
        self.b = as_array(b)
        if self.M is not None and not isinstance(self.M, Expression):
            if self.M.ndim == 0:
                raise ValueError("M must have at least one axis; use s for scaling.")
            if 0 in self.M.shape:
                raise ValueError("M must have nonzero dimensions.")

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

    def inv(self, y: Array, **context: Any) -> Array:
        """Return a representative input by reversing bias, matrix, and scale."""
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
