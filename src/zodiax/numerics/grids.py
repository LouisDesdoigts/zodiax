"""Regular Cartesian coordinate grids with optional unit conversion."""

from __future__ import annotations

from operator import index as integer_index
from typing import Any

import equinox as eqx
import jax.numpy as np
from jax import Array

from .arrays import _as_inexact_array
from .expressions import Expression, resolve
from .transforms import _operand, _validate_operand
from .units import Unit, unit_info

__all__ = ["Grid"]


def _sizes(value: Any, name: str = "n") -> tuple[int, ...]:
    """Return positive static sample counts in grid-axis order."""
    values = value if isinstance(value, (tuple, list)) else (value,)
    if not values:
        raise ValueError(f"{name} must contain at least one size.")

    sizes = []
    for value in values:
        if isinstance(value, bool):
            raise TypeError(f"{name} must contain integers.")
        try:
            value = integer_index(value)
        except TypeError as error:
            raise TypeError(f"{name} must contain integers.") from error
        if value < 1:
            raise ValueError(f"{name} must contain positive sizes.")
        sizes.append(value)
    return tuple(sizes)


def _axis(value: Any, ndim: int, name: str, context: dict[str, Any]) -> Array:
    """Resolve and broadcast one coordinate-component vector."""
    value = _as_inexact_array(resolve(value, **context), name)
    if value.ndim == 0:
        return np.broadcast_to(value, (ndim,))
    if value.shape[-1] == 1:
        return np.broadcast_to(value, value.shape[:-1] + (ndim,))
    if value.shape[-1] != ndim:
        raise ValueError(f"{name} must be scalar or end in an axis of size {ndim}.")
    return value


def _stored_axis(value: Any, ndim: int, name: str) -> Any:
    """Broadcast a concrete stored axis while preserving deferred expressions."""
    if value is None or isinstance(value, Expression):
        return value
    return _axis(value, ndim, name, {})


def _unit(value: Any) -> Unit | None:
    """Return an unbound shared Grid unit conversion."""
    if value is None:
        return None
    if isinstance(value, str):
        value = Unit(unit=value)
    if not isinstance(value, Unit):
        raise TypeError("unit must be a unit string, Unit, or None.")
    if value.x is not None:
        raise ValueError("A Grid Unit must be unbound; d and c supply its inputs.")
    info = unit_info(value.unit)
    if info.power != 1 or info.category not in ("cartesian", "angular"):
        raise ValueError("Grid units must be simple Cartesian or angular units.")
    return value


class Grid(Expression):
    """Generate a regular Cartesian coordinate array from ``n``, ``d``, and ``c``.

    Grid axes use dimension-generic ``ij`` indexing. Component ``i`` varies along
    sample axis ``i``, so ``n=(n0, n1, ...)`` produces coordinates with shape
    ``(..., ndim, n0, n1, ...)``. ``d`` and ``c`` are ordinary arrays or numerical
    expressions. An optional shared ``unit`` converts both into an explicit or
    globally selected output unit.
    """

    n: tuple[int, ...] = eqx.field(static=True)
    d: Any = None
    c: Any = None
    unit: Unit | None = None

    def __init__(
        self,
        n: Any,
        d: Any,
        c: Any = None,
        unit: str | Unit | None = None,
        *,
        alias: Any = None,
    ):
        self.alias = alias
        self.n = _sizes(n)
        self.d = _operand(d, "d")
        self.c = _operand(c, "c", optional=True)
        self.unit = _unit(unit)

        if not isinstance(self.d, Expression):
            self.d = _axis(self.d, self.ndim, "d", {})
        if self.c is not None and not isinstance(self.c, Expression):
            self.c = _axis(self.c, self.ndim, "c", {})

    @property
    def ndim(self) -> int:
        """Number of coordinate dimensions."""
        return len(self.n)

    @property
    def shape(self) -> tuple[int, ...]:
        """Spatial sample shape in ``ij`` grid-axis order."""
        return self.n

    def _vectors(self, context: dict[str, Any]) -> tuple[Array, Array]:
        """Return broadcast spacing and centre vectors in component order."""
        d = _axis(self.d, self.ndim, "d", context)
        if self.unit is not None:
            d = self.unit(d, **context)
        d = eqx.error_if(d, np.any(d <= 0), "d must contain positive values.")

        if self.c is None:
            c = np.zeros((self.ndim,), dtype=d.dtype)
        else:
            c = _axis(self.c, self.ndim, "c", context)
            if self.unit is not None:
                c = self.unit(c, **context)

        batch_shape = np.broadcast_shapes(d.shape[:-1], c.shape[:-1])
        shape = batch_shape + (self.ndim,)
        return np.broadcast_to(d, shape), np.broadcast_to(c, shape)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Broadcast metadata batch shape for context-free grid leaves."""
        d, _ = self._vectors({})
        return d.shape[:-1]

    def axes_for(self, n: Any, **context: Any) -> tuple[Array, ...]:
        """Return coordinate vectors for explicit grid-axis sample counts."""
        n = _sizes(n)
        if len(n) != self.ndim:
            raise ValueError("n dimensionality must match Grid.ndim.")
        d, c = self._vectors(context)
        return tuple(
            c[..., i, None] + (np.arange(size) - (size - 1) / 2) * d[..., i, None]
            for i, size in enumerate(n)
        )

    @property
    def axes(self) -> tuple[Array, ...]:
        """Coordinate vectors in grid-axis order without additional context."""
        return self.axes_for(self.n)

    def _coordinates(self, context: dict[str, Any]) -> Array:
        """Generate the full coordinate mesh with ``ij`` indexing."""
        d, c = self._vectors(context)
        batch_shape = d.shape[:-1]
        axes = []
        for i, size in enumerate(self.n):
            axis = (np.arange(size) - (size - 1) / 2) * d[..., i, None]
            spatial_shape = [1] * self.ndim
            spatial_shape[i] = size
            axis = axis.reshape(batch_shape + tuple(spatial_shape))
            axes.append(np.broadcast_to(axis, batch_shape + self.shape))

        coordinates = np.stack(tuple(axes), axis=len(batch_shape))
        center = c.reshape(batch_shape + (self.ndim,) + (1,) * self.ndim)
        return coordinates + center

    def evaluate(self, **context: Any) -> Array:
        """Resolve sampling leaves and return the full coordinate array."""
        return self._coordinates(context)

    @property
    def coordinates(self) -> Array:
        """Full coordinate array without additional resolution context."""
        return self.evaluate()

    @property
    def fov(self) -> Array:
        """Sampled field of view in grid-axis order."""
        d, _ = self._vectors({})
        return d * np.asarray(self.n)

    @property
    def measure(self) -> Array:
        """Sample-cell measure across all physical axes."""
        d, _ = self._vectors({})
        return np.prod(d, axis=-1)

    def broadcast(self, ndim: int) -> Grid:
        """Broadcast a one-dimensional definition to ``ndim`` grid axes."""
        if isinstance(ndim, bool):
            raise TypeError("ndim must be a positive integer.")
        try:
            ndim = integer_index(ndim)
        except TypeError as error:
            raise TypeError("ndim must be a positive integer.") from error
        if ndim < 1:
            raise ValueError("ndim must be a positive integer.")
        if self.ndim not in (1, ndim):
            raise ValueError("Grid dimensionality cannot broadcast to ndim.")

        n = self.n * ndim if self.ndim == 1 else self.n
        return Grid(
            n=n,
            d=_stored_axis(self.d, ndim, "d"),
            c=_stored_axis(self.c, ndim, "c"),
            unit=self.unit,
            alias=self.alias,
        )

    def match_shape(self, shape: Any) -> Grid:
        """Return a grid whose sample counts match an ``ij`` spatial shape."""
        shape = _sizes(shape, "shape")
        grid = self.broadcast(len(shape))
        n = shape
        if grid.n != n:
            raise ValueError("Spatial shape must match Grid.n.")
        return grid

    def resize(self, n: Any) -> Grid:
        """Change sample counts while retaining spacing and centre leaves."""
        n = _sizes(n)
        if len(n) != self.ndim:
            raise ValueError("n dimensionality must match Grid.ndim.")
        return Grid(n=n, d=self.d, c=self.c, unit=self.unit, alias=self.alias)

    @classmethod
    def from_axes(
        cls,
        axes: Any,
        unit: str | Unit | None = None,
        *,
        alias: Any = None,
    ) -> Grid:
        """Construct a grid from regularly sampled coordinate axes."""
        axes = tuple(_as_inexact_array(axis, "axis") for axis in axes)
        if not axes:
            raise ValueError("axes must contain at least one coordinate vector.")
        if any(axis.ndim == 0 or axis.shape[-1] < 2 for axis in axes):
            raise ValueError("Every coordinate axis must contain at least two values.")

        n = tuple(axis.shape[-1] for axis in axes)
        d = np.stack([axis[..., 1] - axis[..., 0] for axis in axes], axis=-1)
        c = np.stack(
            [(axis[..., -1] + axis[..., 0]) / 2 for axis in axes],
            axis=-1,
        )
        return cls(n=n, d=d, c=c, unit=unit, alias=alias)

    def __zodiax_validate__(self) -> None:
        """Validate constructor-free archive reconstruction."""
        if type(self.n) is not tuple or not self.n:
            raise TypeError("n must be stored as a non-empty tuple.")
        if any(type(size) is not int or size < 1 for size in self.n):
            raise ValueError("n must contain positive Python integers.")
        _validate_operand(self.d, "d")
        _validate_operand(self.c, "c", optional=True)
        _unit(self.unit)
        if not isinstance(self.d, Expression):
            _axis(self.d, self.ndim, "d", {})
        if self.c is not None and not isinstance(self.c, Expression):
            _axis(self.c, self.ndim, "c", {})
