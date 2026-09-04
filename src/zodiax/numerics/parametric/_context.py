"""Shared context conventions for numerical parameterisations."""

from __future__ import annotations

from typing import Any


def coordinate_axis(coordinates: Any, ndim: int, name: str = "coords") -> int:
    """Return the inferred component axis for a sampled ``ndim``-D field.

    Coordinate consumers share the :class:`Grid` convention
    ``(..., ndim, *spatial_shape)`` with exactly ``ndim`` trailing spatial axes.
    The component axis is consequently always ``-ndim - 1`` and never needs to be
    retained as model metadata.
    """
    axis = -ndim - 1
    if coordinates.ndim < ndim + 1 or coordinates.shape[axis] != ndim:
        raise ValueError(
            f"{name} must have shape (..., {ndim}, *spatial_shape) with "
            f"{ndim} final spatial axes."
        )
    return axis


def coordinate_context(
    coords: Any,
    coordinates: Any,
    context: dict[str, Any],
    *,
    required: bool,
) -> tuple[Any, dict[str, Any]]:
    """Return canonical coordinates and a context containing both spellings."""
    if coords is not None and coordinates is not None:
        raise ValueError("Provide only one of coords or coordinates.")
    value = coords if coords is not None else coordinates
    if value is None:
        if required:
            raise ValueError("Provide coords.")
        return None, context
    return value, {"coords": value, "coordinates": value, **context}
