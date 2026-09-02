"""Shared context conventions for numerical parameterisations."""

from __future__ import annotations

from typing import Any


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
