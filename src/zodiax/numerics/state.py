"""Running state stored in a Module, with expression references to its values."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax.tree_util as jtu
from jax import Array

from ..base import Module
from .arrays import _as_array
from .expressions import Expression

__all__ = ["State", "StateRef", "validate_state"]


class State(Module):
    """An immutable mapping of named numerical values.

    Each entry becomes a JAX array at construction, including numerical lists and
    tuples and an integer ``index``. Store expressions in the model and use
    StateRef to read these current values.

    Update entries with the inherited Module.set API. It does not convert
    replacement values; supply arrays.

    A scalar integer array supports x[index] and x.at[index].set(value) under JIT.
    For a slice starting at a dynamic index, use jax.lax.dynamic_slice_in_dim
    with a fixed slice size; ordinary Python slice bounds must remain static.
    When carrying State through a JAX loop, preserve its structure and array
    shapes and dtypes.
    """

    values: dict[str, Array]

    def __init__(
        self,
        values: Mapping[str, Any] | None = None,
        *,
        alias: Any = None,
        **entries: Any,
    ):
        """Store a mapping and keyword entries; keywords override mapping entries."""
        combined = {} if values is None else dict(values)
        combined.update(entries)

        stored_values = {}
        for name, value in combined.items():
            stored_values[name] = _as_array(value, name)

        self.alias = alias
        self.values = stored_values

    def value(self, path: str) -> Array:
        """Read a value using ordinary Module paths and aliases.

        Use a qualified path such as "values.time" when a name is ambiguous or
        shared with a State method or field. Missing entries raise KeyError.
        """
        try:
            return self.get(path, to_array=False)
        except (AttributeError, IndexError, KeyError) as error:
            raise KeyError(f"State has no value at {path!r}.") from error

    def ref(self, path: str) -> StateRef:
        """Return an expression reference to an existing state path."""
        self.value(path)
        return StateRef(path)

    def step(self, dt: Any = None) -> State:
        """Advance an index by one and time by an explicit or stored dt.

        Index and time are optional entries and may be aliases. Both updates read
        the original state before either is applied. Callers manage random keys
        and any other state transitions themselves.
        """
        updates = {}

        try:
            index = self.value("index")
        except KeyError:
            index = None
        if index is not None:
            updates["index"] = index + 1

        try:
            time = self.value("time")
        except KeyError:
            time = None
        if time is not None:
            if dt is None:
                dt = self.value("dt")
            dt = _as_array(dt, "dt")
            next_time = time + dt
            if next_time.shape != time.shape:
                raise ValueError("State dt must broadcast without changing time.shape.")
            updates["time"] = next_time
        elif dt is not None:
            raise ValueError("State.step received dt but State has no time entry.")

        if not updates:
            raise ValueError("State.step found no index or time entry to advance.")
        return self.set(**updates)


class StateRef(Expression):
    """A static path into the State supplied when resolving an expression.

    The required state keyword follows the normal Expression context contract.
    A reference remains unresolved until a state is supplied.
    """

    path: str = eqx.field(static=True)

    def __init__(self, path: str):
        """Store a path without requiring the State to be available yet."""
        self.alias = None
        self.path = path

    def evaluate(self, *, state: State, **context: Any) -> Array:
        """Read the current numerical value from the supplied state."""
        return state.value(self.path)


def validate_state(model: Any, state: State) -> None:
    """Check that references in a model point to existing state entries."""
    is_ref = lambda value: isinstance(value, StateRef)
    missing = set()
    for leaf in jtu.tree_leaves(model, is_leaf=is_ref):
        if not isinstance(leaf, StateRef):
            continue
        try:
            state.value(leaf.path)
        except KeyError:
            missing.add(leaf.path)

    if missing:
        paths = ", ".join(repr(path) for path in sorted(missing))
        raise ValueError(f"State is missing referenced path(s): {paths}.")
