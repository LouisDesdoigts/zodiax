"""Zodiax-owned running state and expression references into it."""

from __future__ import annotations

import keyword
from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax.numpy as np
import jax.tree_util as jtu
import wadler_lindig as wl

from ..base import Module
from .arrays import as_array
from .expressions import Expression, resolve

__all__ = ["State", "StateRef", "validate_state"]


def _validate_name(name: Any, path: str) -> None:
    """Validate one concise mapping component used by a state path."""
    if not isinstance(name, str):
        raise TypeError(f"{path} state keys must be strings.")
    if not name.isidentifier() or name.startswith("_") or keyword.iskeyword(name):
        raise ValueError(
            f"{path} state key {name!r} must be a public Python identifier."
        )


def _validate_path(path: Any) -> str:
    """Return one canonical relative path into a State value mapping."""
    if not isinstance(path, str):
        raise TypeError("StateRef path must be a string.")
    parts = path.split(".")
    if not path or any(not part for part in parts):
        raise ValueError("StateRef path must be a non-empty canonical dotted path.")
    for part in parts:
        sequence_index = part.isdecimal() and str(int(part)) == part
        if sequence_index:
            continue
        _validate_name(part, "StateRef")
    return path


def _reserved_names(cls: type) -> set[str]:
    """Return declared State API names that cannot also be top-level entries."""
    return {name for base in cls.__mro__ for name in vars(base)}


def _normalise_value(value: Any, path: str) -> Any:
    """Convert numerical state leaves while retaining explicit PyTree structure."""
    if value is None:
        raise ValueError(f"{path} state value must not be None.")
    if isinstance(value, eqx.Module):
        return value
    if isinstance(value, Mapping):
        for name in value:
            _validate_name(name, path)
        normalised = {}
        for name in sorted(value):
            item = value[name]
            normalised[name] = _normalise_value(item, f"{path}.{name}")
        return normalised

    try:
        return as_array(value)
    except TypeError:
        if isinstance(value, tuple):
            items = tuple(
                _normalise_value(item, f"{path}.{index}")
                for index, item in enumerate(value)
            )
            if hasattr(value, "_fields"):
                return type(value)(*items)
            return items
        if isinstance(value, list):
            return [
                _normalise_value(item, f"{path}.{index}")
                for index, item in enumerate(value)
            ]
        raise TypeError(
            f"{path} must contain arrays, Expressions, or array PyTrees."
        ) from None


def _normalise_values(values: Mapping[str, Any], cls: type) -> dict[str, Any]:
    """Return a canonical state mapping with collision-free top-level names."""
    if not isinstance(values, Mapping):
        raise TypeError("State values must be a mapping.")
    reserved = _reserved_names(cls)
    for name in values:
        _validate_name(name, "Top-level")
        if name in reserved:
            raise ValueError(f"State key {name!r} collides with the State API.")
    normalised = {}
    for name in sorted(values):
        normalised[name] = _normalise_value(values[name], name)
    return normalised


def _leaf_signature(value: Any) -> tuple[Any, ...]:
    """Return the JAX carry-relevant signature of one PyTree leaf."""
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None or dtype is None:
        return ("python", type(value))
    return (
        "array",
        tuple(shape),
        str(dtype),
        bool(getattr(value, "weak_type", False)),
    )


def _value_signature(value: Any) -> tuple[tuple[Any, ...], ...]:
    """Return leaf signatures after PyTree topology has been checked separately."""
    return tuple(_leaf_signature(leaf) for leaf in jtu.tree_leaves(value))


def _validate_values(values: Any, cls: type) -> None:
    """Validate constructor-free reconstruction of canonical State storage."""
    if type(values) is not dict:
        raise TypeError("State values must be stored as a dict.")
    reserved = _reserved_names(cls)

    def visit(mapping: dict[str, Any], path: str, *, top: bool = False) -> None:
        if type(mapping) is not dict:
            raise TypeError(f"{path} state mappings must be stored as dicts.")
        if tuple(mapping) != tuple(sorted(mapping)):
            raise ValueError(f"{path} state keys must be stored canonically.")
        for name, value in mapping.items():
            _validate_name(name, path)
            if top and name in reserved:
                raise ValueError(f"State key {name!r} collides with the State API.")
            if isinstance(value, Mapping):
                visit(value, f"{path}.{name}")

    visit(values, "State", top=True)


def _state_pdoc(
    owner: Any,
    name: str,
    fields: list[tuple[str, Any]],
    **kwargs: Any,
) -> Any:
    """Build a compact Equinox-compatible state representation."""
    alias = getattr(owner, "alias", None)
    if alias is not None:
        fields = [("alias", alias), *fields]
    return wl.bracketed(
        begin=wl.TextDoc(f"{name}("),
        docs=wl.named_objs(fields, **kwargs),
        sep=wl.comma,
        end=wl.TextDoc(")"),
        indent=kwargs["indent"],
    )


class State(Module):
    """An immutable, array-valued running-state PyTree.

    Entries are owned by this object and updated with the ordinary Zodiax
    :meth:`Module.set` path API. A :class:`StateRef` contains only a static path and
    reads the corresponding entry when an expression is resolved with
    ``state=state``. State topology is fixed after construction, which keeps its JAX
    PyTree definition stable across iterative calculations.
    """

    values: dict[str, Any]

    def __init__(
        self,
        values: Mapping[str, Any] | None = None,
        *,
        alias: Any = None,
        **entries: Any,
    ):
        if values is None:
            values = {}
        elif not isinstance(values, Mapping):
            raise TypeError("values must be a mapping or None.")
        overlap = set(values).intersection(entries)
        if overlap:
            names = ", ".join(repr(name) for name in sorted(overlap))
            raise ValueError(f"State entries were provided twice: {names}.")

        combined = dict(values)
        combined.update(entries)
        self.alias = alias
        self.values = _normalise_values(combined, type(self))

    def value(self, path: str) -> Any:
        """Return one raw entry selected by a concise path relative to State."""
        path = _validate_path(path)
        if path.split(".", 1)[0] in _reserved_names(type(self)):
            raise KeyError(f"State has no value at {path!r}.")
        try:
            return self.get(path, to_array=False)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
            raise KeyError(f"State has no value at {path!r}.") from error

    def ref(self, path: str) -> StateRef:
        """Return a static expression reference to an existing state path."""
        path = _validate_path(path)
        self.value(path)
        return StateRef(path)

    def set(
        self,
        parameters: Any = None,
        values: Any = None,
        /,
        **updates: Any,
    ) -> State:
        """Apply the normal Zodiax path update API and normalise new values."""
        updated = super().set(parameters, values, **updates)
        normalised = _normalise_values(updated.values, type(updated))
        if jtu.tree_structure(normalised) != jtu.tree_structure(self.values):
            raise ValueError(
                "State.set cannot add, remove, or change the structure of entries."
            )
        if _value_signature(normalised) != _value_signature(self.values):
            raise ValueError(
                "State.set cannot change the shape, dtype, or weak type of entries."
            )
        return eqx.tree_at(lambda state: state.values, updated, normalised)

    def step(self, dt: Any = None, **context: Any) -> State:
        """Advance any conventional ``index`` and ``time`` entries.

        An available ``index`` is incremented. An available ``time`` is advanced by
        explicit or stored ``dt``. A ``key`` entry remains a stable root from which
        consumers can derive per-step keys using the current index. All names may be
        ordinary State aliases. The transition is pure and does not sample on behalf
        of a caller.
        """
        if "state" in context:
            raise TypeError("State.step supplies its own state resolution context.")

        updates = {}
        try:
            index = self.value("index")
        except KeyError:
            pass
        else:
            index = np.asarray(resolve(index, state=self, **context))
            if index.ndim != 0 or not np.issubdtype(index.dtype, np.integer):
                raise TypeError("State index must be a scalar integer array.")
            updates["index"] = index + np.asarray(1, dtype=index.dtype)

        try:
            time = self.value("time")
        except KeyError:
            if dt is not None:
                raise ValueError("State.step received dt but State has no time entry.")
        else:
            if dt is None:
                try:
                    dt = self.value("dt")
                except KeyError as error:
                    raise ValueError(
                        "State.step requires dt when State contains time."
                    ) from error

            time = np.asarray(resolve(time, state=self, **context))
            dt = np.asarray(resolve(dt, state=self, **context))
            if np.iscomplexobj(time) or not np.issubdtype(time.dtype, np.inexact):
                raise TypeError("State time must be a real floating array.")
            if np.iscomplexobj(dt) or not np.issubdtype(dt.dtype, np.inexact):
                raise TypeError("State dt must be a real floating array.")
            next_time = time + dt
            if next_time.shape != time.shape:
                raise ValueError("State dt must broadcast without changing time.shape.")
            updates["time"] = next_time

        if not updates:
            raise ValueError("State.step found no index or time entry to advance.")
        return self.set(**updates)

    def __pdoc__(self, **kwargs: Any) -> Any:
        return _state_pdoc(self, "State", list(self.values.items()), **kwargs)

    def __zodiax_validate__(self) -> None:
        _validate_values(self.values, type(self))


class StateRef(Expression):
    """A static path to a value owned by a call-local :class:`State`."""

    path: str = eqx.field(static=True, repr=False)

    def __init__(self, path: str):
        self.alias = None
        self.path = _validate_path(path)

    def evaluate(self, *, state: State, **context: Any) -> Any:
        """Read and recursively resolve the referenced state value."""
        if not isinstance(state, State):
            raise TypeError("state must be a zodiax State.")
        value = state.value(self.path)
        return resolve(value, state=state, **context)

    def __pdoc__(self, **kwargs: Any) -> Any:
        return _state_pdoc(self, "StateRef", [("path", self.path)], **kwargs)

    def __zodiax_validate__(self) -> None:
        if self.path != _validate_path(self.path):
            raise ValueError("StateRef path must be stored canonically.")


def validate_state(model: Any, state: State) -> None:
    """Validate that every StateRef in a model or State resolves to an entry."""
    if not isinstance(state, State):
        raise TypeError("state must be a zodiax State.")

    is_ref = lambda value: isinstance(value, StateRef)
    references = [
        value
        for value in jtu.tree_leaves((model, state.values), is_leaf=is_ref)
        if is_ref(value)
    ]
    missing = []
    for reference in references:
        try:
            state.value(reference.path)
        except KeyError:
            missing.append(reference.path)

    if missing:
        paths = ", ".join(repr(path) for path in sorted(set(missing)))
        raise ValueError(f"State is missing referenced path(s): {paths}.")
