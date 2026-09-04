"""Opt-in semantic validation after constructor-free reconstruction."""

from collections.abc import Mapping
from dataclasses import fields
from typing import Any

import equinox as eqx

from ..base import Base, Module, _validate_aliases, _validate_mapping_keys

_VALIDATION_HOOK = "__zodiax_validate__"


def _bound_validation_hooks(value: eqx.Module):
    """Resolve every class-local hook from base to derived class."""
    hooks = []
    for cls in reversed(type(value).__mro__):
        namespace = type.__getattribute__(cls, "__dict__")
        if _VALIDATION_HOOK in namespace:
            descriptor = namespace[_VALIDATION_HOOK]
            getter = getattr(descriptor, "__get__", None)
            hook = descriptor if getter is None else getter(value, type(value))
            hooks.append((cls, hook))
    return hooks


def _validate_rebuilt(tree: Any) -> None:
    """Call opt-in validation hooks throughout a reconstructed object graph."""
    visited: set[int] = set()

    def visit(value: Any, path: str) -> None:
        if not isinstance(value, (eqx.Module, eqx.nn.State, Mapping, tuple, list)):
            return
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)

        if isinstance(value, eqx.Module):
            for field in fields(value):
                child = object.__getattribute__(value, field.name)
                visit(child, f"{path}.{field.name}")

            if isinstance(value, Base):
                try:
                    _validate_mapping_keys(value)
                except Exception as error:
                    raise ValueError(
                        f"Path validation failed at {path} "
                        f"({type(value).__name__})."
                    ) from error

            # Alias validation is a Module invariant, not an optional subclass hook.
            # Running it centrally prevents a numerical __zodiax_validate__ method
            # from shadowing the inherited Module validation contract.
            if isinstance(value, Module):
                try:
                    _validate_aliases(value)
                except Exception as error:
                    raise ValueError(
                        f"Alias validation failed at {path} "
                        f"({type(value).__name__})."
                    ) from error

            # Each declaring class owns its local invariant. Running every hook
            # base-to-derived prevents a subclass validator from shadowing parent
            # storage contracts; hooks should therefore not call ``super()``.
            for declaring_class, hook in _bound_validation_hooks(value):
                if not callable(hook):
                    raise TypeError(
                        f"{declaring_class.__name__}.{_VALIDATION_HOOK} must be "
                        "callable."
                    )
                try:
                    hook()
                except Exception as error:
                    raise ValueError(
                        f"Object validation failed at {path} "
                        f"({type(value).__name__}, declared by "
                        f"{declaring_class.__name__})."
                    ) from error
            return

        if isinstance(value, eqx.nn.State):
            try:
                children, _ = value.tree_flatten()
            except ValueError as error:
                raise ValueError(f"Invalid Equinox State at {path}.") from error
            for index, child in enumerate(children):
                visit(child, f"{path}.values[{index}]")
            return

        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(child, f"{path}[{key!r}]")
            return

        for index, child in enumerate(value):
            visit(child, f"{path}[{index}]")

    visit(tree, "root")
