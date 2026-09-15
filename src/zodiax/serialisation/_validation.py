"""Validate Zodiax model paths after constructor-free reconstruction.

Archive schema validation owns representation; these checks own Module's public
path and alias contract. Arbitrary downstream constructor invariants are not rerun.
"""

from collections.abc import Mapping
from dataclasses import fields
from typing import Any

import equinox as eqx

from ..base import _validate_mapping_keys
from ..module import Module, _validate_aliases


def _validate_modules(tree: Any) -> None:
    """Check Module paths and aliases, including Modules inside static metadata."""
    visited: set[int] = set()

    def visit(value: Any, path: str) -> None:
        if not isinstance(value, (eqx.Module, eqx.nn.State, Mapping, tuple, list)):
            return
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)

        if isinstance(value, eqx.Module):
            # Read stored fields directly: descendant lookup must not supply a
            # missing field, and computed properties are not archive state.
            for field in fields(value):
                child = object.__getattribute__(value, field.name)
                visit(child, f"{path}.{field.name}")

            # Base permits ordinary metadata keys, including dots. The stronger
            # structural-path contract belongs specifically to Module.
            if isinstance(value, Module):
                try:
                    _validate_mapping_keys(value)
                    _validate_aliases(value)
                except (AttributeError, KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid Module paths or aliases at {path} "
                        f"({type(value).__name__}): {error}"
                    ) from error
            return

        if isinstance(value, eqx.nn.State):
            children, _ = value.tree_flatten()
        elif isinstance(value, Mapping):
            children = value.values()
        else:
            children = value
        for index, child in enumerate(children):
            visit(child, f"{path}[{index}]")

    visit(tree, "root")
