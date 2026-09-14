"""Build constructor-free templates from an already validated definition.

_schema owns wire-format checks. This module checks the installed classes, then
restores the recorded fields directly: constructors and converters may depend on
inputs or context that are not part of the archive.
"""

from collections import OrderedDict
from dataclasses import fields
from typing import Any

import equinox as eqx

from ._leaves import _NOT_PAYLOAD, _payload_placeholder
from ._types import _resolve_type, _type_identifier


def _build_template(definition: Any, custom_types, path: str = "root") -> Any:
    """Decode validated literals and containers, using placeholders for arrays."""
    if definition is None or type(definition) in (bool, int, float, str):
        return definition

    kind = definition["kind"]
    if kind == "complex":
        return complex(*definition["value"])
    if kind == "range":
        return range(definition["start"], definition["stop"], definition["step"])
    if kind == "slice":
        return slice(
            *(
                _build_template(definition[name], custom_types, f"{path}.{name}")
                for name in ("start", "stop", "step")
            )
        )
    if kind == "type":
        return _resolve_type(definition["type"], custom_types, path)
    placeholder = _payload_placeholder(definition, path)
    if placeholder is not _NOT_PAYLOAD:
        return placeholder

    if kind == "module":
        cls = _resolve_type(definition["type"], custom_types, path)
        if not issubclass(cls, eqx.Module):
            raise ValueError(
                f"Type {_type_identifier(cls)!r} at {path} is not a Module."
            )
        stored_fields = definition["fields"]
        stored_schema = [(field["name"], field["static"]) for field in stored_fields]
        declared_schema = [
            (field.name, bool(field.metadata.get("static", False)))
            for field in fields(cls)
        ]
        if stored_schema != declared_schema:
            raise ValueError(f"The installed class schema for {path} has changed.")

        # Equinox freezes the resulting instance normally. Bypassing construction
        # here restores stored values without reapplying field conversions.
        value = object.__new__(cls)
        for field in stored_fields:
            name = field["name"]
            child = _build_template(field["value"], custom_types, f"{path}.{name}")
            object.__setattr__(value, name, child)
        return value

    if kind in ("list", "tuple"):
        values = [
            _build_template(item, custom_types, f"{path}[{index}]")
            for index, item in enumerate(definition["items"])
        ]
        return values if kind == "list" else tuple(values)

    if kind == "mapping":
        # Schema validation permits exactly these two inert container types.
        cls = dict if definition["type"] == "builtins:dict" else OrderedDict
        values = []
        for index, entry in enumerate(definition["entries"]):
            key = _build_template(entry["key"], custom_types, f"{path}.keys[{index}]")
            item = _build_template(
                entry["value"], custom_types, f"{path}.values[{index}]"
            )
            values.append((key, item))
        return cls(values)

    if kind == "equinox_state":
        keys = []
        values = []
        for index, entry in enumerate(definition["entries"]):
            keys.append(entry["key"])
            values.append(
                _build_template(entry["value"], custom_types, f"{path}.values[{index}]")
            )
        return eqx.nn.State.tree_unflatten(tuple(keys), tuple(values))

    raise ValueError(f"Unsupported object definition kind {kind!r} at {path}.")
