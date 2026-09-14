"""Safe type identifiers and resolution for object definitions."""

import sys
from types import ModuleType


def _type_identifier(cls):
    """Return the nominal identifier recorded for a Python type."""
    module = type.__getattribute__(cls, "__module__")
    qualname = type.__getattribute__(cls, "__qualname__")
    return f"{module}:{qualname}"


def _resolve_type(identifier, custom_types, path):
    """Resolve a type without importing code named by an archive."""
    if custom_types is not None and identifier in custom_types:
        cls = custom_types[identifier]
        if not isinstance(cls, type):
            raise TypeError(f"The custom type for {identifier!r} is not a class.")
        if _type_identifier(cls) != identifier:
            raise ValueError(
                f"The custom type for {identifier!r} has nominal identifier "
                f"{_type_identifier(cls)!r}."
            )
        return cls

    # Identifier syntax was validated before any type lookup. Local classes have
    # no stable module namespace entry, so they require an explicit trusted input.
    module_name, qualname = identifier.split(":", 1)
    qualname_parts = qualname.split(".")
    if "<locals>" in qualname_parts:
        raise ValueError(
            f"Local type {identifier!r} at {path} cannot be resolved automatically. "
            "Supply like= or provide it through custom_types."
        )
    module = sys.modules.get(module_name)
    if not isinstance(module, ModuleType):
        raise ValueError(
            f"Type {identifier!r} at {path} is not available. Import its package, "
            "supply like=, or provide it through custom_types."
        )

    # Inspect namespaces directly: getattr could invoke module hooks or class
    # descriptors selected by the archive. No import or expression evaluation runs.
    namespace = ModuleType.__getattribute__(module, "__dict__")
    try:
        value = namespace[qualname_parts[0]]
        for name in qualname_parts[1:]:
            if not isinstance(value, type):
                raise KeyError(name)
            namespace = type.__getattribute__(value, "__dict__")
            value = namespace[name]
    except KeyError as error:
        raise ValueError(
            f"Type {identifier!r} at {path} is not available. Supply like= or "
            "provide it through custom_types."
        ) from error
    if not isinstance(value, type) or _type_identifier(value) != identifier:
        raise ValueError(f"Resolved value for {identifier!r} at {path} is not a type.")
    return value
