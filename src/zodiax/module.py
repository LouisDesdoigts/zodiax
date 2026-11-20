"""Model containers with aliases, descendant lookup, and numerical resolution.

Python reads ordinary attributes first. For a missing public name, Module checks
its aliases, then searches its children one level at a time. Aliases name a fixed
path; descendant lookup prefers populated fields, then falls back to defaults.
Within each group, the uniquely nearest match wins.
"""

from collections.abc import Mapping
from difflib import get_close_matches
from dataclasses import MISSING
import keyword
from typing import Any

import equinox as eqx

from .base import Base, _validate_mapping_keys

__all__ = ["Alias", "Module", "update", "validate_aliases"]

Alias = tuple[tuple[str, str], ...]


def _normalise_alias(value: Any) -> Alias | None:
    """Convert a mapping or pairs into immutable, consistently ordered metadata."""
    if value is None:
        return None

    if isinstance(value, Mapping):
        pairs = value.items()
    elif isinstance(value, (tuple, list)):
        if len(value) == 2 and all(isinstance(item, str) for item in value):
            pairs = [value]
        else:
            pairs = value
    else:
        raise TypeError("alias must be None, a mapping, or (name, path) pairs.")

    aliases = []
    for pair in pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise TypeError("Every alias entry must be a (name, path) pair.")
        name, path = pair
        if not isinstance(name, str) or not isinstance(path, str):
            raise TypeError("Alias names and paths must be strings.")
        if not name.isidentifier() or name.startswith("_") or keyword.iskeyword(name):
            raise ValueError(f"Alias name {name!r} must be a public Python identifier.")
        if not path or any(not part for part in path.split(".")):
            raise ValueError(f"Alias {name!r} needs a non-empty dotted path.")
        aliases.append((name, path))

    names = [name for name, _ in aliases]
    if len(names) != len(set(names)):
        raise ValueError("Alias names must be unique.")

    # Equinox stores this static field in the tree definition. Sorting gives
    # equivalent mappings the same JAX structure, regardless of insertion order.
    return tuple(sorted(aliases)) or None


def _local_alias(owner: Any, key: str) -> str | None:
    """Find a path declared on this object, without searching its children."""
    # Bypass __getattr__: asking it for 'alias' would start another alias lookup.
    # The field can still be absent partway through a handwritten constructor.
    try:
        aliases = object.__getattribute__(owner, "alias")
    except AttributeError:
        return None
    for name, path in aliases or ():
        if name == key:
            return path
    return None


def _resolve_alias(owner: Any, name: str, path: str) -> Any:
    """Follow an alias through stored, dynamic fields and container entries.

    Alias targets must also work with Base.set, so they cannot pass through a
    property, static field, another alias, or a descendant-name shortcut. Reading
    the fields directly also prevents aliases from recursively calling each other.
    """
    value = owner
    try:
        for key in path.split("."):
            if isinstance(value, Mapping):
                # Require an existing entry: defaultdict would create a missing
                # value if we only indexed it, changing the user's model.
                if key not in value:
                    raise KeyError(f"No mapping key {key!r}.")
                value = value[key]
            elif isinstance(value, (tuple, list)):
                index = int(key)
                if index < 0 or str(index) != key:
                    raise KeyError(
                        "Alias indices must be canonical non-negative integers."
                    )
                value = value[index]
            else:
                if not isinstance(value, eqx.Module):
                    raise KeyError(f"{type(value).__name__} has no dynamic fields.")
                fields = type(value).__dataclass_fields__
                field = fields.get(key)
                if field is None or key.startswith("_"):
                    raise KeyError(f"No public field {key!r}.")
                if field.metadata.get("static", False):
                    raise KeyError(f"Field {key!r} is static.")
                value = object.__getattribute__(value, key)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
        raise AttributeError(
            f"{type(owner).__name__} alias {name!r} points to unavailable "
            f"dynamic structural path {path!r}."
        ) from error
    return value


def _children(value: Any):
    """Yield stored children without invoking aliases or computed properties.

    Equinox modules are dataclasses: __dataclass_fields__ lists their declared
    fields. Reading those fields directly also supports ordinary dataclass children.
    """
    if isinstance(value, Mapping):
        yield from value.items()
    elif isinstance(value, (tuple, list)):
        yield from enumerate(value)
    else:
        for name in getattr(type(value), "__dataclass_fields__", {}):
            if name.startswith("_"):
                continue
            try:
                child = object.__getattribute__(value, name)
            except AttributeError:
                # Attribute lookup can run before every field is initialised.
                continue
            yield name, child


def _declares_attribute(value: Any, key: str) -> bool:
    """Check the class definition, without triggering descendant lookup."""
    if key in getattr(type(value), "__dataclass_fields__", {}):
        return True
    # __mro__ lists the class and its parents in Python's attribute lookup order.
    return any(key in vars(cls) for cls in type(value).__mro__)


def _attribute_names(value: Any) -> set[str]:
    """Collect public field, property, and alias names for spelling suggestions."""
    if isinstance(value, Mapping):
        return {str(name) for name in value}
    if not hasattr(type(value), "__dataclass_fields__"):
        return set()
    names = set(getattr(type(value), "__dataclass_fields__", {}))
    for cls in type(value).__mro__:
        for name, member in vars(cls).items():
            if isinstance(member, property):
                names.add(name)
    if isinstance(value, Module):
        try:
            aliases = object.__getattribute__(value, "alias")
        except AttributeError:
            aliases = None
        names.update(name for name, _ in aliases or ())
    return {name for name in names if not name.startswith("_")}


def _resolve_raised_attribute(owner: Any, key: str, *, return_path=False) -> Any:
    """Find the nearest non-default match, falling back to the nearest default.

    Only None and equal explicitly static defaults are deferred. Dynamic numbers
    remain populated, even when equal to their default: JIT may trace their values.
    Exact attributes and explicit aliases have already been handled by the caller.
    """
    available = _attribute_names(owner)
    frontier = []
    for name, child in _children(owner):
        frontier.append((child, str(name), frozenset({id(owner)})))

    default_matches = []

    # Each pass examines one complete depth. Returning at the first individual
    # match would silently choose whichever child happened to be stored first.
    while frontier:
        matches = []
        defaults = []
        following = []
        for value, path, ancestors in frontier:
            # Object identities let us stop if a container leads back to itself.
            if id(value) in ancestors:
                continue
            available.update(_attribute_names(value))

            if isinstance(value, Mapping):
                if key in value:
                    matches.append((f"{path}.{key}", value[key]))
            elif hasattr(type(value), "__dataclass_fields__"):
                if _declares_attribute(value, key):
                    try:
                        result = object.__getattribute__(value, key)
                    except AttributeError:
                        pass
                    else:
                        field = type(value).__dataclass_fields__.get(key)
                        default = MISSING if field is None else field.default
                        match = (f"{path}.{key}", result)
                        is_default = default is None and result is None

                        # Compare static metadata by value, not Python identity.
                        # Never turn an array comparison into a Python condition.
                        if field is not None and field.metadata.get("static", False):
                            if default is not MISSING:
                                is_default = (result == default) is True
                        if is_default:
                            defaults.append(match)
                        else:
                            matches.append(match)
                elif isinstance(value, Module):
                    target = _local_alias(value, key)
                    if target is not None:
                        result = _resolve_alias(value, key, target)
                        matches.append((f"{path}.{key}", result))

            # Track ancestors separately for each path. The same object can
            # occupy two sibling slots, which must still count as two matches.
            child_ancestors = ancestors | {id(value)}
            for name, child in _children(value):
                following.append((child, f"{path}.{name}", child_ancestors))

        # Remember the nearest defaults, but keep searching for a populated field.
        if not default_matches:
            default_matches = defaults
        if not following and not matches:
            matches = default_matches
        if len(matches) == 1:
            path, result = matches[0]
            return path if return_path else result
        if len(matches) > 1:
            paths = ", ".join(repr(path) for path, _ in sorted(matches))
            raise AttributeError(
                f"{type(owner).__name__} has an ambiguous attribute {key!r}; "
                f"matches {paths}. Use a qualified path."
            )
        frontier = following

    message = f"{type(owner).__name__} has no attribute {key!r}."
    suggestions = get_close_matches(key, sorted(available), n=3, cutoff=0.6)
    if suggestions:
        names = ", ".join(repr(name) for name in suggestions)
        message += f" Did you mean {names}?"
    raise AttributeError(message)


def _structural_path(owner: Any, keys: list[str]) -> list[str]:
    """Expand shortcuts before Equinox temporarily wraps leaves during updates.

    The chosen route is fixed using the original object. tree_at can then follow
    ordinary fields without repeating value-dependent default selection.
    """
    path = []
    pending = list(keys)
    while pending:
        key = pending.pop(0)
        if isinstance(owner, Module) and not _declares_attribute(owner, key):
            target = _local_alias(owner, key)
            if target is None:
                target = _resolve_raised_attribute(owner, key, return_path=True)
            pending = target.split(".") + pending
            continue
        if isinstance(owner, Mapping):
            owner = owner[key]
        elif isinstance(owner, (list, tuple)):
            owner = owner[int(key)]
        else:
            owner = getattr(owner, key)
        path.append(key)
    return path


def _validate_aliases(owner: Any) -> None:
    """Check the stored alias metadata and its targets on one completed module."""
    aliases = object.__getattribute__(owner, "alias")
    if aliases is None:
        return
    if aliases != _normalise_alias(aliases):
        raise ValueError("Aliases must be stored as consistently ordered pairs.")

    for name, path in aliases:
        if _declares_attribute(owner, name):
            raise ValueError(
                f"Alias {name!r} collides with a declared attribute on "
                f"{type(owner).__name__}."
            )
        try:
            _resolve_alias(owner, name, path)
        except AttributeError as error:
            raise ValueError(
                f"Alias {name!r} does not target an available, dynamic structural "
                f"path: {path!r}."
            ) from error


def validate_aliases(tree: Any) -> None:
    """Check every alias against the tree's current stored structure.

    Construction checks aliases once. Replacing subtrees or resolving expressions
    may remove fields from an alias path; call this explicitly to check afterwards.
    """
    pending = [tree]
    visited = set()
    while pending:
        value = pending.pop()
        if not isinstance(value, (eqx.Module, Mapping, tuple, list)):
            continue
        if id(value) in visited:
            continue
        visited.add(id(value))

        if isinstance(value, Module):
            _validate_aliases(value)
        for _, child in _children(value):
            pending.append(child)


class Module(Base):
    """An immutable model with aliases and convenient access to its descendants.

    Ordinary attributes take precedence, followed by local aliases, then the
    nearest non-default descendant, then the nearest default-valued descendant.
    Equally near matches in the selected group require a qualified path.
    Inherited Base methods use these same paths for getting and updating values.

    ``resolve`` prepares numerical fields while retaining the model's class.
    Expression subclasses instead represent a value and may resolve to an array.
    """

    # Aliases describe where parameters live; they are not numerical parameters.
    # Equinox excludes static fields from the array leaves transformed by JAX.
    alias: Alias | None = eqx.field(
        default=None, static=True, kw_only=True, converter=_normalise_alias
    )

    def __check_init__(self) -> None:
        """Equinox calls this after construction and field conversion finish."""
        _validate_mapping_keys(self)
        _validate_aliases(self)

    def __getattr__(self, key: str) -> Any:
        """Handle names that ordinary Python attribute access could not find."""
        if key.startswith("_"):
            raise AttributeError(f"{type(self).__name__} has no attribute {key!r}.")

        # A missing declared field or failing property belongs to this object.
        # Do not replace it with an unrelated, same-named value from a child.
        if _declares_attribute(self, key):
            fields = type(self).__dataclass_fields__
            if key in fields:
                raise AttributeError(
                    f"{type(self).__name__} field {key!r} has not been initialised."
                )
            return object.__getattribute__(self, key)

        target = _local_alias(self, key)
        if target is not None:
            return _resolve_alias(self, key, target)
        return _resolve_raised_attribute(self, key)

    def validate_aliases(self) -> None:
        """Check aliases throughout this model after structural changes."""
        validate_aliases(self)

    def resolve(self, *, populate: bool = True, **context: Any) -> Any:
        """Return a copy with ready numerical expressions replaced by values.

        Linked values are populated first by default. Expressions needing more
        context remain partially prepared for a later call. An ordinary Module
        retains its class; an Expression returns its result when ready.
        """
        # Expressions inherit Module, so import the numerical traversal only
        # when it is used, after both class definitions are available.
        from .numerics.expressions import resolve

        return resolve(self, populate=populate, **context)

    def populate(self) -> Any:
        """Return a copy with shared links populated, without evaluating them.

        This is the structural step already included in normal ``resolve`` calls.
        Use it directly when linked expressions should remain unevaluated.
        """
        from .numerics.links import populate

        return populate(self)


def update(
    parameters: Mapping[str, Any],
    *objects: Module,
    strict: bool = True,
    mode: str = "first",
) -> tuple[Module, ...]:
    """Distribute parameter updates across modules, returning immutable copies.

    ``mode="first"`` assigns each path to its first matching object; ``mode="all"``
    assigns it to every match. ``strict=True`` rejects paths matching no object.
    """
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping of paths to values.")
    if not all(isinstance(path, str) for path in parameters):
        raise TypeError("parameter paths must be strings.")
    if not isinstance(strict, bool):
        raise TypeError("strict must be a bool.")
    if mode not in ("first", "all"):
        raise ValueError("mode must be 'first' or 'all'.")
    if not all(isinstance(obj, Module) for obj in objects):
        raise TypeError("objects must be Zodiax Module instances.")

    # Select paths against the original object, then apply its changes together.
    # Track unmatched paths separately so first/all and strict stay independent.
    unmatched = set(parameters)
    updated = []
    for obj in objects:
        values = {}
        for path, value in parameters.items():
            if mode == "first" and path not in unmatched:
                continue
            try:
                obj.get(path, to_array=False)
            except (AttributeError, IndexError, KeyError):
                continue
            values[path] = value
            unmatched.discard(path)
        updated.append(obj.set(**values) if values else obj)

    if strict and unmatched:
        paths = ", ".join(repr(path) for path in parameters if path in unmatched)
        raise KeyError(f"Unused parameter paths: {paths}.")
    return tuple(updated)
