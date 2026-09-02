from collections.abc import Mapping
from difflib import get_close_matches
import keyword
from typing import Any

import equinox as eqx
import jax.numpy as np
import jax.tree as jtu
from jax import Array, lax

__all__ = [
    "Alias",
    "field",
    "Module",
    "update",
    "validate_aliases",
    "build_wrapper",
    "EquinoxWrapper",
    "WrapperHolder",
]

PyTree = dict | list | tuple | eqx.Module
Params = str | list[str] | tuple[str] | dict[str, Any]
Values = Any | list[Any] | tuple[Any]
Alias = tuple[tuple[str, str], ...]

_NO_ALIAS = object()
_RUNTIME_FIELD = "zodiax_runtime"


def field(*, runtime: bool = False, metadata: Any = None, **kwargs: Any) -> Any:
    """Declare an Equinox field with optional runtime-resolution protection.

    A field declared with ``runtime=True`` is treated as an opaque subtree by
    :meth:`Module.resolve` unless that call explicitly supplies ``runtime=True``.
    This lets an owning module retain coordinate- or input-dependent numerical
    definitions until it has the authoritative local context required to evaluate
    them.

    All remaining arguments are forwarded to :func:`equinox.field`.
    """
    if type(runtime) is not bool:
        raise TypeError("runtime must be a bool.")
    if runtime and bool(kwargs.get("static", False)):
        raise ValueError("A runtime field cannot also be static.")

    metadata = {} if metadata is None else dict(metadata)
    if _RUNTIME_FIELD in metadata:
        raise ValueError(f"metadata must not define {_RUNTIME_FIELD!r}.")
    if runtime:
        metadata[_RUNTIME_FIELD] = True
    return eqx.field(metadata=metadata, **kwargs)


def _is_runtime_field(owner: Any, name: str) -> bool:
    """Return whether ``name`` is runtime-protected on ``owner``."""
    definition = getattr(type(owner), "__dataclass_fields__", {}).get(name)
    if definition is None:
        return False
    return bool(definition.metadata.get(_RUNTIME_FIELD, False))


def _normalise_alias(value: Any) -> Alias | None:
    """Normalise user-facing alias inputs to canonical static metadata."""
    if value is None:
        return None

    if isinstance(value, Mapping):
        pairs = list(value.items())
    elif isinstance(value, (tuple, list)):
        if len(value) == 2 and all(isinstance(item, str) for item in value):
            pairs = [value]
        else:
            pairs = list(value)
    else:
        raise TypeError(
            "alias must be None, a mapping, one (name, path) pair, or a "
            "sequence of pairs."
        )

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
            raise ValueError(
                f"Alias {name!r} must target a non-empty dotted path without "
                "empty components."
            )
        aliases.append((name, path))

    names = [name for name, _ in aliases]
    if len(names) != len(set(names)):
        raise ValueError("Alias names must be unique.")
    # Alias entries are mapping-like metadata. Canonical order avoids distinct JAX
    # tree definitions for equivalent mappings supplied in different orders.
    return tuple(sorted(aliases)) or None


def _local_alias(owner: Any, key: str) -> str | object:
    """Return one alias target declared directly on ``owner``."""
    # Hand-written Equinox constructors initialise fields in ordinary Python order.
    # Missing-attribute lookup may therefore occur before the inherited alias field
    # has been assigned; treat that transient state exactly like ``alias=None``.
    try:
        aliases = object.__getattribute__(owner, "alias")
    except AttributeError:
        aliases = None
    if aliases is not None:
        for name, path in aliases:
            if name == key:
                return path
    return _NO_ALIAS


def _sequence_index(value: tuple | list, key: str, path: str) -> int:
    """Parse one canonical, non-negative sequence path component."""
    try:
        index = int(key)
    except (TypeError, ValueError) as error:
        raise KeyError(f"{path} must use an integer sequence index.") from error
    if index < 0 or str(index) != key:
        raise KeyError(f"{path} must use a canonical non-negative sequence index.")
    if index >= len(value):
        raise KeyError(f"{path} is outside a sequence of length {len(value)}.")
    return index


def _resolve_structural_path(owner: Any, path: str) -> Any:
    """Resolve a canonical, updateable path without raised or property lookup."""
    value = owner
    traversed = []
    for key in path.split("."):
        traversed.append(key)
        current_path = ".".join(traversed)

        if isinstance(value, Mapping):
            if key not in value:
                raise KeyError(f"mapping has no key {key!r} at {current_path!r}.")
            value = value[key]
            continue

        if isinstance(value, (tuple, list)):
            value = value[_sequence_index(value, key, current_path)]
            continue

        if not isinstance(value, eqx.Module):
            raise KeyError(
                f"{current_path!r} traverses non-structural " f"{type(value).__name__}."
            )

        fields = getattr(type(value), "__dataclass_fields__", {})
        field = fields.get(key)
        if field is None or key.startswith("_"):
            raise KeyError(
                f"{type(value).__name__} has no public field {key!r} at "
                f"{current_path!r}."
            )
        if field.metadata.get("static", False):
            raise KeyError(
                f"Alias path {path!r} crosses static field {current_path!r}."
            )
        try:
            value = object.__getattribute__(value, key)
        except AttributeError as error:
            raise KeyError(
                f"Field {current_path!r} has not been initialised."
            ) from error

    return value


def _resolve_local_alias(owner: Any, key: str, path: str) -> Any:
    """Resolve one local alias while diagnosing topology-changing stale paths."""
    try:
        return _resolve_structural_path(owner, path)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise AttributeError(
            f"{type(owner).__name__} alias {key!r} points to unavailable "
            f"structural path {path!r}."
        ) from error


def _unpack(dict: dict) -> dict:
    """
    Unpacks a dictionary with potentially nested keys into a dictionary with a
    one to one mapping of keys to values.
    """
    # Check for any tuples in the parameters and cast to lists.
    unpacked = {}
    for key, value in dict.items():
        if isinstance(key, tuple):
            for param in key:
                unpacked[param] = value
        else:
            unpacked[key] = value
    return unpacked


def _get_leaf(pytree: PyTree, param: Params) -> Any:
    """
    A helper function designed to recurse down a pytree following the param,
    returning the leaf at the end of the param.

    Base case: len(param) == 1
        In this case the leaf referred to by the single param entry is
        returned (and hence recursively sent up to the initial call).

    Recursive case: len(param) > 1
        In this case the function takes the Base like object referred to
        by the first entry in param, and recursively calls this function
        with this new pytree object and the param without the first entry.

    Parameters
    ----------
    pytree : PyTree
        The pytree object to recurse through.
    param : Params
        The param to recurse down.

    Returns
    -------
    leaf : Any
        The leaf object specified at the end of the param object.
    """
    key = param[0]
    # ``Module`` deliberately preserves its informative ``AttributeError`` for
    # missing or ambiguous raised attributes. ``hasattr`` would swallow that error
    # and turn it into the less useful generic ``KeyError`` below.
    if isinstance(pytree, Module):
        pytree = _resolve_module_attribute(pytree, key)
    elif isinstance(pytree, Mapping):
        pytree = pytree[key]
    elif isinstance(pytree, (list, tuple)):
        pytree = pytree[int(key)]
    elif hasattr(pytree, key):
        pytree = getattr(pytree, key)
    else:
        raise KeyError("key: {} not found in object: {}".format(key, type(pytree)))

    # Return param if at the end of param, else recurse
    return pytree if len(param) == 1 else _get_leaf(pytree, param[1:])


def _public_fields(value: Any) -> tuple[str, ...]:
    """Return the public dataclass fields declared by ``value`` in stable order."""
    fields = getattr(type(value), "__dataclass_fields__", {})
    return tuple(name for name in fields if not name.startswith("_"))


def _is_dataclass_node(value: Any) -> bool:
    """Return whether ``value`` exposes dataclass field metadata."""
    return hasattr(type(value), "__dataclass_fields__")


def _public_properties(value: Any) -> set[str]:
    """Return public property names exposed by ``value``'s class hierarchy."""
    names = set()
    for cls in type(value).__mro__:
        names.update(
            name
            for name, member in vars(cls).items()
            if not name.startswith("_") and isinstance(member, property)
        )
    return names


def _declares_attribute(value: Any, key: str) -> bool:
    """Return whether a dataclass node declares ``key`` without raised lookup."""
    if key in getattr(type(value), "__dataclass_fields__", {}):
        return True
    return any(key in vars(cls) for cls in type(value).__mro__)


def _resolve_module_attribute(owner: Any, key: str) -> Any:
    """Resolve a direct field, local alias, or uniquely raised descendant."""
    target = _local_alias(owner, key)
    fields = getattr(type(owner), "__dataclass_fields__", {})
    class_attribute = any(key in vars(cls) for cls in type(owner).__mro__)
    if key in fields or class_attribute:
        if target is not _NO_ALIAS:
            raise AttributeError(
                f"{type(owner).__name__} alias {key!r} collides with a declared "
                "attribute. Reconstruct the module with a different alias."
            )
        try:
            return object.__getattribute__(owner, key)
        except AttributeError as error:
            if key in fields:
                raise AttributeError(
                    f"{type(owner).__name__} field {key!r} has not been " "initialised."
                ) from error
            raise

    if target is not _NO_ALIAS:
        return _resolve_local_alias(owner, key, target)
    return _resolve_raised_attribute(owner, key)


def _children(value: Any, path: str, ancestors: frozenset[int]):
    """Yield public child nodes together with diagnostic paths and cycle guards."""
    identity = id(value)
    if identity in ancestors:
        return
    ancestors = ancestors | {identity}

    if isinstance(value, Mapping):
        for name, child in value.items():
            yield child, f"{path}.{name}", ancestors
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield child, f"{path}.{index}", ancestors
    else:
        for name in _public_fields(value):
            try:
                child = object.__getattribute__(value, name)
            except AttributeError:
                continue
            yield child, f"{path}.{name}", ancestors


def _raised_candidates(owner: Any, key: str) -> tuple[list[tuple[str, Any]], set[str]]:
    """Find the nearest named descendants and names used for missing-name hints."""
    fields = _public_fields(owner)
    available = set(fields) | _public_properties(owner)
    frontier = []
    for name in fields:
        try:
            child = object.__getattribute__(owner, name)
        except AttributeError:
            continue
        frontier.append((child, name, frozenset({id(owner)})))

    # Breadth-first lookup gives nearer descendants precedence while retaining all
    # matches at that depth so ambiguous shorthand is never order-dependent.
    while frontier:
        matches = []
        following = []
        for value, path, ancestors in frontier:
            if isinstance(value, Mapping):
                available.update(str(name) for name in value)
                if key in value:
                    matches.append((f"{path}.{key}", value[key]))
            elif not isinstance(value, (tuple, list)) and _is_dataclass_node(value):
                available.update(_public_fields(value))
                available.update(_public_properties(value))
                if _declares_attribute(value, key):
                    try:
                        result = object.__getattribute__(value, key)
                    except AttributeError:
                        pass
                    else:
                        matches.append((f"{path}.{key}", result))

            following.extend(_children(value, path, ancestors))

        if matches:
            return matches, available
        frontier = following

    return [], available


def _resolve_raised_attribute(owner: Any, key: str) -> Any:
    """Resolve a unique descendant attribute or raise a deterministic diagnostic."""
    if key.startswith("_"):
        raise AttributeError(f"{type(owner).__name__} has no attribute {key!r}.")
    if key in getattr(type(owner), "__dataclass_fields__", {}):
        raise AttributeError(
            f"{type(owner).__name__} field {key!r} has not been initialised."
        )

    matches, available = _raised_candidates(owner, key)
    if len(matches) == 1:
        return matches[0][1]
    if len(matches) > 1:
        paths = ", ".join(repr(path) for path, _ in sorted(matches))
        raise AttributeError(
            f"{type(owner).__name__} has an ambiguous attribute {key!r}; "
            f"matches {paths}. Use a qualified path."
        )

    message = f"{type(owner).__name__} has no attribute {key!r}."
    suggestions = get_close_matches(key, sorted(available), n=3, cutoff=0.6)
    if suggestions:
        names = ", ".join(repr(name) for name in suggestions)
        message += f" Did you mean {names}?"
    raise AttributeError(message)


def _validate_aliases(owner: Any) -> None:
    """Validate canonical local aliases against one completed module topology."""
    aliases = object.__getattribute__(owner, "alias")
    if aliases is None:
        return

    try:
        canonical = _normalise_alias(aliases)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{type(owner).__name__}.alias is not a valid alias specification."
        ) from error
    if aliases != canonical:
        raise ValueError(
            f"{type(owner).__name__}.alias must be stored in canonical order."
        )

    for name, path in aliases:
        if _declares_attribute(owner, name):
            raise ValueError(
                f"Alias {name!r} collides with a declared attribute on "
                f"{type(owner).__name__}."
            )

        try:
            _resolve_structural_path(owner, path)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Alias {name!r} does not target an available, dynamic structural "
                f"path: {path!r}."
            ) from error


def validate_aliases(tree: Any) -> None:
    """Validate every local alias against the current unresolved tree topology.

    Ordinary construction validates each module once. This explicit traversal is
    useful after whole-subtree replacement, link population, or expression
    resolution, any of which may make a previously valid topology-bound path stale.
    """
    visited: set[int] = set()

    def visit(value: Any) -> None:
        if not isinstance(value, (eqx.Module, Mapping, tuple, list)):
            return
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)

        if isinstance(value, Module):
            _validate_aliases(value)

        if isinstance(value, Mapping):
            children = value.values()
        elif isinstance(value, (tuple, list)):
            children = value
        else:
            children = (
                object.__getattribute__(value, field)
                for field in _public_fields(value)
                if field != "alias"
            )
        for child in children:
            visit(child)

    visit(tree)


def _get_leaves(pytree: PyTree, parameters: list) -> list:
    """
    Returns a list of leaves specified by the parameters.

    Parameters
    ----------
    pytree : PyTree
        The pytree object to recurse through.
    parameters : list
        A list/tuple of nested parameters. Note param objects can only be
        nested a single time.

    Returns
    -------
    leaves : list
        The list of leaf objects specified by the parameters object
    """
    return [_get_leaf(pytree, param) for param in parameters]


def _unwrap(parameters: Params, values_in: list = None) -> list:
    """
    Unwraps the provided parameters into the correct list-based format for the
    _get_leaves and _get_leaf methods, returning a single dimensional list
    of input parameters.

    Parameters
    ----------
    parameters : Params
        A list/tuple of nested parameters to unwrap.
    values_in : list = None
        The list of values to be unwrapped.

    Returns
    -------
    parameters, values : list, list
        The list of unwrapped parameters or parameters and values.
    """
    # Initialise empty lists
    parameters_out, values_out = [], []

    # If values are provided, apply transformation to both
    if values_in is not None:
        # Make sure values is list
        values = (
            list(values_in) if isinstance(values_in, (list, tuple)) else [values_in]
        )

        # Repeat values to match length of parameters
        if len(values) == 1:
            values = values * len(parameters)

        # Ensure correct length
        if len(values) != len(parameters):
            raise ValueError(
                "The number of values must match the number of parameters."
            )

        # Iterate over parameters and values
        for param, value in zip(parameters, values):
            # Recurse and add in the case of list inputs
            if isinstance(param, (list, tuple)):
                new_parameters, new_values = _unwrap(param, value)
                parameters_out += new_parameters
                values_out += new_values

            # Params must already be absolute
            else:
                parameters_out.append(param)
                values_out.append(value)
        return parameters_out, values_out

    # Just parameters provided
    else:
        # Iterate over parameters
        for param in parameters:
            # Recurse and add in the case of list inputs
            if isinstance(param, (list, tuple)):
                new_parameters = _unwrap(param)
                parameters_out += new_parameters

            # Params must already be absolute
            else:
                parameters_out.append(param)
        return parameters_out


def _format(parameters: Params, values: list = None) -> list:
    """
    Formats the provided parameters into the correct list-based format for the
    _get_leaves and _get_leaf methods, returning a single dimensional list
    of input parameters.

    Parameters
    ----------
    parameters : Params
        A list/tuple of nested parameters to unwrap.
    values : list = None
        The list of values to be unwrapped.

    Returns
    -------
    parameters, values : list, list
        The list of unwrapped parameters or parameters and values.
    """
    # Nested/multiple inputs
    if isinstance(parameters, (list, tuple)):
        parameters = list(parameters)

        # If there is nesting, ensure correct dimensions
        if (
            len(parameters) > 1
            and values is not None
            and True in [isinstance(p, (list, tuple)) for p in parameters]
        ):
            assert isinstance(values, (list, tuple)) and len(values) == len(
                parameters
            ), (
                "If a list of parameters is provided, the list of values must be "
                "of equal length."
            )

        # It's a list - iterate and unbind all the keys
        if values is not None:
            flat_parameters, new_values = _unwrap(parameters, values)
        else:
            flat_parameters = _unwrap(parameters)

        # Turn into separate strings
        new_parameters = [
            param.split(".") if "." in param else [param] for param in flat_parameters
        ]

    # Un-nested/singular input
    else:
        # Turn into separate strings
        new_parameters = [parameters.split(".") if "." in parameters else [parameters]]
        new_values = [values]

    # Return
    return new_parameters if values is None else (new_parameters, new_values)


def _normalise_mutation_inputs(
    parameters: Params = None,
    values: Values = None,
    updates: dict = None,
    method_name: str = "method",
    require_values: bool = False,
) -> tuple:
    """
    Normalises mutation inputs for methods that can accept either:
    - (parameters, values)
    - a mapping of parameter->value
    - keyword arguments of parameter=value

    Parameters
    ----------
    require_values : bool, optional
        If True, raises a TypeError when the resolved values are None.
    """
    updates = {} if updates is None else updates

    if len(updates) > 0:
        if parameters is not None or values is not None:
            raise TypeError(
                f"{method_name}() received mixed input styles. Use either "
                "(parameters, values), a mapping, or keyword arguments."
            )
        mapping = _unpack(updates)
        parameters, values = list(mapping.keys()), list(mapping.values())
    elif isinstance(parameters, dict):
        if values is not None:
            raise TypeError(
                f"{method_name}() received both a mapping and values. "
                "Provide only one input style."
            )
        mapping = _unpack(parameters)
        parameters, values = list(mapping.keys()), list(mapping.values())
    else:
        if parameters is None:
            raise TypeError(
                f"{method_name}() requires input via (parameters, values), "
                "a mapping, or keyword arguments."
            )

    if require_values and values is None:
        raise TypeError(
            f"{method_name}() missing values. Use (parameters, values), a mapping, "
            "or keyword arguments."
        )

    return parameters, values


class _Base(eqx.Module):
    """
    Extend the Equinox.Module class to give a user-friendly 'param based' API
    for working with pytrees by adding a series of methods used to interface
    with the leaves of the pytree using parameters.
    """

    def save(self, file_or_path: Any) -> None:
        """Save this object to a validated Zodiax archive."""
        from .serialisation import save

        save(file_or_path, self)

    def load(self, file_or_path: Any) -> Any:
        """Load an archive using this object as its structural template."""
        from .serialisation import load

        return load(file_or_path, like=self)

    def get(
        self: PyTree,
        parameters: Params,
        as_dict: bool = False,
        to_array: bool = True,
    ) -> Any:
        """
        Get the leaf specified by param.

        Parameters
        ----------
        parameters : Params
            Parameter selector. Supported forms are:
            - ``"param"`` (single path string)
            - ``["param", "b.param"]`` (list of path strings)
            - ``("param", "b.param")`` (tuple of path strings)
            - Interleaved list/tuple nesting of path strings.
        as_dict : bool = False
            If True, returns a dictionary mapping parameters to their values. If False,
            returns a list of values in the same order as the input parameters.
        to_array : bool = True
            If True, converts the output to an float Array for later jax
            transformations. If False, returns the raw value (which may be a scalar or
            other type).

        Returns
        -------
        values : Any
            The value(s) corresponding to the input parameter(s). If `as_dict`
            is True, returns a dictionary mapping parameters to their values. If False,
            returns a list of values in the same order as the input parameters. If only
            a single parameter is provided and `as_dict` is False, returns the
            single value corresponding to that parameter.
        """
        new_parameters = _format(parameters)
        values = _get_leaves(self, new_parameters)
        if to_array:
            values = jtu.map(lambda x: np.array(x, float), values)
        if as_dict:
            keys = [".".join(param) for param in new_parameters]
            return dict(zip(keys, values))
        return values[0] if len(new_parameters) == 1 else values

    def set(
        self: PyTree,
        parameters: Params = None,
        values: Values = None,
        /,
        **updates,
    ) -> PyTree:
        """
        Set the leaves specified by parameters with values.

        Parameters
        ----------
        parameters : Params, optional
            Parameter selector for positional style. Supported forms are:
            - ``"param"``
            - list/tuple of path strings
            - interleaved list/tuple nesting of path strings
            - a mapping ``{path: value}`` (dictionary style)
        values : Values, optional
            Values for positional style ``set(parameters, values)``.
            Can be a scalar, list, or tuple matching ``parameters``.
        **updates
            Keyword update style, e.g. ``set(param=1.0)`` and nested via
            ``set(**{"b.param": 2.0})``.

        Returns
        -------
        pytree : PyTree
            The pytree with leaves specified by parameters updated with values.
        """
        parameters, values = _normalise_mutation_inputs(
            parameters,
            values,
            updates=updates,
            method_name="set",
        )

        # Allow explicit None values
        if values is None:
            values = [None]
            if isinstance(parameters, str):
                parameters = [parameters]
        new_parameters, new_values = _format(parameters, values)

        # Define 'where' function and update pytree
        def leaves_fn(pytree):
            return _get_leaves(pytree, new_parameters)

        return eqx.tree_at(
            leaves_fn, self, new_values, is_leaf=lambda leaf: leaf is None
        )

    def add(
        self: PyTree,
        parameters: Params = None,
        values: Values = None,
        /,
        **updates,
    ) -> PyTree:
        """
        Add to the leaves specified by parameters with values.

        Parameters
        ----------
        parameters : Params, optional
            Parameter selector or mapping. Supported forms are:
            - path string
            - list/tuple (including interleaved nesting) of path strings
            - mapping ``{path: value}``
        values : Values, optional
            Values for positional style ``add(parameters, values)``.
        **updates
            Keyword update style, e.g. ``add(param=1.0)`` and nested via
            ``add(**{"b.param": 2.0})``.

        Returns
        -------
        pytree : PyTree
            The pytree with values added to leaves specified by parameters.
        """
        parameters, values = _normalise_mutation_inputs(
            parameters,
            values,
            updates=updates,
            method_name="add",
            require_values=True,
        )
        new_parameters, new_values = _format(parameters, values)
        new_values = [
            leaf + value
            for value, leaf in zip(new_values, _get_leaves(self, new_parameters))
        ]

        # Define 'where' function and update pytree
        def leaves_fn(pytree):
            return _get_leaves(pytree, new_parameters)

        return eqx.tree_at(
            leaves_fn, self, new_values, is_leaf=lambda leaf: leaf is None
        )

    def multiply(
        self: PyTree,
        parameters: Params = None,
        values: Values = None,
        /,
        **updates,
    ) -> PyTree:
        """
        Multiplies the leaves specified by parameters with values.

        Parameters
        ----------
        parameters : Params, optional
            Parameter selector or mapping (string, list/tuple paths, nested
            list/tuple paths, or ``{path: value}``).
        values : Values, optional
            Values for positional style ``multiply(parameters, values)``.
        **updates
            Keyword update style, e.g. ``multiply(param=2.0)`` and nested via
            ``multiply(**{"b.param": 3.0})``.

        Returns
        -------
        pytree : PyTree
            The pytree with values multiplied by leaves specified by parameters.
        """
        parameters, values = _normalise_mutation_inputs(
            parameters,
            values,
            updates=updates,
            method_name="multiply",
            require_values=True,
        )
        new_parameters, new_values = _format(parameters, values)
        new_values = [
            leaf * value
            for value, leaf in zip(new_values, _get_leaves(self, new_parameters))
        ]

        # Define 'where' function and update pytree
        def leaves_fn(pytree):
            return _get_leaves(pytree, new_parameters)

        return eqx.tree_at(
            leaves_fn, self, new_values, is_leaf=lambda leaf: leaf is None
        )

    def divide(
        self: PyTree,
        parameters: Params = None,
        values: Values = None,
        /,
        **updates,
    ) -> PyTree:
        """
        Divides the leaves specified by parameters with values.

        Parameters
        ----------
        parameters : Params, optional
            Parameter selector or mapping (string, list/tuple paths, nested
            list/tuple paths, or ``{path: value}``).
        values : Values, optional
            Values for positional style ``divide(parameters, values)``.
        **updates
            Keyword update style, e.g. ``divide(param=2.0)`` and nested via
            ``divide(**{"b.param": 4.0})``.

        Returns
        -------
        pytree : PyTree
            The pytree with values divided by leaves specified by parameters.
        """
        parameters, values = _normalise_mutation_inputs(
            parameters,
            values,
            updates=updates,
            method_name="divide",
            require_values=True,
        )
        new_parameters, new_values = _format(parameters, values)
        new_values = [
            leaf / value
            for value, leaf in zip(new_values, _get_leaves(self, new_parameters))
        ]

        # Define 'where' function and update pytree
        def leaves_fn(pytree):
            return _get_leaves(pytree, new_parameters)

        return eqx.tree_at(
            leaves_fn, self, new_values, is_leaf=lambda leaf: leaf is None
        )

    def power(
        self: PyTree,
        parameters: Params = None,
        values: Values = None,
        /,
        **updates,
    ) -> PyTree:
        """
        Raises the leaves specified by parameters to the power of values.

        Parameters
        ----------
        parameters : Params, optional
            Parameter selector or mapping (string, list/tuple paths, nested
            list/tuple paths, or ``{path: value}``).
        values : Values, optional
            Values for positional style ``power(parameters, values)``.
        **updates
            Keyword update style, e.g. ``power(param=3.0)`` and nested via
            ``power(**{"b.param": 2.0})``.

        Returns
        -------
        pytree : PyTree
            The pytree with the leaves specified by parameters raised to the power
            of values.
        """
        parameters, values = _normalise_mutation_inputs(
            parameters,
            values,
            updates=updates,
            method_name="power",
            require_values=True,
        )
        new_parameters, new_values = _format(parameters, values)
        new_values = [
            leaf**value
            for value, leaf in zip(new_values, _get_leaves(self, new_parameters))
        ]

        # Define 'where' function and update pytree
        def leaves_fn(pytree):
            return _get_leaves(pytree, new_parameters)

        return eqx.tree_at(
            leaves_fn, self, new_values, is_leaf=lambda leaf: leaf is None
        )

    def min(
        self: PyTree,
        parameters: Params = None,
        values: Values = None,
        /,
        **updates,
    ) -> PyTree:
        """
        Updates the leaves specified by parameters with the minimum value of the
        leaves and values.

        Parameters
        ----------
        parameters : Params, optional
            Parameter selector or mapping (string, list/tuple paths, nested
            list/tuple paths, or ``{path: value}``).
        values : Values, optional
            Values for positional style ``min(parameters, values)``.
        **updates
            Keyword update style, e.g. ``min(param=0.5)`` and nested via
            ``min(**{"b.param": 3.0})``.

        Returns
        -------
        pytree : PyTree
            The pytree with the leaves specified by parameters updated with the
            minimum value of the leaf and values.
        """
        parameters, values = _normalise_mutation_inputs(
            parameters,
            values,
            updates=updates,
            method_name="min",
            require_values=True,
        )
        new_parameters, new_values = _format(parameters, values)
        new_values = [
            np.minimum(leaf, value)
            for value, leaf in zip(new_values, _get_leaves(self, new_parameters))
        ]

        # Define 'where' function and update pytree
        def leaves_fn(pytree):
            return _get_leaves(pytree, new_parameters)

        return eqx.tree_at(
            leaves_fn, self, new_values, is_leaf=lambda leaf: leaf is None
        )

    def max(
        self: PyTree,
        parameters: Params = None,
        values: Values = None,
        /,
        **updates,
    ) -> PyTree:
        """
        Updates the leaves specified by parameters with the maximum value of the
        leaves and values.


        Parameters
        ----------
        parameters : Params, optional
            Parameter selector or mapping (string, list/tuple paths, nested
            list/tuple paths, or ``{path: value}``).
        values : Values, optional
            Values for positional style ``max(parameters, values)``.
        **updates
            Keyword update style, e.g. ``max(param=10.0)`` and nested via
            ``max(**{"b.param": 1.0})``.

        Returns
        -------
        pytree : PyTree
            The pytree with the leaves specified by parameters updated with the
            maximum value of the leaf and values.
        """
        parameters, values = _normalise_mutation_inputs(
            parameters,
            values,
            updates=updates,
            method_name="max",
            require_values=True,
        )
        new_parameters, new_values = _format(parameters, values)
        new_values = [
            np.maximum(leaf, value)
            for value, leaf in zip(new_values, _get_leaves(self, new_parameters))
        ]

        # Define 'where' function and update pytree
        def leaves_fn(pytree):
            return _get_leaves(pytree, new_parameters)

        return eqx.tree_at(
            leaves_fn, self, new_values, is_leaf=lambda leaf: leaf is None
        )


class Module(_Base):
    """A Zodiax module with local aliases and raised descendant attributes.

    ``Module`` provides immutable path operations, numerical-expression population
    and resolution, explicit aliases, and shorthand access to attributes held by its
    dataclass descendants. Aliases
    are local, static mappings from a concise name to a canonical structural path.
    The nearest matching descendant is otherwise used. Multiple matches at the same
    depth are rejected with their qualified paths so lookup never depends on child
    ordering.

    All public Zodiax model objects use this common module contract.
    """

    alias: Alias | None = eqx.field(
        default=None,
        static=True,
        kw_only=True,
        converter=_normalise_alias,
    )

    def populate(self) -> Any:
        """Return an ephemeral copy with linked numerical values populated.

        Population is intended to happen inside the function transformed by JAX.
        It establishes shared owners without evaluating contextual expressions.
        """
        from .numerics.links import populate

        return populate(self)

    def resolve(self, *, runtime: bool = False, **context: Any) -> Any:
        """Return a copy with eligible numerical expressions evaluated in context.

        Resolution is separate from :meth:`populate`: link population establishes
        shared owners, while resolution evaluates contextual definitions. Fields
        declared with :func:`field` using ``runtime=True`` remain unrealised unless
        this call also supplies ``runtime=True``.
        """
        from .numerics.expressions import resolve

        return resolve(self, runtime=runtime, **context)

    def __check_init__(self) -> None:
        """Validate aliases after every ordinary Equinox construction."""
        _validate_aliases(self)

    def validate_aliases(self) -> None:
        """Validate every alias in this module's current unresolved topology."""
        validate_aliases(self)

    def __getattr__(self, key: str) -> Any:
        """Return a local alias or uniquely named descendant attribute."""
        return _resolve_module_attribute(self, key)


def update(
    parameters: Mapping[str, Any],
    *objects: Module,
    strict: bool = True,
    mode: str = "first",
) -> tuple[Module, ...]:
    """Immutably update matching parameter paths across several modules.

    ``mode="first"`` assigns each path to its first matching object. ``mode="all"``
    applies it to every matching object. Independently, ``strict=True`` rejects paths
    that matched no object.
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

    unmatched = set(parameters)
    updated = []
    for obj in objects:
        candidates = (
            {path: value for path, value in parameters.items() if path in unmatched}
            if mode == "first"
            else parameters
        )
        values = {}
        for path, value in candidates.items():
            try:
                obj.get(path, to_array=False)
            except (AttributeError, KeyError):
                continue
            values[path] = value
            unmatched.discard(path)
        updated.append(obj.set(**values) if values else obj)

    if strict and unmatched:
        paths = ", ".join(repr(path) for path in parameters if path in unmatched)
        raise KeyError(f"Unused parameter paths: {paths}.")
    return tuple(updated)


def build_wrapper(pytree: PyTree, filter_fn: callable = eqx.is_array):
    """
    Deconstructs an equinox model into its values and structure, and returns a
    `WrapperHolder` object that can be used to interact with the model in a way
    that is compatible with the Zodiax framework.

    Parameters
    ----------
    pytree : PyTree
        The pytree to deconstruct.
    filter_fn : callable, optional
        A function that takes a leaf of the pytree and returns a boolean value

    Returns
    -------
    values : Array
        The values of the model, flattened and concatenated.
    structure : EquinoxWrapper
        The structure of the model, stored in a `EquinoxWrapper` object.
    """
    arr_mask = jtu.map(lambda leaf: filter_fn(leaf), pytree)
    dyn, static = eqx.partition(pytree, arr_mask)
    leaves, tree_def = jtu.flatten(dyn)
    values = np.concatenate([val.flatten() for val in leaves])
    return values, EquinoxWrapper(static, leaves, tree_def)


class EquinoxWrapper(Module):
    """
    A wrapper class designed to store an Equinox model (typically a neural network)
    in a way that makes it easily compatible within the Zodiax framework. This is
    necessary as Equinox operates on _whole_ models, where as Zodiax operates on
    model _leaves_. This class is designed to bridge that gap.

    This class should not need to be interacted with directly, and is designed to be
    held within the `WrapperHolder` class.
    """

    static: eqx.Module
    shapes: list
    sizes: list
    starts: list
    tree_def: None

    def __init__(self, static, leaves, tree_def, *, alias=None):
        self.static = static
        self.tree_def = tree_def
        self.shapes = [v.shape for v in leaves]
        self.sizes = [int(v.size) for v in leaves]
        self.starts = [int(i) for i in np.cumsum(np.array([0] + self.sizes))]
        self.alias = alias

    def inject(self, values):
        leaves = [
            lax.dynamic_slice(values, (start,), (size,)).reshape(shape)
            for start, size, shape in zip(self.starts, self.sizes, self.shapes)
        ]
        return eqx.combine(jtu.unflatten(self.tree_def, leaves), self.static)


class WrapperHolder(Module):
    """
    A class designed to hold an Equinox model, its structure and values. This helps it
    operate smoothly within the Zodiax framework.

    To apply transformations to the Equinox model values, operate on the `values` leaf
    of this class. To build the model, call the `build` property, and the Equinox model
    will be constructed with the stored values and be able to be used as if it
    were a regular Equinox model.

    This class is designed to be instantiated by the `build_wrapper` function.

    Example
    -------

    import equinox as eqx
    import zodiax as zdx
    import jax.numpy as np
    import jax.random as jr

    eqx_model = eqx.nn.MLP(
        in_size=16, out_size=16, width_size=32, depth=1, key=jr.PRNGKey(0)
    )

    class Foo(zdx.WrapperHolder):

        def __init__(self, nn):
            values, structure = zdx.build_wrapper(nn)
            self.values = values
            self.structure = structure

        def __call__(self, x):
            return self.build(x)

    x = np.ones(16)
    foo = Foo(eqx_model)

    Now we can use the model as if it were a regular Equinox model
    print(foo(x))

    `[ 0.1767296   0.15628047 -0.63250038 -0.01583058  0.39692974  0.4556041
    0.33121592 -0.3183221  -0.75008567 -0.32724514  0.28351735 -0.03595607
    -0.53921278 -0.20966474 -0.33641739 -0.28726151]`

    We can also apply Zodiax transformations to the model!
    print(foo.multiply("values", 0.)(x))

    `[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]`
    """

    values: Array
    structure: EquinoxWrapper

    @property
    def build(self):
        """
        Builds the Equinox model with the stored values and structure.
        """
        return self.structure.inject(self.values)

    def __getattr__(self, name):
        if hasattr(self.structure, name):
            return getattr(self.structure, name)
        raise AttributeError(f"Attribute {name} not found in {self.__class__.__name__}")
