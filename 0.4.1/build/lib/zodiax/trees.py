from __future__ import annotations
import jax.numpy as np
from jax.tree_util import tree_map
from equinox import tree_at, Module, is_array, filter as eqx_filter
from optax import adam, multi_transform
from typing import Union, NewType, Any, Callable


__all__ = ["get_args"] #, "apply_updates"]


def _get_leaf(
              pytree : Pytree,
              path   : Union[str, list]) -> Leaf:
    """
    A hidden class desinged to recurse down a pytree following the path,
    returning the leaf at the end of the path.

    Base case: len(path) == 1
        In this case the leaf referred to by the single path entry is
        returned (and hence recursively sent up to the initial call).

    Recursive case: len(path) > 1
        In this case the function takes the PyTree like object referred to
        by the first entry in path, and recursively calls this function
        with this new pytree object and the path without the first entry.

    Parameters
    ----------
    pytree : Pytree
        The pytee object to recurse though.
    path : Union[str, list]
        The path to recurse down.

    Returns
    -------
    leaf : Leaf
        The leaf object specified at the end of the path object.
    """
    key = path[0]
    if hasattr(pytree, key):
        pytree = getattr(pytree, key)
    elif isinstance(pytree, dict):
        pytree = pytree[key]
    elif isinstance(pytree, (list, tuple)):
        pytree = pytree[int(key)]
    else:
        raise ValueError("key: {} not found in object: {}".format(key,
                                                        type(pytree)))

    # Return param if at the end of path, else recurse
    return pytree if len(path) == 1 else _get_leaf(pytree, path[1:])


def _get_leaves(pytree : Pytree, paths : list_like) -> list:
    """
    Returns a list of leaves specified by the paths.

    Parameters
    ----------
    paths : Union[str, list]
        A list/tuple of nested paths. Note path objects can only be
        nested a single time.
    pmap : dict = None
        A dictionary of absolute paths.

    Returns
    -------
    leaves : list
        The list of leaf objects specified by the paths object
    """
    return [_get_leaf(pytree, path) for path in paths]


def _unwrap( #pytree    : Pytree,
            paths     : Union[str, list],
            values_in : list = None,
            pmap      : dict = None) -> list:
    """
    Unwraps the provided paths in to the correct list-based format for the
    _get_leaves and _get_leaf methods, returning a single dimensional list
    of input paths.

    Parameters
    ----------
    paths : Union[str, list]
        A list/tuple of nested paths to unwrap.
    values_in : list = None
        The list of values to be unwrapped.
    pmap : dict = None
        A dictionary of paths.

    Returns
    -------
    paths, values : list, list
        The list of unwrapped paths or paths and values.
    """
    # Get keys
    keys = pmap.keys() if pmap is not None else []

    # Inititalise empty lists
    paths_out, values_out = [], []

    # Make sure values is list
    values = values_in if isinstance(values_in, list) else [values_in]

    # Repeat values to match length of paths
    values = values * len(paths) if len(values) == 1 else values
    assert len(values) == len(paths), ("Something odd has happened, this "
    "is likely due to a missmatch between the input paths and values.")

    # Iterate over paths and values
    for path, value in zip(paths, values):

        # Recurse and add in the case of list inputs
        if isinstance(path, list):
            new_paths, new_values = _unwrap(path, [value], pmap)
            paths_out  += new_paths
            values_out += new_values

        # Get the absolute path and append
        elif path in keys:
            paths_out.append(pmap[path])
            values_out.append(value)

        # Union[str, list] must already be absolute
        else:
            paths_out.append(path)
            values_out.append(value)

    # Return
    return paths_out if values_in is None else (paths_out, values_out)


def _format(paths  : Union[str, list],
            values : list = None,
            pmap   : dict = None) -> list:
    """
    Formats the provided paths in to the correct list-based format for the
    _get_leaves and _get_leaf methods, returning a single dimensional list
    of input paths, with the 'path map' (pmap) values applied.

    Parameters
    ----------
    paths : Union[str, list]
        A list/tuple of nested paths to unwrap.
    values : list = None
        The list of values to be unwrapped.
    pmap : dict = None
        A dictionary of paths.

    Returns
    -------
    paths, values : list, list
        The list of unwrapped paths or paths and values.
    """
    # Nested/multiple inputs
    if isinstance(paths, list):

        # If there is nesting, ensure correct dis
        if len(paths) > 1 and values is not None \
            and True in [isinstance(p, list) for p in paths]:
            assert isinstance(values, list) and len(values) == len(paths), \
            ("If a list of paths is provided, the list of values must be "
                "of equal length.")

        # Its a list - iterate and unbind all the keys
        if values is not None:
            flat_paths, new_values = _unwrap(paths, values, pmap)
        else:
            flat_paths = _unwrap(paths, pmap=pmap)
        
        # Turn into seperate strings
        new_paths = [path.split('.') if '.' in path else [path] \
                        for path in flat_paths]

    # Un-nested/singular input
    else:
        # Get from dict if it extsts
        keys = pmap.keys() if pmap is not None else []
        paths = pmap[paths] if paths in keys else paths

        # Turn into seperate strings
        new_paths = [paths.split('.') if '.' in paths else [paths]]
        new_values = [values]

    # Return
    return new_paths if values is None else (new_paths, new_values)
















def get_args(pytree: Pytree,
             paths : Union[str, list],
             pmap  : dict = None) -> Pytree:
    """
    Returns a pytree of matching structure with boolean values at the leaves.
    Leaves specified by paths will be True, all others will be False. 

    Parameters
    ----------
    paths : Union[str, list]
        A path or list of paths or list of nested paths.
    pmap : dict = None
        A dictionary of paths.

    Returns
    -------
    args : Pytree
        An pytree of matching structre with boolean values at the leaves.
    """
    args = tree_map(lambda _: False, pytree)
    paths = paths if isinstance(paths, list) else [paths]
    values = len(paths) * [True]
    return args.set(paths, values, pmap)



