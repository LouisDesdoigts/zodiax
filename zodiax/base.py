from __future__ import annotations
import zodiax
import jax.numpy as np
from equinox import tree_at, Module
from typing import Union, Any, Callable
from jaxtyping import Array, PyTree


__all__ = ["Base"]


def _get_leaf(pytree : PyTree, path : Union[str, list]) -> Any:
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
    pytree : PyTree
        The pytee object to recurse though.
    path : Union[str, list]
        The path to recurse down.

    Returns
    -------
    leaf : Any
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


def _get_leaves(pytree : PyTree, paths : list) -> list:
    """
    Returns a list of leaves specified by the paths.

    Parameters
    ----------
    paths : list
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


def _unwrap(paths     : Union[str, list],
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


###############
### Classes ###
###############
class Base(Module):
    """
    Extend the Equninox.Module class to give a user-friendly 'path based' API
    for working with pytrees by adding a series of methods used to interface
    with the leaves of the pytree using paths.
    """
    def get(self  : PyTree,
            paths : str,
            pmap  : dict = None) -> Any:
        """
        Get the leaf specified by path.

        Parameters
        ----------
        paths : Union[str, list]
            A list/tuple of nested paths to unwrap.
        pmap : dict = None
            A dictionary of paths.

        Returns
        -------
        leaf, leaves : Any, list
            The leaf or list of leaves specified by paths.
        """
        new_paths = _format(paths, pmap=pmap)
        values = _get_leaves(self, new_paths)
        return values[0] if len(new_paths) == 1 else values


    def set(self   : PyTree,
            paths  : Union[str, list],
            values : Union[Any, list],
            pmap   : dict = None) -> PyTree:
        """
        Set the leaves specified by paths with values.

        Parameters
        ----------
        paths : Union[str, list]
            A path or list of paths or list of nested paths.
        values : Union[Any, list]
            The list of values to set at the leaves specified by paths.
        pmap : dict = None
            A dictionary of paths.

        Returns
        -------
        pytree : PyTree
            The pytree with leaves specified by paths updated with values.
        """
        new_paths, new_values = _format(paths, values, pmap)

        # Define 'where' function and update pytree
        leaves_fn = lambda pytree: _get_leaves(pytree, new_paths)
        return tree_at(leaves_fn, self, new_values,
                      is_leaf = lambda leaf: leaf is None)


    def add(self   : PyTree,
            paths  : Union[str, list],
            values : Union[Any, list],
            pmap   : dict = None) -> PyTree:
        """
        Add to the the leaves specified by paths with values.

        Parameters
        ----------
        paths : Union[str, list]
            A path or list of paths or list of nested paths.
        values : Union[Any, list]
            The list of values to add to the leaves specified by paths.
        pmap : dict = None
            A dictionary of paths.

        Returns
        -------
        pytree : PyTree
            The pytree with values added to leaves specified by paths.
        """
        new_paths, new_values = _format(paths, values, pmap)
        new_values = [leaf + value for value, leaf in zip(new_values, \
                                    _get_leaves(self, new_paths))]

        # Define 'where' function and update pytree
        leaves_fn = lambda pytree: _get_leaves(pytree, new_paths)
        return tree_at(leaves_fn, self, new_values,
                      is_leaf = lambda leaf: leaf is None)


    def multiply(self   : PyTree,
                 paths  : Union[str, list],
                 values : Union[list, Any],
                 pmap   : dict = None) -> PyTree:
        """
        Multiplies the the leaves specified by paths with values.

        Parameters
        ----------
        paths : Union[str, list]
            A path or list of paths or list of nested paths.
        values : Union[Any, list]
            The list of values to multiply the leaves specified by paths.
        pmap : dict = None
            A dictionary of paths.

        Returns
        -------
        pytree : PyTree
            The pytree with values multiplied by leaves specified by paths.
        """
        new_paths, new_values = _format(paths, values, pmap)
        new_values = [leaf * value for value, leaf in zip(new_values, \
                                    _get_leaves(self, new_paths))]

        # Define 'where' function and update pytree
        leaves_fn = lambda pytree: _get_leaves(pytree, new_paths)
        return tree_at(leaves_fn, self, new_values,
                      is_leaf = lambda leaf: leaf is None)


    def divide(self   : PyTree,
               paths  : Union[str, list],
               values : Union[list, Any],
               pmap   : dict = None) -> PyTree:
        """
        Divides the the leaves specified by paths with values.

        Parameters
        ----------
        paths : Union[str, list]
            A path or list of paths or list of nested paths.
        values : list
            The list of values to divide the leaves specified by paths.
        pmap : dict = None
            A dictionary of paths.

        Returns
        -------
        pytree : PyTree
            The pytree with values divided by leaves specified by paths.
        """
        new_paths, new_values = _format(paths, values, pmap)
        new_values = [leaf / value for value, leaf in zip(new_values, \
                                    _get_leaves(self, new_paths))]

        # Define 'where' function and update pytree
        leaves_fn = lambda pytree: _get_leaves(pytree, new_paths)
        return tree_at(leaves_fn, self, new_values,
                      is_leaf = lambda leaf: leaf is None)


    def power(self   : PyTree,
              paths  : Union[str, list],
              values : Union[list, Any],
              pmap   : dict = None) -> PyTree:
        """
        Raises th leaves specified by paths to the power of values.

        Parameters
        ----------
        paths : PAth
            A path or list of paths or list of nested paths.
        values : list
            The list of values to take the leaves specified by paths to the
            power of.
        pmap : dict = None
            A dictionary of paths.

        Returns
        -------
        pytree : PyTree
            The pytree with the leaves specified by paths raised to the power
            of values.
        """
        new_paths, new_values = _format(paths, values, pmap)
        new_values = [leaf ** value for value, leaf in zip(new_values, \
                                    _get_leaves(self, new_paths))]

        # Define 'where' function and update pytree
        leaves_fn = lambda pytree: _get_leaves(pytree, new_paths)
        return tree_at(leaves_fn, self, new_values,
                      is_leaf = lambda leaf: leaf is None)


    def min(self   : PyTree,
            paths  : Union[str, list],
            values : Union[list, Any],
            pmap   : dict = None) -> PyTree:
        """
        Updates the leaves specified by paths with the minimum value of the
        leaves and values.

        Parameters
        ----------
        paths : Union[str, list]
            A path or list of paths or list of nested paths.
        values : list
            The list of values to take the minimum of and the leaf.
        pmap : dict = None
            A dictionary of paths.

        Returns
        -------
        pytree : PyTree
            The pytree with the leaves specified by paths updated with the
            minimum value of the leaf and values.
        """
        new_paths, new_values = _format(paths, values, pmap)
        new_values = [np.minimum(leaf, value) for value, leaf in \
                    zip(new_values, _get_leaves(self, new_paths))]

        # Define 'where' function and update pytree
        leaves_fn = lambda pytree: _get_leaves(pytree, new_paths)
        return tree_at(leaves_fn, self, new_values,
                      is_leaf = lambda leaf: leaf is None)


    def max(self   : PyTree,
            paths  : Union[str, list],
            values : Union[list, Any],
            pmap   : dict = None) -> PyTree:
        """
        Updates the leaves specified by paths with the maximum value of the
        leaves and values.

        Parameters
        ----------
        paths : Union[str, list]
            A path or list of paths or list of nested paths.
        values : list
            The list of values to take the maximum of and the leaf.
        pmap : dict = None
            A dictionary of paths.

        Returns
        -------
        pytree : PyTree
            The pytree with the leaves specified by paths updated with the
            maximum value of the leaf and values.
        """
        new_paths, new_values = _format(paths, values, pmap)
        new_values = [np.maximum(leaf, value) for value, leaf in \
                    zip(new_values, _get_leaves(self, new_paths))]

        # Define 'where' function and update pytree
        leaves_fn = lambda pytree: _get_leaves(pytree, new_paths)
        return tree_at(leaves_fn, self, new_values,
                      is_leaf = lambda leaf: leaf is None)


    def apply(self  : PyTree,
              paths : Union[str, list],
              fns   : Union[list, Callable],
              pmap  : dict = None) -> PyTree:
        """
        Applies the functions within fns the leaves specified by paths.

        Parameters
        ----------
        paths : Union[str, list]
            A path or list of paths or list of nested paths.
        fns : Union[list, Callable]
            The list of functions to apply to the leaves.
        pmap : dict = None
            A dictionary of paths.

        Returns
        -------
        pytree : PyTree
            The pytree with fns applied to the leaves specified by paths.
        """
        new_paths, new_fns = _format(paths, fns, pmap)
        new_values = [fn(leaf) for fn, leaf in zip(new_fns, \
                                    _get_leaves(self, new_paths))]

        # Define 'where' function and update pytree
        leaves_fn = lambda pytree: _get_leaves(pytree, new_paths)
        return tree_at(leaves_fn, self, new_values,
                      is_leaf = lambda leaf: leaf is None)


    def apply_args(self  : PyTree,
                   paths : Union[str, list],
                   fns   : Union[list, Any],
                   args  : Union[list, tuple],
                   pmap  : dict = None) -> PyTree:
        """
        Applies the functions within fns the leaves specified by paths, while
        also passing in args to the function.

        Parameters
        ----------
        paths : Union[str, list]
            A path or list of paths or list of nested paths.
        fns : Union[list, Callable]
            The list of functions to apply to the leaves.
        args : Union[list, tuple]
            The tupe or list of tuples of extra arguments to pass into fns.
        pmap : dict = None
            A dictionary of paths.

        Returns
        -------
        pytree : PyTree
            The pytree with fns applied to the leaves specified by paths with
            the extra args passed in.
        """
        new_paths, new_fns = _format(paths, fns, pmap)
        new_paths, new_args = _format(paths, args, pmap)
        new_values = [fn(leaf, *args) for fn, args, leaf in zip(new_fns, \
                            new_args, _get_leaves(self, new_paths))]

        # Define 'where' function and update pytree
        leaves_fn = lambda pytree: _get_leaves(pytree, new_paths)
        return tree_at(leaves_fn, self, new_values,
                      is_leaf = lambda leaf: leaf is None)


    def set_and_call(self      : PyTree,
                     paths     : Union[str, list],
                     values    : Union[Any, list],
                     call_fn   : str,
                     pmap      : dict = None,
                     **kwargs) -> Any:
        """
        Updates the leaves speficied by paths with values, and then calls the
        function specified by the string call_fn, returning whatever is
        returnd by the call_fn. Any extra positional arguments or key-word
        arguments are passed through to the modelling function.

        This function is desigend to be used in conjunction with numpyro.
        Please go through the 'PyTree interface' tutorial to see how this
        is used.

        Parameters
        ----------
        paths : Union[str, list]
            A path or list of paths or list of nested paths.
        values : Union[Any, list]
            The list of values to set at the leaves specified by paths.
        call_fn : str
            A string specifying which model function to call.
        pmap : dict = None
            A dictionary of paths.

        Returns
        -------
            : Any
            Whatever object is returned by call_fn.
        """
        return getattr(self.set(paths, values, pmap), call_fn)(**kwargs)


    def apply_and_call(self      : PyTree,
                       paths     : Union[str, list],
                       fns       : Union[Callable, list],
                       call_fn  : str,
                       pmap      : dict = None,
                       **kwargs) -> object:
        """
        Applies the functions specified by fns to the leaves speficied by
        paths, and then calls the function specified by the string call_fn,
        returning whatever is returnd by the call_fn. Any extra positional
        arguments or keyword arguments are passed through to the modelling
        function.

        Parameters
        ----------
        call_fn : str
            A string specifying which model function to call.
        paths : Union[str, list]
            A path or list of paths or list of nested paths.
        fns : Union[Callable, list]
            The list of functions to apply to the leaves.
        pmap : dict = None
            A dictionary of paths.

        Returns
        -------
            : Any
            Whatever object is returned by call_fn.
        """
        return getattr(self.apply(paths, fns, pmap), call_fn)(**kwargs)


"""
Extra methods for possible future classes that recursively searches for 
parameters.
"""
#     Method 1, get first
#     def __getattr__(self, key):
#         """

#         """
#         # Found it, nice work
#         dict_like = self.__dict__
#         if key in dict_like.keys():
#             return dict_like[key]

#         # Expand and iterate though items
#         for value in dict_like.values():

#             # Dictionary, call the recursive method
#             if isinstance(value, dict):
#                 try:
#                     return _recurse_dict(value, key)
#                 except ValueError as e:
#                     pass

#             # dLux object, recurse
#             if isinstance(value, Base):
#                 try:
#                     return getattr(value, key)
#                 except ValueError as e:
#                     pass

#         # Not found, raise error
#         raise ValueError("'{}' object has no attribute '{}'"\
#                              .format(type(self), key))


#     def _recurse_dict(self, dict_like, key):
#         """

#         """
#         # Return item if it exists
#         if key in dict_like:
#             return dict_like[key]

#         # Iterate through values
#         for value in dict_like.values():

#             # Value is a dict, Recurse
#             if isinstance(value, dict):
#                 try:
#                     return self._recurse_dict(value, key)
#                 except ValueError as e:
#                     pass

#             # Value is a dLux object, recall the getattr method
#             if isinstance(value, Base):
#                 try:
#                     return getattr(value, key)
#                 except ValueError as e:
#                     pass

#         # Nothing found, raise Error
#         raise ValueError("'{}' not found.".format(key))


#     # Method 2, get all
#     def __getattr__(self, key):
#         """

#         """
#         return self._get_all(key, [])


#     def _get_all(self, key, values):

#         # Found it, nice work
#         dict_like = self.__dict__
#         if key in dict_like.keys():
#             values.append(dict_like[key])

#         # Expand and iterate though items
#         for value in dict_like.values():

#             # Dictionary, call the recursive method
#             if isinstance(value, dict):
#                 # values.append(self._recurse_dict(value, key, values))
#                 values = self._recurse_dict(value, key, values)

#             # dLux object, recurse
#             if isinstance(value, Base):
#                 # values.append(value._get_all(key, values))
#                 values = value._get_all(key, values)

#         return values


#     def _recurse_dict(self, dict_like, key, values):
#         """

#         """
#         # Return item if it exists
#         if key in dict_like:
#             values.append(dict_like[key])

#         # Iterate through values
#         for value in dict_like.values():

#             # Value is a dict, Recurse
#             if isinstance(value, dict):
#                 # values.append(self._recurse_dict(value, key, values))
#                 values = self._recurse_dict(value, key, values)

#             # Value is a dLux object, recall the getattr method
#             if isinstance(value, Base):
#                 # values.append(value._get_all(key, values))
#                 values = value._get_all(key, values)

#         return values