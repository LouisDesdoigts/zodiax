from __future__ import annotations
import zodiax
import jax.numpy as np
from equinox import tree_at, Module
from typing import Union, Any, Callable, List
from jaxtyping import Array


__all__ = ["Base"]


Params = Union[str, List[str]]


def _get_leaf(pytree : Base, param : Params) -> Any:
    """
    A hidden class desinged to recurse down a pytree following the param,
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
    pytree : Base
        The pytee object to recurse though.
    param : Params
        The param to recurse down.

    Returns
    -------
    leaf : Any
        The leaf object specified at the end of the param object.
    """
    key = param[0]
    if hasattr(pytree, key):
        pytree = getattr(pytree, key)
    elif isinstance(pytree, dict):
        pytree = pytree[key]
    elif isinstance(pytree, (list, tuple)):
        pytree = pytree[int(key)]
    else:
        raise ValueError(
            "key: {} not found in object: {}".format(key,type(pytree)))

    # Return param if at the end of param, else recurse
    return pytree if len(param) == 1 else _get_leaf(pytree, param[1:])


def _get_leaves(pytree : Base, parameters : list) -> list:
    """
    Returns a list of leaves specified by the parameters.

    Parameters
    ----------
    pytree : Base
        The pytee object to recurse though.
    parameters : list
        A list/tuple of nested parameters. Note param objects can only be
        nested a single time.

    Returns
    -------
    leaves : list
        The list of leaf objects specified by the parameters object
    """
    return [_get_leaf(pytree, param) for param in parameters]


def _unwrap(parameters : Params, values_in : list = None) -> list:
    """
    Unwraps the provided parameters in to the correct list-based format for the
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
    # Inititalise empty lists
    parameters_out, values_out = [], []

    # If values are provided, apply transformation to both
    if values_in is not None:
        # Make sure values is list
        values = values_in if isinstance(values_in, list) else [values_in]

        # Repeat values to match length of parameters
        if len(values) == 1:
            values = values * len(parameters)
        
        # Ensure correct length
        if len(values) != len(parameters):
            raise ValueError(
                "The number of values must match the number of parameters.")

        # Iterate over parameters and values
        for param, value in zip(parameters, values):

            # Recurse and add in the case of list inputs
            if isinstance(param, list):
                new_parameters, new_values = _unwrap(param, value)
                parameters_out  += new_parameters
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
            if isinstance(param, list):
                new_parameters = _unwrap(param)
                parameters_out += new_parameters

            # Params must already be absolute
            else:
                parameters_out.append(param)
        return parameters_out


def _format(parameters : Params, values : list = None) -> list:
    """
    Formats the provided parameters in to the correct list-based format for the
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
    if isinstance(parameters, list):

        # If there is nesting, ensure correct dis
        if len(parameters) > 1 and values is not None \
            and True in [isinstance(p, list) for p in parameters]:
            assert isinstance(values, list) and len(values) == len(parameters), \
            ("If a list of parameters is provided, the list of values must be "
                "of equal length.")

        # Its a list - iterate and unbind all the keys
        if values is not None:
            flat_parameters, new_values = _unwrap(parameters, values)
        else:
            flat_parameters = _unwrap(parameters)
        
        # Turn into seperate strings
        new_parameters = [param.split('.') if '.' in param else [param] \
                        for param in flat_parameters]

    # Un-nested/singular input
    else:
        # Turn into seperate strings
        new_parameters = [parameters.split('.') if '.' in parameters else [parameters]]
        new_values = [values]

    # Return
    return new_parameters if values is None else (new_parameters, new_values)


###############
### Classes ###
###############
class Base(Module):
    """
    Extend the Equninox.Module class to give a user-friendly 'param based' API
    for working with pytrees by adding a series of methods used to interface
    with the leaves of the pytree using parameters.
    """
    def get(self : Base, parameters : Params) -> Any:
        """
        Get the leaf specified by param.

        Parameters
        ----------
        parameters : Params
            A list/tuple of nested parameters to unwrap.

        Returns
        -------
        leaf, leaves : Any, list
            The leaf or list of leaves specified by parameters.
        """
        new_parameters = _format(parameters)
        values = _get_leaves(self, new_parameters)
        return values[0] if len(new_parameters) == 1 else values


    def set(self       : Base,
            parameters : Params,
            values     : Union[List[Any], Any]) -> Base:
        """
        Set the leaves specified by parameters with values.

        Parameters
        ----------
        parameters : Params
            A param or list of parameters or list of nested parameters.
        values : Union[List[Any], Any]
            The list of values to set at the leaves specified by parameters.

        Returns
        -------
        pytree : Base
            The pytree with leaves specified by parameters updated with values.
        """
        # Allow None inputs
        if values is None:
            values = [None]
            if isinstance(parameters, str):
                parameters = [parameters]
        new_parameters, new_values = _format(parameters, values)

        # Define 'where' function and update pytree
        leaves_fn = lambda pytree: _get_leaves(pytree, new_parameters)
        return tree_at(leaves_fn, self, new_values,
                      is_leaf = lambda leaf: leaf is None)


    def add(self       : Base,
            parameters : Params,
            values     : Union[List[Any], Any]) -> Base:
        """
        Add to the the leaves specified by parameters with values.

        Parameters
        ----------
        parameters : Params
            A param or list of parameters or list of nested parameters.
        values : Union[List[Any], Any]
            The list of values to add to the leaves specified by parameters.

        Returns
        -------
        pytree : Base
            The pytree with values added to leaves specified by parameters.
        """
        new_parameters, new_values = _format(parameters, values)
        new_values = [leaf + value for value, leaf in zip(new_values, \
                                    _get_leaves(self, new_parameters))]

        # Define 'where' function and update pytree
        leaves_fn = lambda pytree: _get_leaves(pytree, new_parameters)
        return tree_at(leaves_fn, self, new_values,
                      is_leaf = lambda leaf: leaf is None)


    def multiply(self       : Base,
                 parameters : Params,
                 values     : Union[List[Any], Any]) -> Base:
        """
        Multiplies the the leaves specified by parameters with values.

        Parameters
        ----------
        parameters : Params
            A param or list of parameters or list of nested parameters.
        values : Union[List[Any], Any]
            The list of values to multiply the leaves specified by parameters.

        Returns
        -------
        pytree : Base
            The pytree with values multiplied by leaves specified by parameters.
        """
        new_parameters, new_values = _format(parameters, values)
        new_values = [leaf * value for value, leaf in zip(new_values, \
                                    _get_leaves(self, new_parameters))]

        # Define 'where' function and update pytree
        leaves_fn = lambda pytree: _get_leaves(pytree, new_parameters)
        return tree_at(leaves_fn, self, new_values,
                      is_leaf = lambda leaf: leaf is None)


    def divide(self       : Base,
               parameters : Params,
               values     : Union[List[Any], Any]) -> Base:
        """
        Divides the the leaves specified by parameters with values.

        Parameters
        ----------
        parameters : Params
            A param or list of parameters or list of nested parameters.
        values : Union[List[Any], Any]
            The list of values to divide the leaves specified by parameters.

        Returns
        -------
        pytree : Base
            The pytree with values divided by leaves specified by parameters.
        """
        new_parameters, new_values = _format(parameters, values)
        new_values = [leaf / value for value, leaf in zip(new_values, \
                                    _get_leaves(self, new_parameters))]

        # Define 'where' function and update pytree
        leaves_fn = lambda pytree: _get_leaves(pytree, new_parameters)
        return tree_at(leaves_fn, self, new_values,
                      is_leaf = lambda leaf: leaf is None)


    def power(self       : Base,
              parameters : Params,
              values     : Union[List[Any], Any]) -> Base:
        """
        Raises th leaves specified by parameters to the power of values.

        Parameters
        ----------
        parameters : Params
            A param or list of parameters or list of nested parameters.
        values : Union[List[Any], Any]
            The list of values to take the leaves specified by parameters to the
            power of.

        Returns
        -------
        pytree : Base
            The pytree with the leaves specified by parameters raised to the power
            of values.
        """
        new_parameters, new_values = _format(parameters, values)
        new_values = [leaf ** value for value, leaf in zip(new_values, \
                                    _get_leaves(self, new_parameters))]

        # Define 'where' function and update pytree
        leaves_fn = lambda pytree: _get_leaves(pytree, new_parameters)
        return tree_at(leaves_fn, self, new_values,
                      is_leaf = lambda leaf: leaf is None)


    def min(self       : Base,
            parameters : Params,
            values     : Union[List[Any], Any]) -> Base:
        """
        Updates the leaves specified by parameters with the minimum value of the
        leaves and values.

        Parameters
        ----------
        parameters : Params
            A param or list of parameters or list of nested parameters.
        values : Union[List[Any], Any]
            The list of values to take the minimum of and the leaf.

        Returns
        -------
        pytree : Base
            The pytree with the leaves specified by parameters updated with the
            minimum value of the leaf and values.
        """
        new_parameters, new_values = _format(parameters, values)
        new_values = [np.minimum(leaf, value) for value, leaf in \
                    zip(new_values, _get_leaves(self, new_parameters))]

        # Define 'where' function and update pytree
        leaves_fn = lambda pytree: _get_leaves(pytree, new_parameters)
        return tree_at(leaves_fn, self, new_values,
                      is_leaf = lambda leaf: leaf is None)


    def max(self       : Base,
            parameters : Params,
            values     : Union[List[Any], Any]) -> Base:
        """
        Updates the leaves specified by parameters with the maximum value of the
        leaves and values.


        Parameters
        ----------
        parameters : Params
            A param or list of parameters or list of nested parameters.
        values : Union[List[Any], Any]
            The list of values to take the maximum of and the leaf.

        Returns
        -------
        pytree : Base
            The pytree with the leaves specified by parameters updated with the
            maximum value of the leaf and values.
        """
        new_parameters, new_values = _format(parameters, values)
        new_values = [np.maximum(leaf, value) for value, leaf in \
                    zip(new_values, _get_leaves(self, new_parameters))]

        # Define 'where' function and update pytree
        leaves_fn = lambda pytree: _get_leaves(pytree, new_parameters)
        return tree_at(leaves_fn, self, new_values,
                      is_leaf = lambda leaf: leaf is None)


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