from __future__ import annotations
import jax
import jax.numpy as np
import equinox as eqx
from equinox import tree_at, Module
from typing import Union, Any, List


__all__ = ["Base"]


Params = Union[str, List[str]]


def _get_leaf(pytree: Base, param: Params) -> Any:
    """
    A hidden class designed to recurse down a pytree following the param,
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
        The pytree object to recurse though.
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
        raise KeyError("key: {} not found in object: {}".format(key, type(pytree)))

    # Return param if at the end of param, else recurse
    return pytree if len(param) == 1 else _get_leaf(pytree, param[1:])


def _get_leaves(pytree: Base, parameters: list) -> list:
    """
    Returns a list of leaves specified by the parameters.

    Parameters
    ----------
    pytree : Base
        The pytree object to recurse though.
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
    # Initialise empty lists
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
                "The number of values must match the number of parameters."
            )

        # Iterate over parameters and values
        for param, value in zip(parameters, values):
            # Recurse and add in the case of list inputs
            if isinstance(param, list):
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
            if isinstance(param, list):
                new_parameters = _unwrap(param)
                parameters_out += new_parameters

            # Params must already be absolute
            else:
                parameters_out.append(param)
        return parameters_out


def _format(parameters: Params, values: list = None) -> list:
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
        if (
            len(parameters) > 1
            and values is not None
            and True in [isinstance(p, list) for p in parameters]
        ):
            assert isinstance(values, list) and len(values) == len(parameters), (
                "If a list of parameters is provided, the list of values must be "
                "of equal length."
            )

        # Its a list - iterate and unbind all the keys
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


###############
### Classes ###
###############
class Base(Module):
    """
    Extend the Equinox.Module class to give a user-friendly 'param based' API
    for working with pytrees by adding a series of methods used to interface
    with the leaves of the pytree using parameters.
    """

    def get(self: Base, parameters: Params) -> Any:
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

    def set(self: Base, parameters: Params, values: Union[List[Any], Any]) -> Base:
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
        def leaves_fn(pytree):
            return _get_leaves(pytree, new_parameters)

        return tree_at(leaves_fn, self, new_values, is_leaf=lambda leaf: leaf is None)

    def update(self: Base, dict: dict) -> Base:
        """
        Calls the set method to update the leaves specified by the keys
        of the dictionary with the values of the dictionary.

        Parameters
        ----------
        dict : dict
            The dictionary of parameters and values to update the leaves with.

        Returns
        -------
        pytree : Base
            The pytree with updated parameters.
        """

        # Grabbing the parameters and values from the dictionary
        parameters, values = list(dict.keys()), list(dict.values())

        # Calling the set method
        return self.set(parameters, values)

    def add(self: Base, parameters: Params, values: Union[List[Any], Any]) -> Base:
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
        new_values = [
            leaf + value
            for value, leaf in zip(new_values, _get_leaves(self, new_parameters))
        ]

        # Define 'where' function and update pytree
        def leaves_fn(pytree):
            return _get_leaves(pytree, new_parameters)

        return tree_at(leaves_fn, self, new_values, is_leaf=lambda leaf: leaf is None)

    def multiply(self: Base, parameters: Params, values: Union[List[Any], Any]) -> Base:
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
        new_values = [
            leaf * value
            for value, leaf in zip(new_values, _get_leaves(self, new_parameters))
        ]

        # Define 'where' function and update pytree
        def leaves_fn(pytree):
            return _get_leaves(pytree, new_parameters)

        return tree_at(leaves_fn, self, new_values, is_leaf=lambda leaf: leaf is None)

    def divide(self: Base, parameters: Params, values: Union[List[Any], Any]) -> Base:
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
        new_values = [
            leaf / value
            for value, leaf in zip(new_values, _get_leaves(self, new_parameters))
        ]

        # Define 'where' function and update pytree
        def leaves_fn(pytree):
            return _get_leaves(pytree, new_parameters)

        return tree_at(leaves_fn, self, new_values, is_leaf=lambda leaf: leaf is None)

    def power(self: Base, parameters: Params, values: Union[List[Any], Any]) -> Base:
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
        new_values = [
            leaf**value
            for value, leaf in zip(new_values, _get_leaves(self, new_parameters))
        ]

        # Define 'where' function and update pytree
        def leaves_fn(pytree):
            return _get_leaves(pytree, new_parameters)

        return tree_at(leaves_fn, self, new_values, is_leaf=lambda leaf: leaf is None)

    def min(self: Base, parameters: Params, values: Union[List[Any], Any]) -> Base:
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
        new_values = [
            np.minimum(leaf, value)
            for value, leaf in zip(new_values, _get_leaves(self, new_parameters))
        ]

        # Define 'where' function and update pytree
        def leaves_fn(pytree):
            return _get_leaves(pytree, new_parameters)

        return tree_at(leaves_fn, self, new_values, is_leaf=lambda leaf: leaf is None)

    def max(self: Base, parameters: Params, values: Union[List[Any], Any]) -> Base:
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
        new_values = [
            np.maximum(leaf, value)
            for value, leaf in zip(new_values, _get_leaves(self, new_parameters))
        ]

        # Define 'where' function and update pytree
        def leaves_fn(pytree):
            return _get_leaves(pytree, new_parameters)

        return tree_at(leaves_fn, self, new_values, is_leaf=lambda leaf: leaf is None)


class BaseModeller(Base):
    # TODO proper documentation

    """
    A base class for modelling that extends `zodiax.Base`. This class allows for
    dynamic attribute access and dictionary-like item retrieval from the `params`
    dictionary.
    Attributes:
        params (dict): A dictionary of parameters.
    Methods:
        __getattr__(key):
            Dynamically retrieves the value associated with `key` from the `params`
            dictionary. If `key` is not found directly in `params`, it searches
            through the values of `params` to find an attribute named `key`.
            Raises an `AttributeError` if `key` is not found.
        __getitem__(key):
            Retrieves a dictionary of values from `params` where the `key` is present
            in the nested dictionaries of `params`.
    """
    params: dict

    def __init__(self, params):
        self.params = params

    def __getattr__(self, key):
        """
        Dynamically retrieves the value associated with `key` from the `params`
        dictionary. If `key` is not found directly in `params`, it searches
        through the values of `params` to find an attribute named `key`.
        Raises an `AttributeError` if `key` is not found.
        """
        if key in self.params:
            return self.params[key]
        for k, val in self.params.items():
            if hasattr(val, key):
                return getattr(val, key)
        raise AttributeError(
            f"Attribute {key} not found in params of {self.__class__.__name__} object"
        )

    def __getitem__(self, key):
        """
        Retrieves a dictionary of values from `params` where the `key` is present
        in the nested dictionaries of `params`.
        """
        values = {}
        for param, item in self.params.items():
            if isinstance(item, dict) and key in item.keys():
                values[param] = item[key]

        return values


class ModelParams(BaseModeller):
    # TODO proper documentation
    """
    A class to manage model parameters with various utility
    methods for manipulation and access.

    Methods
    -------
    keys:
        Returns a list of parameter keys.
    values:
        Returns a list of parameter values.
    __getattr__(key):
        Retrieves the value of the specified parameter key.
    replace(values):
        Takes in a super-set class and updates this class with input values
    from_model(values):
        Sets the parameters from a given model's values.
    __add__(values):
        Adds the provided values to the current parameters.
    __iadd__(values):
        In-place addition of the provided values to the current parameters.
    __mul__(values):
        Multiplies the provided values with the current parameters.
    __imul__(values):
        In-place multiplication of the provided values with the current parameters.
    inject(other):
        Injects the values of this class into another class.
    """

    @property
    def keys(self):
        return list(self.params.keys())

    @property
    def values(self):
        return list(self.params.values())

    def __getattr__(self, key):
        if key in self.keys:
            return self.params[key]
        for k, val in self.params.items():
            if hasattr(val, key):
                return getattr(val, key)
        raise AttributeError(
            f"Attribute {key} not found in params of {self.__class__.__name__} object"
        )

    def replace(self, values):
        """
        Takes in a super-set class and updates this class with input values
        """
        return self.set(
            "params", dict([(param, getattr(values, param)) for param in self.keys])
        )

    def from_model(self, model: Base):
        return self.set(
            "params", dict([(param, model.get(param)) for param in self.keys])
        )

    def __add__(self, values):
        matched = self.replace(values)
        return jax.tree_map(lambda x, y: x + y, self, matched)

    def __iadd__(self, values):
        return self.__add__(values)

    def __mul__(self, values):
        matched = self.replace(values)
        return jax.tree_map(lambda x, y: x * y, self, matched)

    def __imul__(self, values):
        return self.__mul__(values)

    def inject(self, other: Base):
        # Injects the values of this class into another class
        return other.set(self.keys, self.values)


class ModelHistory(ModelParams):
    # TODO proper documentation

    """
    Tracks the history of a set of parameters in a model via tuples.
    Adds a series of convenience functions to interface with it.

    NOTE This could have issues with leaves not being jax.Arrays,
    so at some point it should be explicitly enforced that
    only array_likes are tracked.
    """

    def __init__(self, model, tracked):

        history = {}
        for param in tracked:
            leaf = model.get(param)
            if not eqx.is_array_like(leaf):
                history[param] = jax.tree_map(lambda sub_leaf: [sub_leaf], leaf)
            else:
                history[param] = [leaf]

        self.params = history

    def append(self, model):
        history = self.params
        for param, leaf_history in history.items():
            if hasattr(model, param):
                new_leaf = getattr(model, param)
            else:
                new_leaf = model.get(param)

            # Tree-like case
            if not eqx.is_array_like(new_leaf):

                def append_fn(history, value):
                    return history + [value]

                def leaf_fn(leaf):
                    return isinstance(leaf, list)

                new_leaf_history = jax.tree_map(
                    append_fn, leaf_history, new_leaf, is_leaf=leaf_fn
                )
                history[param] = new_leaf_history

            # Non-tree case
            else:
                history[param] = leaf_history + [new_leaf]
        return self.set("params", history)
