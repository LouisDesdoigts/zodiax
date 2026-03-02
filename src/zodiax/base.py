from __future__ import annotations
import jax
import jax.numpy as np
import equinox as eqx
from jax import lax, Array
from typing import Union, Any

__all__ = ["Base", "build_wrapper", "EquinoxWrapper", "WrapperHolder"]

PyTree = Union[dict, list, tuple, eqx.Module]
Params = Union[str, list[str], tuple[str]]
Values = Union[Any, list[Any], tuple[Any]]


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
    pytree : PyTree
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


def _get_leaves(pytree: PyTree, parameters: list) -> list:
    """
    Returns a list of leaves specified by the parameters.

    Parameters
    ----------
    pytree : PyTree
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


class Base(eqx.Module):
    """
    Extend the Equinox.Module class to give a user-friendly 'param based' API
    for working with pytrees by adding a series of methods used to interface
    with the leaves of the pytree using parameters.
    """

    def get(self: PyTree, parameters: Params) -> Any:
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

    def set(self: PyTree, parameters: Params, values: Values) -> PyTree:
        """
        Set the leaves specified by parameters with values.

        Parameters
        ----------
        parameters : Params
            A param or list of parameters or list of nested parameters.
        values : Values
            The list of values to set at the leaves specified by parameters.

        Returns
        -------
        pytree : PyTree
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

        return eqx.tree_at(
            leaves_fn, self, new_values, is_leaf=lambda leaf: leaf is None
        )

    def update(self: PyTree, dict: dict) -> PyTree:
        """
        Calls the set method to update the leaves specified by the keys
        of the dictionary with the values of the dictionary.

        Parameters
        ----------
        dict : dict
            The dictionary of parameters and values to update the leaves with.

        Returns
        -------
        pytree : PyTree
            The pytree with updated parameters.
        """

        dict = _unpack(dict)
        parameters, values = list(dict.keys()), list(dict.values())

        # Calling the set method
        return self.set(parameters, values)

    def add(self: PyTree, parameters: Params, values: Values) -> PyTree:
        """
        Add to the the leaves specified by parameters with values.

        Parameters
        ----------
        parameters : Params
            A param or list of parameters or list of nested parameters.
        values : Values
            The list of values to add to the leaves specified by parameters.

        Returns
        -------
        pytree : PyTree
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

        return eqx.tree_at(
            leaves_fn, self, new_values, is_leaf=lambda leaf: leaf is None
        )

    def multiply(self: PyTree, parameters: Params, values: Values) -> PyTree:
        """
        Multiplies the the leaves specified by parameters with values.

        Parameters
        ----------
        parameters : Params
            A param or list of parameters or list of nested parameters.
        values : Values
            The list of values to multiply the leaves specified by parameters.

        Returns
        -------
        pytree : PyTree
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

        return eqx.tree_at(
            leaves_fn, self, new_values, is_leaf=lambda leaf: leaf is None
        )

    def divide(self: PyTree, parameters: Params, values: Values) -> PyTree:
        """
        Divides the the leaves specified by parameters with values.

        Parameters
        ----------
        parameters : Params
            A param or list of parameters or list of nested parameters.
        values : Values
            The list of values to divide the leaves specified by parameters.

        Returns
        -------
        pytree : PyTree
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

        return eqx.tree_at(
            leaves_fn, self, new_values, is_leaf=lambda leaf: leaf is None
        )

    def power(self: PyTree, parameters: Params, values: Values) -> PyTree:
        """
        Raises th leaves specified by parameters to the power of values.

        Parameters
        ----------
        parameters : Params
            A param or list of parameters or list of nested parameters.
        values : Values
            The list of values to take the leaves specified by parameters to the
            power of.

        Returns
        -------
        pytree : PyTree
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

        return eqx.tree_at(
            leaves_fn, self, new_values, is_leaf=lambda leaf: leaf is None
        )

    def min(self: PyTree, parameters: Params, values: Values) -> PyTree:
        """
        Updates the leaves specified by parameters with the minimum value of the
        leaves and values.

        Parameters
        ----------
        parameters : Params
            A param or list of parameters or list of nested parameters.
        values : Values
            The list of values to take the minimum of and the leaf.

        Returns
        -------
        pytree : PyTree
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

        return eqx.tree_at(
            leaves_fn, self, new_values, is_leaf=lambda leaf: leaf is None
        )

    def max(self: PyTree, parameters: Params, values: Values) -> PyTree:
        """
        Updates the leaves specified by parameters with the maximum value of the
        leaves and values.


        Parameters
        ----------
        parameters : Params
            A param or list of parameters or list of nested parameters.
        values : Values
            The list of values to take the maximum of and the leaf.

        Returns
        -------
        pytree : PyTree
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

        return eqx.tree_at(
            leaves_fn, self, new_values, is_leaf=lambda leaf: leaf is None
        )


def build_wrapper(eqx_model, filter_fn=eqx.is_array):
    """
    Deconstructs an equinox model into its values and structure, and returns a
    `WrapperHolder` object that can be used to interact with the model in a way
    that is compatible with the Zodiax framework.

    Parameters
    ----------
    eqx_model : equinox.Module
        The Equinox model to deconstruct.
    filter_fn : callable, optional
        A function that takes a leaf of the model and returns a boolean value

    Returns
    -------
    values : Array
        The values of the model, flattened and concatenated.
    structure : EquinoxWrapper
        The structure of the model, stored in a `EquinoxWrapper` object.
    """
    arr_mask = jax.tree.map(lambda leaf: filter_fn(leaf), eqx_model)
    dyn, static = eqx.partition(eqx_model, arr_mask)
    leaves, tree_def = jax.tree.flatten(dyn)
    values = np.concatenate([val.flatten() for val in leaves])
    return values, EquinoxWrapper(static, leaves, tree_def)


class EquinoxWrapper(Base):
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

    def __init__(self, static, leaves, tree_def):
        self.static = static
        self.tree_def = tree_def
        self.shapes = [v.shape for v in leaves]
        self.sizes = [int(v.size) for v in leaves]
        self.starts = [int(i) for i in np.cumsum(np.array([0] + self.sizes))]

    def inject(self, values):
        leaves = [
            lax.dynamic_slice(values, (start,), (size,)).reshape(shape)
            for start, size, shape in zip(self.starts, self.sizes, self.shapes)
        ]
        return eqx.combine(jax.tree.unflatten(self.tree_def, leaves), self.static)


class WrapperHolder(Base):
    """
    A class designed to hold an Equinox model, its structure and values. This helps it
    operate smoothly within the Zodiax framework.

    To apply transformations to the Equinox model values, operate on the `values` leaf
    of this class. To build the model, call the `build` property, and the Equinox model
    will be constructed with the stored values and be able to operated with as if it
    were a regular Equinox model.

    This class is designed to be instantiated by the `build_wrapper` function.

    Example
    -------

    ```python
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

    # Now we can use the model as if it were a regular Equinox model
    print(foo(x))

    >>> [ 0.1767296   0.15628047 -0.63250038 -0.01583058  0.39692974  0.4556041
    >>>   0.33121592 -0.3183221  -0.75008567 -0.32724514  0.28351735 -0.03595607
    >>>  -0.53921278 -0.20966474 -0.33641739 -0.28726151]

    # We can also apply Zodiax transformations to the model!
    print(foo.multiply("values", 0.)(x))
    >>> [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
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
