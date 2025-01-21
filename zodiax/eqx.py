import zodiax
import equinox
from functools import wraps
from jaxtyping import PyTree
from typing import Union, Callable, List

import equinox as eqx
from types import ModuleType


Params = Union[str, List[str]]


def Base():
    return zodiax.base.Base


def filter_grad(parameters: Params, *filter_args, **filter_kwargs) -> Callable:
    """
    Applies the equinox filter_grad function to the input parameters. The
    corresponding equinox docs are found [here](https://docs.kidger.site/
    equinox/api/filtering/transformations/)

    Parameters
    ----------
    parameters : Union[str, List[str]]
        The parameters to filter. Can either be a single string path or a list
        of paths.
    *filter_args : Any
        The args to pass to the equinox filter_grad function.
    **filter_kwargs : Any
        The kwargs to pass to the equinox filter_grad function.

    Returns
    -------
    Callable
        The wrapped function.
    """

    def wrapper(func: Callable):
        @wraps(func)
        def inner_wrapper(pytree: PyTree, *args, **kwargs):
            # Convert parameters
            pytree = zodiax.set_array(pytree)
            boolean_filter = zodiax.tree.boolean_filter(pytree, parameters)

            # Wrap original function
            @equinox.filter_grad(*filter_args, **filter_kwargs)
            def recombine(traced: PyTree, static: PyTree):
                return func(eqx.combine(traced, static), *args, **kwargs)

            # Return wrapped function
            return recombine(*eqx.partition(pytree, boolean_filter))

        return inner_wrapper

    return wrapper


def filter_value_and_grad(
    parameters: Params, *filter_args, **filter_kwargs
) -> Callable:
    """
    Applies the equinox filter_value_and_grad function to the input parameters.
    The corresponding equinox docs are found [here](https://docs.kidger.site/
    equinox/api/filtering/transformations/)

    Parameters
    ----------
    parameters : Union[str, List[str]]
        The parameters to filter. Can either be a single string path or a list
        of paths.
    *filter_args : Any
        The args to pass to the equinox filter_value_and_grad function.
    **filter_kwargs : Any
        The kwargs to pass to the equinox filter_value_and_grad function.

    Returns
    -------
    Callable
        The wrapped function.
    """

    def wrapper(func: Callable):
        @wraps(func)
        def inner_wrapper(pytree: PyTree, *args, **kwargs):
            # Convert parameters
            pytree = zodiax.set_array(pytree)
            boolean_filter = zodiax.tree.boolean_filter(pytree, parameters)

            # Wrap original function
            @equinox.filter_value_and_grad(*filter_args, **filter_kwargs)
            def recombine(traced: PyTree, static: PyTree):
                return func(eqx.combine(traced, static), *args, **kwargs)

            # Return wrapped function
            return recombine(*eqx.partition(pytree, boolean_filter))

        return inner_wrapper

    return wrapper


def partition(
    pytree: Base(), parameters: Params, *partition_args, **partition_kwargs
) -> tuple:
    """
    Wraps the equinox partition function to take in a list of parameters to
    partition. The corresponding equinox docs are found [here](https://docs.
    kidger.site/equinox/api/filtering/transformations/)

    Parameters
    ----------
    pytree : Base()
        The pytree to partition.
    parameters : Union[str, List[str]]
        The parameters to partition. Can either be a single string path or a
        list of paths.
    *partition_args : Any
        The args to pass to the equinox partition function.
    **partition_kwargs : Any
        The kwargs to pass to the equinox partition function.

    Returns
    -------
    pytree1 : Base()
        A matching pytree with Nones at all leaves not specified by the
        parameters.
    pytree2 : Base()
        A matching pytree with Nones at all leaves specified by the parameters.
    """

    if isinstance(parameters, str):
        parameters = [parameters]

    pytree = zodiax.set_array(pytree)
    boolean_filter = zodiax.tree.boolean_filter(pytree, parameters)
    return equinox.partition(
        pytree, boolean_filter, *partition_args, **partition_kwargs
    )


# Dictionary of replaced functions
replaced_dict = {
    "filter_grad": filter_grad,
    "filter_value_and_grad": filter_value_and_grad,
    "partition": partition,
}


# Use the __all__ attribute of the external package to get a list of all
# public functions
external_api = [
    func_name for func_name in dir(equinox) if not func_name.startswith("_")
]


# Create a dictionary of API wrappers that simply point to the
# corresponding function/class/module from the equinox
wrappers = {}

for api_element in external_api:

    # Get the object corresponding to the api_element name
    api_obj = getattr(equinox, api_element)

    # Functions and classes
    if callable(api_obj):

        # if it is rewritten in zodiax, use the rewritten version
        if api_element in replaced_dict.keys():
            wrappers[api_element] = replaced_dict[api_element]

        # otherwise, just use the original equinox function
        else:
            wrappers[api_element] = getattr(equinox, api_element)

    # Modules
    elif isinstance(api_obj, ModuleType):
        wrappers[api_element] = getattr(equinox, api_element)

    # The rest of these should just be custom types that we dont need
    else:
        pass


# Push the wrappers to the global namespace
globals().update(wrappers)

# Adding the equinox public API to the __all__ attribute
# with zodiax rewrites
__all__ = list(wrappers.keys())
