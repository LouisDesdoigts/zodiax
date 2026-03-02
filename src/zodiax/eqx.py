import equinox as eqx
from functools import wraps
from typing import Union, Callable, Any
import warnings
from .tree import set_array, boolean_filter

__all__ = ["filter_grad", "filter_value_and_grad", "partition"]

PyTree = Union[dict, list, tuple, eqx.Module]
Params = Union[str, list[str], tuple[str]]
Values = Union[Any, list[Any], tuple[Any]]


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
    warnings.warn(
        "filter_grad is deprecated as of v4.1 and will be removed in v5.1",
        DeprecationWarning,
    )

    def wrapper(func: Callable):
        @wraps(func)
        def inner_wrapper(pytree, *args, **kwargs):
            # Convert parameters
            pytree = set_array(pytree)
            bool_filter = boolean_filter(pytree, parameters)

            # Wrap original function
            @eqx.filter_grad(*filter_args, **filter_kwargs)
            def recombine(traced, static):
                return func(eqx.combine(traced, static), *args, **kwargs)

            # Return wrapped function
            return recombine(*eqx.partition(pytree, bool_filter))

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
    warnings.warn(
        "filter_value_and_grad is deprecated as of v4.1 and will be removed in v5.1",
        DeprecationWarning,
    )

    def wrapper(func: Callable):
        @wraps(func)
        def inner_wrapper(pytree, *args, **kwargs):
            # Convert parameters
            pytree = set_array(pytree)
            bool_filter = boolean_filter(pytree, parameters)

            # Wrap original function
            @eqx.filter_value_and_grad(*filter_args, **filter_kwargs)
            def recombine(traced, static):
                return func(eqx.combine(traced, static), *args, **kwargs)

            # Return wrapped function
            return recombine(*eqx.partition(pytree, bool_filter))

        return inner_wrapper

    return wrapper


def partition(
    pytree: PyTree, parameters: Params, *partition_args, **partition_kwargs
) -> tuple:
    """
    Wraps the equinox partition function to take in a list of parameters to
    partition. The corresponding equinox docs are found [here](https://docs.
    kidger.site/equinox/api/filtering/transformations/)

    Parameters
    ----------
    pytree : PyTree
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
    pytree1 : PyTree
        A matching pytree with Nones at all leaves not specified by the
        parameters.
    pytree2 : PyTree
        A matching pytree with Nones at all leaves specified by the parameters.
    """

    if isinstance(parameters, str):
        parameters = [parameters]

    pytree = set_array(pytree)
    bool_filter = boolean_filter(pytree, parameters)
    return eqx.partition(pytree, bool_filter, *partition_args, **partition_kwargs)
