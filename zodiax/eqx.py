import zodiax
import equinox
from functools import wraps
from jaxtyping import PyTree
from typing import Union, Callable, List
# from equinox import partition, combine
import equinox as eqx
from types import ModuleType


Params = Union[str, List[str]]
Base = lambda: zodiax.base.Base


def filter_grad(
    parameters : Params, 
    *filter_args,
    **filter_kwargs
    ) -> Callable:
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
    def wrapper(func : Callable):
        
        @wraps(func)
        def inner_wrapper(pytree : PyTree, *args, **kwargs):

            # Convert parameters
            boolean_filter = zodiax.tree.boolean_filter(pytree, parameters)

            # Wrap original function
            @equinox.filter_grad(*filter_args, **filter_kwargs)
            def recombine(traced : PyTree, static : PyTree):
                return func(eqx.combine(traced, static), *args, **kwargs)
            
            # Return wrapped function
            return recombine(*eqx.partition(pytree, boolean_filter))
        return inner_wrapper
    return wrapper


def filter_value_and_grad(
    parameters : Params, 
    *filter_args, 
    **filter_kwargs
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
    def wrapper(func : Callable):

        @wraps(func)
        def inner_wrapper(pytree : PyTree, *args, **kwargs):

            # Convert parameters
            boolean_filter = zodiax.tree.boolean_filter(pytree, parameters)

            # Wrap original function
            @equinox.filter_value_and_grad(*filter_args, **filter_kwargs)
            def recombine(traced : PyTree, static : PyTree):
                return func(eqx.combine(traced, static), *args, **kwargs)
            
            # Return wrapped function
            return recombine(*eqx.partition(pytree, boolean_filter))
        return inner_wrapper
    return wrapper


def partition(
    pytree : Base(), 
    parameters : Params, 
    *partition_args, 
    **partition_kwargs) -> tuple:
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
    boolean_filter = zodiax.tree.boolean_filter(pytree, parameters)
    return equinox.partition(pytree, boolean_filter, *partition_args,
        **partition_kwargs)


# Dictionary of replaced functions
replaced_dict = {
    'filter_grad'           : filter_grad,
    'filter_value_and_grad' : filter_value_and_grad,
    'partition'             : partition,
}


# Use the __all__ attribute of the external package to get a list of all 
# public functions
external_functions = [func_name for func_name in dir(equinox) 
    if not func_name.startswith("_")]


# Define what function we want to overwrite
replace = list(replaced_dict.keys())


# Create a dictionary of wrapper functions that simply call the corresponding 
# function from the external package
wrapper_functions = {}
replaced_functions = []
for func_name in external_functions:
    param = getattr(equinox, func_name)

    # Import functions in the namespace and push them to zodiax namespace
    if callable(param):
        replaced_functions.append(func_name)
        if func_name in replaced_dict.keys():
            wrapper_functions[func_name] = replaced_dict[func_name]
        else:
            wrapper_functions[func_name] = getattr(equinox, func_name)
    
    # If it is a module import it as a submodule
    elif isinstance(param, ModuleType):
        wrapper_functions[func_name] = getattr(equinox, func_name)
    
    # The rest of these should just be custom types that we dont need
    else:
        pass


# Export the wrapper functions from the module
__all__ = replaced_functions
globals().update(wrapper_functions)