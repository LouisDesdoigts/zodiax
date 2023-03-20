import zodiax
import equinox
from functools import wraps
from jaxtyping import PyTree
from typing import Union, Callable
from equinox import partition, combine
from types import ModuleType


def _convert_to_filter(pytree : PyTree, 
                       params : Union[PyTree, list, str]) -> PyTree:
    """
    Converts the input params to a filter spec if not already.

    Parameters
    ----------
    pytree : PyTree
        The pytree to filter.
    params : Union[PyTree, list, str]
        The params to filter, or an existing filter.
    
    Returns
    -------
     : PyTree
        The filter spec.
    """
    if isinstance(params, zodiax.base.Base):
        return params
    else:
        return zodiax.tree.get_args(pytree, params)


def filter_grad(params : Union[PyTree, list, str], 
                *filter_args, **filter_kwargs) -> Callable:
    """
    Applies the equinox filter_grad function to the input params. The 
    corresponding equinox docs are found [here](https://docs.kidger.site/
    equinox/api/filtering/transformations/)


    Parameters
    ----------
    params : Union[PyTree, list, str]
        The params to filter. Can either be a single string path, a list of 
        paths, or a pytree with binary leaves denoting which argument to take
        gradients with respect to.
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

            # Convert params to filter spec if not already
            filter_spec = _convert_to_filter(pytree, params)

            # Wrap original function
            @equinox.filter_grad(*filter_args, **filter_kwargs)
            def recombine(diff : PyTree, non_diff : PyTree):
                return func(combine(diff, non_diff), *args, **kwargs)
            
            # Return wrapped function
            return recombine(*partition(pytree, filter_spec))
        return inner_wrapper
    return wrapper


def filter_value_and_grad(params : Union[PyTree, list, str], 
                *filter_args, **filter_kwargs) -> Callable:
    """
    Applies the equinox filter_value_and_grad function to the input params. The 
    corresponding equinox docs are found [here](https://docs.kidger.site/
    equinox/api/filtering/transformations/)

    Parameters
    ----------
    params : Union[PyTree, list, str]
        The params to filter. Can either be a single string path, a list of 
        paths, or a pytree with binary leaves denoting which argument to take
        gradients with respect to.
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

            # Convert params to filter spec if not already
            filter_spec = _convert_to_filter(pytree, params)

            # Wrap original function
            @equinox.filter_value_and_grad(*filter_args, **filter_kwargs)
            def recombine(diff : PyTree, non_diff : PyTree):
                return func(combine(diff, non_diff), *args, **kwargs)
            
            # Return wrapped function
            return recombine(*partition(pytree, filter_spec))
        return inner_wrapper
    return wrapper


# Dictionary of replaced functions
replaced_dict = {
    'filter_grad': filter_grad,
    'filter_value_and_grad': filter_value_and_grad
}


# Use the __all__ attribute of the external package to get a list of all 
# public functions
external_functions = [func_name for func_name in dir(equinox) 
    if not func_name.startswith("_")]


# Define what function we want to overwrite
replace = ['filter_grad', 'filter_value_and_grad']


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