import zodiax
from functools import wraps
from jaxtyping import Array, PyTree
from typing import Union, Callable, Any

# Wrapped imports
from equinox import partition, combine, \
    filter_grad as fgrad, \
    filter_value_and_grad as fvgrad
    
# Unwrapped import
from equinox import filter_jit, filter_make_jaxpr, filter_eval_shape, \
    filter_jvp, filter_vjp, filter_custom_jvp, filter_custom_vjp, \
    filter_closure_convert, filter_vmap, filter_pmap, filter_pure_callback


__all__ = ["filter_grad", "filter_value_and_grad", "filter_jit", 
           "filter_make_jaxpr", "filter_eval_shape", "filter_jvp", 
           "filter_vjp", "filter_custom_jvp", "filter_custom_vjp", 
           "filter_closure_convert", "filter_vmap", "filter_pmap", 
           "filter_pure_callback"]

def convert_to_filter(pytree : PyTree, 
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
    if isinstance(params, zodiax.base.ExtendedBase):
        return params
    else:
        return pytree.get_args(params)

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
            filter_spec = convert_to_filter(pytree, params)

            # Wrap original function
            @fgrad(*filter_args, **filter_kwargs)
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
            filter_spec = convert_to_filter(pytree, params)

            # Wrap original function
            @fvgrad(*filter_args, **filter_kwargs)
            def recombine(diff : PyTree, non_diff : PyTree):
                return func(combine(diff, non_diff), *args, **kwargs)
            
            # Return wrapped function
            return recombine(*partition(pytree, filter_spec))
        return inner_wrapper
    return wrapper