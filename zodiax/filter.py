import zodiax
from functools import wraps
from jaxtyping import PyTree
from typing import Union, Callable
from equinox import partition, combine

# Equinox modules
from equinox import ad
from equinox import jit
from equinox import make_jaxpr
from equinox import vmap_pmap
from equinox import callback
from equinox import eval_shape


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

# Autodiff
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
            @ad.filter_grad(*filter_args, **filter_kwargs)
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
            @ad.filter_value_and_grad(*filter_args, **filter_kwargs)
            def recombine(diff : PyTree, non_diff : PyTree):
                return func(combine(diff, non_diff), *args, **kwargs)
            
            # Return wrapped function
            return recombine(*partition(pytree, filter_spec))
        return inner_wrapper
    return wrapper


@wraps(ad.filter_jvp)
def filter_jvp(func, *filter_args, **filter_kwargs):
    """
    A simple wrapper fucntion for `equinox.filter_jvp`. The corresponding 
    equinox docs are found [here](https://docs.kidger.site/equinox/api/
    filtering/transformations/)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return ad.filter_jvp(func, 
                              *filter_args, **filter_kwargs)(*args, **kwargs)
    return wrapper


@wraps(ad.filter_vjp)
def filter_vjp(func, *filter_args, **filter_kwargs):
    """
    A simple wrapper fucntion for `equinox.filter_vjp`. The corresponding 
    equinox docs are found [here](https://docs.kidger.site/equinox/api/
    filtering/transformations/)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return ad.filter_vjp(func, 
                              *filter_args, **filter_kwargs)(*args, **kwargs)
    return wrapper


@wraps(ad.filter_closure_convert)
def filter_closure_convert(func, *filter_args, **filter_kwargs):
    """
    A simple wrapper fucntion for `equinox.filter_closure_convert`. The 
    corresponding equinox docs are found [here](https://docs.kidger.site/
    equinox/api/filtering/transformations/)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return ad.filter_closure_convert(func, 
                              *filter_args, **filter_kwargs)(*args, **kwargs)
    return wrapper


@wraps(ad.filter_custom_vjp)
def filter_custom_vjp(func, *filter_args, **filter_kwargs):
    """
    A simple wrapper fucntion for `equinox.filter_custom_vjp`. The corresponding 
    equinox docs are found [here](https://docs.kidger.site/equinox/api/
    filtering/transformations/)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return ad.filter_custom_vjp(func, 
                              *filter_args, **filter_kwargs)(*args, **kwargs)
    return wrapper


@wraps(ad.filter_custom_jvp)
def filter_custom_jvp(func, *filter_args, **filter_kwargs):
    """
    A simple wrapper fucntion for `equinox.filter_custom_jvp`. The corresponding 
    equinox docs are found [here](https://docs.kidger.site/equinox/api/
    filtering/transformations/)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return ad.filter_custom_jvp(func, 
                              *filter_args, **filter_kwargs)(*args, **kwargs)
    return wrapper


# Jit
@wraps(jit.filter_jit)
def filter_jit(func, *filter_args, **filter_kwargs):
    """
    A simple wrapper fucntion for `equinox.filter_jit`. The corresponding 
    equinox docs are found [here](https://docs.kidger.site/equinox/api/
    filtering/transformations/)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return jit.filter_jit(func, 
                              *filter_args, **filter_kwargs)(*args, **kwargs)
    return wrapper


# Callback
@wraps(callback.filter_pure_callback)
def filter_pure_callback(func, *filter_args, **filter_kwargs):
    """
    A simple wrapper fucntion for `equinox.filter_pure_callback`. The 
    corresponding equinox docs are found [here](https://docs.kidger.site/
    equinox/api/filtering/transformations/)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return callback.filter_pure_callback(func, 
                              *filter_args, **filter_kwargs)(*args, **kwargs)
    return wrapper


# Eval shape
@wraps(eval_shape.filter_eval_shape)
def filter_eval_shape(func, *filter_args, **filter_kwargs):
    """
    A simple wrapper fucntion for `equinox.filter_eval_shape`. The corresponding 
    equinox docs are found [here](https://docs.kidger.site/equinox/api/
    filtering/transformations/)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return eval_shape.filter_eval_shape(func, 
                              *filter_args, **filter_kwargs)(*args, **kwargs)
    return wrapper


# Vmap Pmap
@wraps(vmap_pmap.filter_vmap)
def filter_vmap(func, *filter_args, **filter_kwargs):
    """
    A simple wrapper fucntion for `equinox.filter_vmap`. The corresponding 
    equinox docs are found [here](https://docs.kidger.site/equinox/api/
    filtering/transformations/)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return vmap_pmap.filter_vmap(func, 
                              *filter_args, **filter_kwargs)(*args, **kwargs)
    return wrapper


@wraps(vmap_pmap.filter_pmap)
def filter_pmap(func, *filter_args, **filter_kwargs):
    """
    A simple wrapper fucntion for `equinox.filter_pmap`. The corresponding 
    equinox docs are found [here](https://docs.kidger.site/equinox/api/
    filtering/transformations/)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return vmap_pmap.filter_pmap(func, 
                              *filter_args, **filter_kwargs)(*args, **kwargs)
    return wrapper


# Make Jaxpr
@wraps(make_jaxpr.filter_make_jaxpr)
def filter_make_jaxpr(func, *filter_args, **filter_kwargs):
    """
    A simple wrapper fucntion for `equinox.filter_make_jaxpr`. The corresponding 
    equinox docs are found [here](https://docs.kidger.site/equinox/api/
    filtering/transformations/)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return make_jaxpr.filter_make_jaxpr(func, 
                              *filter_args, **filter_kwargs)(*args, **kwargs)
    return wrapper