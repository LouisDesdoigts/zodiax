import zodiax
from functools import wraps

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

def convert_to_filter(pytree, params):
    if isinstance(params, zodiax.base.ExtendedBase):
        return params
    else:
        return pytree.get_args(params)

def filter_grad(params, *filter_args, **filter_kwargs):
    def wrapper(func):
        @wraps(func)
        def inner_wrapper(pytree, *args, **kwargs):

            # Convert params to filter spec if not already
            filter_spec = convert_to_filter(pytree, params)

            # Wrap original function
            @fgrad(*filter_args, **filter_kwargs)
            def recombine(diff, non_diff):
                return func(combine(diff, non_diff), *args, **kwargs)
            
            # Return wrapped function
            return recombine(*partition(pytree, filter_spec))
        return inner_wrapper
    return wrapper

def filter_value_and_grad(params, *filter_args, **filter_kwargs):
    def wrapper(func):
        @wraps(func)
        def inner_wrapper(pytree, *args, **kwargs):

            # Convert params to filter spec if not already
            filter_spec = convert_to_filter(pytree, params)

            # Wrap original function
            @fvgrad(*filter_args, **filter_kwargs)
            def recombine(diff, non_diff):
                return func(combine(diff, non_diff), *args, **kwargs)
            
            # Return wrapped function
            return recombine(*partition(pytree, filter_spec))
        return inner_wrapper
    return wrapper