import zodiax
from jax.tree_util import tree_map
from equinox import is_array, filter as eqx_filter
from optax import adam, multi_transform, GradientTransformation
from typing import Union, List
from jaxtyping import PyTree


__all__ = ["get_optimiser"]


Base = lambda: zodiax.base.Base
Params = Union[str, List[str]]
Optimisers = Union[GradientTransformation, list]


def get_optimiser(pytree     : Base(),
                  parameters : Params,
                  optimisers : Optimisers,
                  ) -> tuple:
    """
    Returns an Optax.GradientTransformion object, with the optimisers
    specified by optimisers applied to the leaves specified by parameters.

    Parameters
    ----------
    pytree : Base
        A zodiax.base.Base object.
    parameters :  Union[str, List[str]]
        A path or list of parameters or list of nested parameters.
    optimisers : Union[optax.GradientTransformation, list]
        A optax.GradientTransformation or list of
        optax.GradientTransformation objects to be applied to the leaves
        specified by parameters.

    Returns
    -------
    optimiser : optax.GradientTransformion
        TODO Update
        A tuple of (Optax.GradientTransformion, optax.MultiTransformState)
        objects, with the optimisers applied to the leaves specified by
        parameters, and the initialised optimisation state.
    state : optax.MultiTransformState
    """
    # Pre-wrap single inputs into a list since optimisers have a length of 2
    if not isinstance(optimisers, list):
        optimisers = [optimisers]
    
    # parameters have to default be wrapped in a list to match optimiser
    if isinstance(parameters, str):
        parameters = [parameters]

    # Construct groups and get param_spec
    groups = [str(i) for i in range(len(optimisers))]
    param_spec = tree_map(lambda _: "null", pytree)
    param_spec = param_spec.set(parameters, groups)

    # Generate optimiser dictionary and Assign the null group
    opt_dict = dict([(groups[i], optimisers[i]) \
                        for i in range(len(groups))])
    opt_dict["null"] = adam(0.0)

    # Get optimiser object and filtered optimiser
    optim = multi_transform(opt_dict, param_spec)
    opt_state = optim.init(eqx_filter(pytree, is_array))

    # Return
    return (optim, opt_state)