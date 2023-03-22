import zodiax
from jax.tree_util import tree_map
from equinox import is_array, filter as eqx_filter
from optax import adam, multi_transform, GradientTransformation
from typing import Union
from jaxtyping import PyTree


__all__ = ["get_optimiser"]


def get_optimiser(pytree     : PyTree,
                  paths      : Union[str, list],
                  optimisers : Union[GradientTransformation, list],
                  ) -> tuple:
    """
    Returns an Optax.GradientTransformion object, with the optimisers
    specified by optimisers applied to the leaves specified by paths.

    Parameters
    ----------
    paths : Union[str, list]
        A path or list of paths or list of nested paths.
    optimisers : Union[optax.GradientTransformation, list]
        A optax.GradientTransformation or list of
        optax.GradientTransformation objects to be applied to the leaves
        specified by paths.

    Returns
    -------
    (optimiser, state) : tuple
        A tuple of (Optax.GradientTransformion, optax.MultiTransformState)
        objects, with the optimisers applied to the leaves specified by
        paths, and the initialised optimisation state.
    """
    # Pre-wrap single inputs into a list since optimisers have a length of 2
    if not isinstance(optimisers, list):
        optimisers = [optimisers]
    
    # Paths have to default be wrapped in a list to match optimiser
    if isinstance(paths, str):
        paths = [paths]

    # Construct groups and get param_spec
    groups = [str(i) for i in range(len(optimisers))]
    param_spec = tree_map(lambda _: "null", pytree)
    param_spec = param_spec.set(paths, groups)

    # Generate optimiser dictionary
    opt_dict = dict([(groups[i], optimisers[i]) \
                        for i in range(len(groups))])

    # Assign the null group
    # TODO: Can this be set to None?
    opt_dict["null"] = adam(0.0)

    # Get optimiser object
    optim = multi_transform(opt_dict, param_spec)

    # Get filtered optimiser
    opt_state = optim.init(eqx_filter(pytree, is_array))

    # Return
    return (optim, opt_state)