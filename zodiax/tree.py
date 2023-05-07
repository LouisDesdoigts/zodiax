import zodiax
import jax.numpy as np
import jax.tree_util as jtu
from jax import Array
from jaxtyping import PyTree
from typing import Union, List, Any
from equinox import partition, combine


__all__ = ['boolean_filter', 'set_array']


Base = lambda: zodiax.base.Base
Params = Union[str, List[str]]


# Boolean
def boolean_filter(
    pytree     : Base(), 
    parameters : Params, 
    inverse    : bool = False) -> Base():
    """
    Returns a pytree of matching structure with boolean values at the leaves.
    Leaves specified by paths will be True, all others will be False. 

    TODO: Possibly improve by setting both true and false simultaneously.
    Maybe do this with jax keypaths?

    Parameters
    ----------
    pytree : PyTree
        The pytree to be filtered.
    parameters : Union[str, list]
        A path or list of paths or list of nested paths.
    inverse : bool = False
        If True, the boolean values will be inverted, by default False

    Returns
    -------
    args : PyTree
        An pytree of matching structre with boolean values at the leaves.
    """
    parameters = parameters if isinstance(parameters, list) else [parameters]
    if not inverse:
        false_pytree = jtu.tree_map(lambda _: False, pytree)
        return false_pytree.set(parameters, len(parameters) * [True])
    else:
        true_pytree = jtu.tree_map(lambda _: True, pytree)
        return true_pytree.set(parameters, len(parameters) * [False])


# Array
def _to_array(leaf : Any):
    if not isinstance(leaf, Array):
        return np.asarray(leaf, dtype=float)
    else:
        return leaf

def set_array(pytree : Base(), parameters : Params) -> Base():
    """
    Converts all leaves specified by parameters in the pytree to arrays to 
    ensure they have a .shape property for static dimensionality and size
    checks. This allows for 'dynamicly generated' array shapes from the path
    based `parameters` input. This is used for dynamically generating the
    latent X parameter that we need to generate in order to calculate the
    hessian.

    Parameters
    ----------
    pytree : Base()
        The pytree to be converted.
    parameters : Params
        The leaves to be converted to arrays.
    
    Returns
    -------
    pytree : Base()
        The pytree with the specified leaves converted to arrays.
    """
    new_leaves = jtu.tree_map(_to_array, pytree.get(parameters))
    return pytree.set(parameters, new_leaves)