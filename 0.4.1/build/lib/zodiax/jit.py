import zodiax
import jax.numpy as np
import jax.tree_util as jtu
from jax import Array
from typing import Union, List, Any


__all__ = ['improve_jit_hash']


Base = lambda: zodiax.base.Base
Params = Union[str, List[str]]


# Improved jit pytree
def _float_from_0d(leaf : Any):
    """
    Turns a 0d array into a float for hashing under jit.
    """
    if isinstance(leaf, Array):
        return float(leaf) if leaf.ndim == 0 else leaf
    else:
        return leaf

def improve_jit_hash(pytree : Base(), parameters : Params):
    """
    # TODO: Cast all leaves not specified by parameters to a numpy (not jax)
    array and all 0d arrays to floats, plus any other possile non-jax types. 
    This could improve performance under jit since this apparently changes the
    jaxpr under the hood, and could potentially result in them being treated
    as static values. This is currently not used since it is not clear if it
    actually improves performance.

    This *could* also be set to be the default actions in a zodiax version
    of equinox filter_jit. Would require changing default equinox filter
    function to be jax type arrays only.


    Replaces all scalar array leaves with python floats. This allows for those
    leaves to be marked as static under jit. The arguments in the params input
    are not cast to python types to prevent the pytree from being recompiled
    every time the arguments change under an optimisation loop.

    Parameters
    ----------
    pytree : zodiax.Base
        The pytree to be jitted.
    parameters : Union[List[str], str]
        The arguments to be optimised under the jit compilation.
    
    Returns
    -------
    pytree : zodiax.Base
        The pytree with scalar arrays leaves replaced with python floats.
    """
    # traced = zodiax.tree.boolean_filter(pytree, parameters)
    dynamic, static = zodiax.eqx.partition(pytree, parameters)
    static = jtu.tree_map(_float_from_0d, static)
    return zodiax.combine(dynamic, static)