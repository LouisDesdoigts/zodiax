import zodiax
import jax.numpy as np
from jax.tree_util import tree_map
from equinox import partition, combine
from typing import Union


__all__ = ['get_jit_model']


def _float_from_0d(leaf):
    """
    Turns a 0d array into a float for hashing under jit.
    """
    if isinstance(leaf, np.ndarray):
        return float(leaf) if leaf.ndim == 0 else leaf
    else:
        return leaf


def get_jit_model(model : zodiax.base.Base, params : Union[list[str], str]):
    """
    Replaces all scalar array leaves with python floats. This allows for those
    leaves to be marked as static under jit. The arguments in the params input
    are not cast to python types to prevent the model from being recompiled
    every time the arguments change under an optimisation loop.

    Parameters
    ----------
    model : zodiax.Base
        The model to be jitted.
    params : Union[list[str], str]
        The arguments to be optimised under the jit compilation.
    
    Returns
    -------
    model : zodiax.Base
        The model with scalar arrays leaves replaced with python floats.
    """
    args = zodiax.optimisation._convert_to_filter(model, params)
    opt, non_opt = partition(model, args)
    float_model = tree_map(_float_from_0d, non_opt)
    return combine(opt, float_model)