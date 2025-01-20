import zodiax
import equinox as eqx
import jax
from jax import config, Array, numpy as np
from typing import Union, List, Any


__all__ = ["boolean_filter", "set_array"]


def Base():
    return zodiax.base.Base


Params = Union[str, List[str]]


# Boolean
def boolean_filter(pytree: Base(), parameters: Params, inverse: bool = False) -> Base():
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
        false_pytree = jax.tree.map(lambda _: False, pytree)
        return false_pytree.set(parameters, len(parameters) * [True])
    else:
        true_pytree = jax.tree.map(lambda _: True, pytree)
        return true_pytree.set(parameters, len(parameters) * [False])


def set_array(pytree: Base(), parameters=None) -> Base():
    """
    Converts all leaves in the pytree to arrays to ensure they have a
    .shape property for static dimensionality and size checks.

    Parameters
    ----------
    pytree : Base()
        The pytree to be converted.

    Returns
    -------
    pytree : Base()
        The pytree with the leaves converted to arrays.
    """

    # Old routine for setting specificed parameters
    if parameters is not None:
        new_leaves = jax.tree.map(_to_array, pytree.get(parameters))
        return pytree.set(parameters, new_leaves)

    # else convert all leaves to arrays

    # grabbing float data type
    dtype = np.float64 if config.x64_enabled else np.float32

    # partitioning the pytree into arrays and other
    floats, other = eqx.partition(pytree, zodiax.is_inexact_array_like)

    # converting the floats to arrays
    floats = jax.tree.map(lambda x: np.array(x, dtype=dtype), floats)

    # recombining
    return eqx.combine(floats, other)


def _to_array(leaf: Any):
    if not isinstance(leaf, Array):
        return np.asarray(leaf, dtype=float)
    else:
        return leaf
