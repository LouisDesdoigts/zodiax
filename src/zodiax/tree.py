import equinox as eqx
import jax.tree as jtu
from jax import config, Array, numpy as np
from typing import Union, Any
import warnings

__all__ = ["boolean_filter", "set_array"]


PyTree = Union[dict, list, tuple, eqx.Module]
Params = Union[str, list[str], tuple[str]]
Values = Union[Any, list[Any], tuple[Any]]


# Boolean
def boolean_filter(pytree: PyTree, parameters: Params, inverse: bool = False) -> PyTree:
    """
    Returns a pytree of matching structure with boolean values at the leaves.
    Leaves specified by paths will be True, all others will be False.

    TODO: Possibly improve by setting both true and false simultaneously.
    Maybe do this with jax keypaths?

    Parameters
    ----------
    pytree : PyTree
        The pytree to be filtered.
    parameters : Params
        A path or list of paths or list of nested paths.
    inverse : bool = False
        If True, the boolean values will be inverted, by default False

    Returns
    -------
    args : PyTree
        An pytree of matching structre with boolean values at the leaves.
    """
    warnings.warn(
        "boolean_filter is deprecated as of v0.5.0 and will be removed in v0.6.0",
        DeprecationWarning,
    )
    parameters = parameters if isinstance(parameters, list) else [parameters]
    if not inverse:
        false_pytree = jtu.map(lambda _: False, pytree)
        return false_pytree.set(parameters, len(parameters) * [True])
    else:
        true_pytree = jtu.map(lambda _: True, pytree)
        return true_pytree.set(parameters, len(parameters) * [False])


def set_array(pytree: PyTree, parameters=None) -> PyTree:
    """
    Converts all leaves in the pytree to arrays to ensure they have a
    .shape property for static dimensionality and size checks.

    Parameters
    ----------
    pytree : PyTree
        The pytree to be converted.

    Returns
    -------
    pytree : PyTree
        The pytree with the leaves converted to arrays.
    """
    warnings.warn(
        "set_array is deprecated as of v0.5.0 and will be removed in v0.6.0",
        DeprecationWarning,
    )
    # Old routine for setting specified parameters
    if parameters is not None:
        new_leaves = jtu.map(_to_array, pytree.get(parameters))
        return pytree.set(parameters, new_leaves)

    # else convert all leaves to arrays

    # grabbing float data type
    dtype = np.float64 if config.x64_enabled else np.float32

    # partitioning the pytree into arrays and other
    floats, other = eqx.partition(pytree, eqx.is_inexact_array_like)

    # converting the floats to arrays
    floats = jtu.map(lambda x: np.array(x, dtype=dtype), floats)

    # recombining
    return eqx.combine(floats, other)


def _to_array(leaf: Any):
    if not isinstance(leaf, Array):
        return np.asarray(leaf, dtype=float)
    else:
        return leaf
