from jax.tree_util import tree_map
from typing import Union
from jaxtyping import PyTree


__all__ = ["get_args"] #, "get_param_spec"]


def get_args(pytree: PyTree, paths : Union[str, list]) -> PyTree:
    """
    Returns a pytree of matching structure with boolean values at the leaves.
    Leaves specified by paths will be True, all others will be False. 

    Parameters
    ----------
    paths : Union[str, list]
        A path or list of paths or list of nested paths.

    Returns
    -------
    args : PyTree
        An pytree of matching structre with boolean values at the leaves.
    """
    args = tree_map(lambda _: False, pytree)
    paths = paths if isinstance(paths, list) else [paths]
    values = len(paths) * [True]
    return args.set(paths, values)


# def get_param_spec(pytree   : PyTree,
#                    paths    : Union[str, list],
#                    groups   : Union[str, list],
#                    get_args : bool = False) -> PyTree:
#     """
#     Returns 'param_spec' object, to be used in conjunction with the
#     Optax.multi_transform functions. The param_spec is a pytree of matching
#     strucutre that has strings assigned to every node, denoting the group
#     that is belongs to. Each of these groups can then have unique optimiser
#     objects assigned to them. This is typically used to assign different
#     learning rates to different parameters.

#     Note this sets the default or non-trainable group to 'null'.

#     Parameters
#     ----------
#     paths : Union[str, list]
#         A path or list of paths or list of nested paths.
#     groups : Union[str, list]
#         A string or list of strings, denoting which group to assign the
#         corresponding leaves denoted by paths to.
#     get_args : bool = False
#         Return a corresponding args pytree or not.

#     Returns
#     -------
#     param_spec : PyTree
#         An pytree of matching structre with string values at the leaves
#         specified by groups.
#     """


#     return param_spec if not get_args \
#         else (param_spec, pytree.get_args(paths))