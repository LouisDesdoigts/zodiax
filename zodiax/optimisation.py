import jax
import jax.numpy as np
import equinox as eqx
import zodiax
from .base import ModelParams
import optax
from optax import GradientTransformation
from typing import Union, List


__all__ = [
    "scheduler",
    "sgd",
    "adam",
    "debug_nan_check",
    "zero_nan_check",
    "get_optimiser",
]


def Base():
    return zodiax.base.Base


Params = Union[str, List[str]]
Optimisers = Union[GradientTransformation, list]


def scheduler(lr: float, start: int, *args):
    """
    Function to easily interface with the optax library to create a piecewise
    constant learning rate schedule. The function takes a learning rate, a
    starting step and optionally, a variable number of tuples. Each tuple
    should contain a step and a multiplier; the learning rate will be multiplied
    by the corresponding multiplier at the specified step.

    Parameters
    ----------
    lr : float
        The initial learning rate.
    start : int
        The starting step (learning rate will be ~0 before this).
    args : tuple
        A variable number of tuples, each containing a step and a multiplier.

    Returns
    -------
    schedule : optax.schedule
        The piecewise constant learning rate schedule.
    """

    # Create a dictionary to store the schedule
    sched_dict = {start: 1e100}

    # looping over learning rate updates
    for start, mul in args:
        sched_dict[start] = mul

    return optax.piecewise_constant_schedule(lr / 1e100, sched_dict)


def _base_sgd(vals):
    return optax.sgd(vals, nesterov=True, momentum=0.6)


def _base_adam(vals):
    return optax.adam(vals)


def sgd(lr: float, start: int, *schedule):
    """
    Wrapper for the optax SGD optimiser with a piecewise constant learning rate
    schedule.

    Parameters
    ----------
    lr : float
        The initial learning rate.
    start : int
        The starting step (learning rate will be ~0 before this).
    args : tuple
        A variable number of tuples, each containing a step and a multiplier.

    Returns
    -------
    optimiser : optax.sgd
        The optimiser with the piecewise constant learning rate schedule.
    """
    return _base_sgd(scheduler(lr, start, *schedule))


def adam(lr: float, start: int, *schedule):
    """
    Wrapper for the optax Adam optimiser with a piecewise constant learning rate
    schedule.

    Parameters
    ----------
    lr : float
        The initial learning rate.
    start : int
        The starting step (learning rate will be ~0 before this).
    args : tuple
        A variable number of tuples, each containing a step and a multiplier.

    Returns
    -------
    optimiser : optax.adam
        The optimiser with the piecewise constant learning rate schedule.
    """
    return _base_adam(scheduler(lr, start, *schedule))


def debug_nan_check(grads: Base()) -> Base():
    """
    Checks for NaN values in the gradients and triggers a breakpoint if any are found.

    Parameters
    ----------
    grads : PyTree
        The gradients to be checked for NaN values.

    Returns
    -------
    grads : PyTree
        The gradients.
    """
    bool_tree = jax.tree_map(lambda x: np.isnan(x).any(), grads)
    vals = np.array(jax.tree_util.tree_flatten(bool_tree)[0])
    eqx.debug.breakpoint_if(vals.sum() > 0)
    return grads


def zero_nan_check(grads: Base()) -> Base():
    """
    Replaces any NaN values in the gradients and with zeros.

    Parameters
    ----------
    grads : PyTree
        The gradients to be checked for NaN values.

    Returns
    -------
    grads : PyTree
        The gradients with NaN values replaced by zeros.
    """
    return jax.tree_map(lambda x: np.where(np.isnan(x), 0.0, x), grads)


def get_optimiser(pytree: Base(), optimisers: dict, parameters: Params = None):
    """
    Returns an Optax.GradientTransformion object, with the optimisers
    specified by optimisers applied to the leaves specified by parameters.

    Parameters
    ----------
    pytree : Base
        A zodiax.base.Base object containing the model parameters.
    optimisers : dict
        A dictionary of optax.GradientTransformation objects to be applied
        to the leaves specified by parameters.
    parameters : Union[str, List[str]] = None
        A path or list of parameters or list of nested parameters. If None,
        all parameters in optimisers will be optimised.

    Returns
    -------
    model_params : ModelParams
        A ModelParams object containing the model parameters.
    optim : optax.GradientTransformion
        An optax.GradientTransformion object with the optimisers applied to the
        leaves specified by parameters.
    state : optax.MultiTransformState
        The initialised optimisation state.
    """

    # Get the parameters and opt_dict
    if parameters is not None:
        optimisers = dict([(p, optimisers[p]) for p in parameters])
    else:
        parameters = list(optimisers.keys())

    # Get the model parameters and optimiser
    model_params = ModelParams(dict([(p, pytree.get(p)) for p in parameters]))
    param_spec = ModelParams(dict([(param, param) for param in parameters]))
    optim = optax.multi_transform(optimisers, param_spec)

    # Build the optimised object - the 'model_params' object
    state = optim.init(model_params)
    return model_params, optim, state


# NOTE old version, consider if its worth keeping
# def get_optimiser(
#     pytree: Base(),
#     parameters: Params,
#     optimisers: Optimisers,
# ) -> tuple:
#     """
#     Returns an Optax.GradientTransformion object, with the optimisers
#     specified by optimisers applied to the leaves specified by parameters.

#     Parameters
#     ----------
#     pytree : Base
#         A zodiax.base.Base object.
#     parameters :  Union[str, List[str]]
#         A path or list of parameters or list of nested parameters.
#     optimisers : Union[optax.GradientTransformation, list]
#         A optax.GradientTransformation or list of
#         optax.GradientTransformation objects to be applied to the leaves
#         specified by parameters.

#     Returns
#     -------
#     optimiser : optax.GradientTransformion
#         TODO Update
#         A tuple of (Optax.GradientTransformion, optax.MultiTransformState)
#         objects, with the optimisers applied to the leaves specified by
#         parameters, and the initialised optimisation state.
#     state : optax.MultiTransformState
#     """
#     # Pre-wrap single inputs into a list since optimisers have a length of 2
#     if not isinstance(optimisers, list):
#         optimisers = [optimisers]

#     # parameters have to default be wrapped in a list to match optimiser
#     if isinstance(parameters, str):
#         parameters = [parameters]

#     # Construct groups and get param_spec
#     groups = [str(i) for i in range(len(optimisers))]
#     param_spec = tree_map(lambda _: "null", pytree)
#     param_spec = param_spec.set(parameters, groups)

#     # Generate optimiser dictionary and Assign the null group
#     opt_dict = dict([(groups[i], optimisers[i]) for i in range(len(groups))])
#     opt_dict["null"] = adam(0.0)

#     # Get optimiser object and filtered optimiser
#     optim = multi_transform(opt_dict, param_spec)
#     opt_state = optim.init(eqx_filter(pytree, is_array))

#     # Return
#     return (optim, opt_state)
