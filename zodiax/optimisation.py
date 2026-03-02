import jax
import jax.numpy as np
import jax.tree as jtu
import equinox as eqx

# from .base import ModelParams
import optax
from typing import Union, Any
from .base import _unpack
import warnings

__all__ = [
    "debug_nan_check",
    "zero_nan_check",
    # New
    "delay",
    "map_optimisers",
    "decompose",
    "eigen_projection",
    # Deprecated
    "scheduler",
    "sgd",
    "adam",
    "get_optimiser",
]


PyTree = Union[dict, list, tuple, eqx.Module]
Params = Union[str, list[str], tuple[str]]
Values = Union[Any, list[Any], tuple[Any]]

# To be deprecated
Optimisers = Union[optax.GradientTransformation, list]

if jax.config.read("jax_enable_x64"):
    BIG = np.finfo(np.float64).max / 1e1
else:
    BIG = np.finfo(np.float32).max / 1e1


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
    warnings.warn(
        "scheduler is deprecated as of v4.1 and will be removed in v5.1",
        DeprecationWarning,
    )

    # Create a dictionary to store the schedule
    sched_dict = {start: BIG}

    # looping over learning rate updates
    for start, mul in args:
        sched_dict[start] = mul

    return optax.piecewise_constant_schedule(lr / BIG, sched_dict)


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
    warnings.warn(
        "sgd is deprecated as of v4.1 and will be removed in v5.1",
        DeprecationWarning,
    )
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
    warnings.warn(
        "adam is deprecated as of v4.1 and will be removed in v5.1",
        DeprecationWarning,
    )
    return _base_adam(scheduler(lr, start, *schedule))


def debug_nan_check(grads: PyTree) -> PyTree:
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
    bool_tree = jax.tree.map(lambda x: np.isnan(x).any(), grads)
    vals = np.array(jax.tree.flatten(bool_tree)[0])
    eqx.debug.breakpoint_if(vals.sum() > 0)
    return grads


def zero_nan_check(grads: PyTree) -> PyTree:
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
    return jax.tree.map(lambda x: np.where(np.isnan(x), 0.0, x), grads)


# NOTE old version, consider if its worth keeping
def get_optimiser(
    pytree: PyTree,
    parameters: Params,
    optimisers: Optimisers,
) -> tuple:
    """
    Returns an Optax.GradientTransformion object, with the optimisers
    specified by optimisers applied to the leaves specified by parameters.

    Parameters
    ----------
    pytree : PyTree
        A zodiax.base.PyTree object.
    parameters : Params
        A path or list of parameters or list of nested parameters.
    optimisers : Optimisers
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
    warnings.warn(
        "get_optimiser is deprecated as of v4.1 and will be removed in v5.1",
        DeprecationWarning,
    )

    # Pre-wrap single inputs into a list since optimisers have a length of 2
    if not isinstance(optimisers, list):
        optimisers = [optimisers]

    # parameters have to default be wrapped in a list to match optimiser
    if isinstance(parameters, str):
        parameters = [parameters]

    # Construct groups and get param_spec
    groups = [str(i) for i in range(len(optimisers))]
    param_spec = jax.tree.map(lambda _: "null", pytree)
    param_spec = param_spec.set(parameters, groups)

    # Generate optimiser dictionary and Assign the null group
    opt_dict = dict([(groups[i], optimisers[i]) for i in range(len(groups))])
    opt_dict["null"] = optax.sgd(0.0)

    # Get optimiser object and filtered optimiser
    optim = optax.multi_transform(opt_dict, param_spec)
    opt_state = optim.init(eqx.filter(pytree, eqx.is_array))

    # Return
    return (optim, opt_state)


### New Functions ###
def map_optimisers(params, optimisers, strict=False):

    # unpack the dicts to ensure matching structures
    params = _unpack(params)
    optimisers = _unpack(optimisers)

    # Make sure all parameters are floating-point jax arrays so they return gradients
    params = jtu.map(lambda x: np.array(x, float), params)

    if strict and params.keys() != optimisers.keys():
        raise ValueError("Params and optimisers must have the same keys")

    # Check for keys in params that aren't in optimisers, and paste an empty optimiser
    for key in params.keys():
        if key not in optimisers.keys():
            optimisers[key] = optax.identity()

    # Check for keys in optimisers that aren't in params, and set to zero
    for key in optimisers.keys():
        if key not in params.keys():
            params[key] = 0.0

    param_spec = {param: param for param in optimisers.keys()}
    optim = optax.multi_transform(optimisers, param_spec)
    state = optim.init(params)
    return optim, state


def delay(lr: float, start: int, length: int = 1):
    return optax.linear_schedule(0.0, lr, length, start)


def decompose(matrix, hermitian=True, normalise=False):
    """
    Returns:
        eigvals: (D,) array sorted descending
        eigvecs: (D, D) array where each ROW is an eigenvector
    """
    if hermitian:
        eigvals, eigvecs = np.linalg.eigh(matrix)
        # Flip to descending and transpose so rows = vectors
        eigvals, eigvecs = eigvals[::-1], eigvecs.T[::-1]
    else:
        eigvals, eigvecs = np.linalg.eig(matrix)
        # Sort manually for non-hermitian consistency
        idx = np.argsort(eigvals.real)[::-1]
        eigvals, eigvecs = eigvals.real[idx], eigvecs.real.T[idx]

    if normalise:
        eigvals /= eigvals[0]

    return eigvals, eigvecs


def eigen_projection(fmat=None, cov=None):
    # Make sure we have one input
    if fmat is None and cov is None:
        raise ValueError("Must provide either fmat or cov")

    # Select the matrix to decompose
    mat = fmat if fmat is not None else cov

    # Ensure symmetry to avoid complex eigvals from numerical noise
    mat = (mat + mat.T) / 2.0

    # Decompose the matrix to get eigenvalues and eigenvectors
    vals, vecs = decompose(mat, normalise=False)

    # 1. Extract physical scales (the diagonal)
    # For Fisher, diag is 1/sigma^2. For Cov, diag is sigma^2.
    diag = np.diag(mat)

    if fmat is not None:
        # Physical 'step' size for each parameter
        phys_scale = 1.0 / np.sqrt(diag)
    else:
        phys_scale = np.sqrt(diag)

    # 2. Convert to Correlation-like matrix (dimensionless)
    # This prevents physical units from dominating the eigenvalue spectrum
    S_inv = 1.0 / np.sqrt(diag)
    norm_mat = S_inv[:, None] * mat * S_inv[None, :]

    # 3. Decompose the normalized matrix
    # All diagonal elements are now 1.0.
    # Eigenvalues now represent 'redundancy' or 'degeneracy' rather than 'units'.
    vals, vecs = decompose(norm_mat, normalise=False)

    # 4. Build the Projection Matrix P
    # P = (Physical Scale) * (Rotation) * (Orthonormal Scaling)
    # We apply the phys_scale back to return to the original units
    if fmat is not None:
        # For Fisher, eigenvalues of norm_mat are 'information'
        inner_scale = 1.0 / np.sqrt(vals)
    else:
        # For Cov, eigenvalues of norm_mat are 'variance'
        inner_scale = np.sqrt(vals)

    # P maps: (Reduced Space) -> (Dimensionless Space) -> (Physical Space)
    P = vecs.T * inner_scale[None, :]
    P = phys_scale[:, None] * P

    return P
