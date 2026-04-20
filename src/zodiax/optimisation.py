import optax
import jax.numpy as np
import jax.tree as jtu
import equinox as eqx
from jax import Array
from typing import Union, Any
from .base import _unpack

__all__ = [
    "debug_nan_check",
    "zero_nan_check",
    "delay",
    "map_optimisers",
    "decompose",
    "eigen_projection",
]


PyTree = Union[dict, list, tuple, eqx.Module]
Params = Union[str, list[str], tuple[str]]
Values = Union[Any, list[Any], tuple[Any]]


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
    bool_tree = jtu.map(lambda x: np.isnan(x).any(), grads)
    vals = np.array(jtu.flatten(bool_tree)[0])
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
    return jtu.map(lambda x: np.where(np.isnan(x), 0.0, x), grads)


def map_optimisers(params: dict, optimisers: dict, strict: bool = False) -> tuple:
    """
    Maps optimiser from a dictionary of optax optimisers to a dictionary of parameters.

    TODO: Develop docs more
    """

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
            optimisers[key] = optax.set_to_zero()

    # Check for keys in optimisers that aren't in params, and set to zero
    for key in optimisers.keys():
        if key not in params.keys():
            params[key] = 0.0

    param_spec = {param: param for param in optimisers.keys()}
    optim = optax.multi_transform(optimisers, param_spec)
    state = optim.init(params)
    return optim, state


def delay(lr: float, start: int, length: int = 1) -> optax.Schedule:
    """
    Delays the learning rate by starting at 0 and linearly increasing to the specified
    learning rate over a specified number of steps.
    """
    return optax.linear_schedule(0.0, lr, length, start)


def decompose(
    matrix: Array, hermitian: bool = True, normalise: bool = False
) -> tuple[Array, Array]:
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


def eigen_projection(fmat: Array = None, cov: Array = None) -> Array:
    """
    Projects the parameter space into the an orthonormal basis

    TODO: develop docs more
    """
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
