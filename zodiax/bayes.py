import zodiax
import jax
from jax import lax, Array, numpy as np
from typing import Union, List, Any


__all__ = [
    "calc_entropy",
    "hessian",
    "fisher_matrix",
    "covariance_matrix",
]


"""
Assumes all pytrees have a .model() function
TODO: Allow *args and **kwargs to be passed to pytree.model()
    -> Actually kind of hard, since we already have * args and ** kwargs for
    the loglike function
"""


def Base():
    return zodiax.base.Base


Params = Union[str, List[str]]


def calc_entropy(cov_matrix: Array) -> Array:
    """
    Calculates the entropy of a covariance matrix.

    Parameters
    ----------
    cov_matrix : Array
        The covariance matrix to calculate the entropy of.

    Returns
    -------
    entropy : Array
        The entropy of the covariance matrix.
    """
    sign, logdet = np.linalg.slogdet(cov_matrix)
    return 0.5 * (np.log(2 * np.pi * np.e) + (sign * logdet))


def hessian(
    pytree: Base(),
    parameters: Params,
    loglike_fn: callable,
    *loglike_args,
    shape_dict: dict = {},
    save_memory: bool = False,
    **loglike_kwargs,
) -> Array:
    """
    Calculates the Hessian of the log likelihood function with respect to the
    parameters of the pytree. It is evaluated at the current values of the
    parameters as listed in the pytree.

    Parameters
    ----------
    pytree : Base
        Pytree holding the parameters values.
    parameters : Union[str, list]
        Names of parameters to be used in Hessian calculation.
    loglike_fn : callable
        The log likelihood function to differentiate.
    shape_dict : dict = {}
        A dictionary specifying the shape of the differentiated vector for
        specific parameters.
    save_memory : bool = False
        If True, the Hessian is calculated column by column to save memory.
    *loglike_args : Any
        The args to pass to the log likelihood function.
    **loglike_kwargs : Any
        The kwargs to pass to the log likelihood function.

    Returns
    -------
    hessian : Array
        The Hessian of the log likelihood function with respect to the
        parameters of the pytree.
    """

    # If only one parameter is passed, make it a list
    if len(parameters) == 1:
        parameters = [parameters]

    # Build empty vector to perturb
    pytree = zodiax.tree.set_array(pytree, parameters)
    shapes, lengths = _shapes_and_lengths(pytree, parameters, shape_dict)
    X = np.zeros(_lengths_to_N(lengths))

    # Build perturbation function to differentiate
    def loglike_fn_vec(X):
        parametric_pytree = _perturb(X, pytree, parameters, shapes, lengths)
        return loglike_fn(parametric_pytree, *loglike_args, **loglike_kwargs)

    # optional column by column hessian calculation for RAM mercy
    if save_memory:
        return _hessian_col_by_col(loglike_fn_vec, X)

    elif not save_memory:
        return jax.hessian(loglike_fn_vec)(X)


def fisher_matrix(
    pytree: Base(),
    parameters: Params,
    loglike_fn: callable,
    *loglike_args,
    shape_dict: dict = {},
    save_memory: bool = False,
    **loglike_kwargs,
) -> Array:
    """
    Calculates the Fisher information matrix of the log likelihood function with
    respect to the parameters of the pytree. It is evaluated at the current values
    of the parameters as listed in the pytree. Simply returns the negative Hessian.
    """

    return -hessian(
        pytree,
        parameters,
        loglike_fn,
        shape_dict=shape_dict,
        save_memory=save_memory,
        *loglike_args,
        **loglike_kwargs,
    )


def covariance_matrix(
    pytree: Base(),
    parameters: Params,
    loglike_fn: callable,
    *loglike_args: Any,
    shape_dict: dict = {},
    save_memory: bool = False,
    **loglike_kwargs: Any,
) -> Array:
    """
    Calculates the covariance matrix of the pytree parameters. The
    `shaped_dict` parameter is used to specify the shape of the differentiated
    vector for specific parameters. For example, if the parameter `param` is
    a 1D array of shape (5,) and we wanted to calculate the covariance matrix
    of the mean, we can pass in `shape_dict={'param': (1,)}`. This will
    differentiate the log likelihood with respect to the mean of the parameters.

    Parameters
    ----------
    pytree : Base
        Pytree with a .model() function.
    parameters : Union[str, list]
        A path or list of paths or list of nested paths.
    loglike_fn : callable
        The log likelihood function to differentiate.
    shape_dict : dict = {}
        A dictionary specifying the shape of the differentiated vector for
        specific parameters.
    save_memory : bool = False
        If True, the Hessian is calculated column by column to save memory.
        This is slightly slower.
    *loglike_args : Any
        The args to pass to the log likelihood function.
    **loglike_kwargs : Any
        The kwargs to pass to the log likelihood function.

    Returns
    -------
    covariance_matrix : Array
        The covariance matrix of the pytree parameters.
    """
    return -np.linalg.inv(
        fisher_matrix(
            pytree,
            parameters,
            loglike_fn,
            shape_dict=shape_dict,
            save_memory=save_memory,
            *loglike_args,
            **loglike_kwargs,
        )
    )


def _perturb(
    X: Array, pytree: Base(), parameters: Params, shapes: list, lengths: list
) -> Base():
    """
    Perturbs the pytree parameters by the values in X, automatically setting
    up the correct sizes based on each parameter.

    Parameters
    ----------
    X : Array
        The vector to perturb the parameters by.
    pytree : Base
        Pytree with a .model() function.
    parameters : Union[str, list]
        A path or list of paths or list of nested paths.
    shapes : list
        A list of the shapes of the parameters.
    lengths : list
        A list of the lengths of the parameters.

    Returns
    -------
    pytree : Base
        The perturbed pytree.
    """
    # Improve with lax.scan or lax.carry?
    n, xs = 0, []
    if isinstance(parameters, str):
        parameters = [parameters]
    indexes = range(len(parameters))
    for i, shape, length in zip(indexes, shapes, lengths):
        if length == 1:
            xs.append(X[i + n])
        else:
            xs.append(lax.dynamic_slice(X, (i + n,), (length,)).reshape(shape))
            n += length
    return pytree.add(parameters, xs)


def _hessian_col_by_col(f: callable, X: Array) -> Array:
    """
    Calculates the Hessian of a function f at x, column by column.
    This may take longer than the full Hessian calculation, but is more
    memory efficient.

    Parameters
    ----------
    f : callable
        The function to differentiate.
    X : Array
        The point to calculate the Hessian at.

    Returns
    -------
    hessian : Array
        The Hessian of the function at x.
    """

    # Jit the sub-function here since it is called many times
    _, hvp = jax.linearize(jax.grad(f), X)
    hvp = jax.jit(hvp)

    # Build and stack
    basis = np.eye(X.size).reshape(-1, *X.shape)
    return np.stack([hvp(e) for e in basis]).reshape(X.shape + X.shape)


# Functions for calculating lengths and shapes for 'dynamically' generated X
# Vectors for the hessian calculation.
def _lengths_to_N(lengths: list, N: int = 0) -> int:
    """
    Converts an list of shapes into a single length.

    Parameters
    ----------
    lengths : list
        A list of the lengths of the parameters.
    N : int = 0
        The current length.
    """
    if len(lengths) == 1:
        return lengths[0] + N
    return _lengths_to_N(lengths[1:], N=lengths[0] + N)


def _shape_to_length(shape: Union[tuple, int], length: int = 0) -> int:
    """
    Converts a shape into a single length.

    Parameters
    ----------
    shape : Union[tuple, int]
        The shape of the parameter.
    length : int = 0
        The current length.

    Returns
    -------
    length : int
        The length of the parameter.
    """
    if isinstance(shape, int):
        return shape
    elif len(shape) == 1:
        return shape[0]
    return _shape_to_length(shape[1:], length=shape[0] * length)


def _get_shape(
    pytree: Base(), parameter: str, shape_dict: dict = {}
) -> Union[tuple, int]:
    """
    Gets the shape of a parameter in the pytree. If the parameter is in the
    shape_dict, then the shape is taken from there. Otherwise, the shape is
    taken from the pytree.

    Parameters
    ----------
    pytree : Base
        Pytree with a .model() function.
    parameter : str
        The path to the parameter.
    shape_dict : dict = {}
        A dictionary specifying the shape of the differentiated vector for
        specific parameters.

    Returns
    -------
    shape : Union[tuple, int]
        The shape of the parameter.
    """
    shape = pytree.get(parameter).shape
    if parameter in shape_dict:
        return shape_dict[parameter]
    return 1 if shape == () else shape


def _shapes_and_lengths(
    pytree: Base(), parameters: Params, shape_dict: dict = {}
) -> tuple:
    """
    Calculates the shapes and lenths of the parameters in the pytree. If the
    parameter is in the shape_dict, then the shape is taken from there.
    Otherwise, the shape is taken from the pytree.

    Parameters
    ----------
    pytree : Base
        Pytree with a .model() function.
    parameters : Union[str, list]
        A path or list of paths or list of nested paths.
    shape_dict : dict = {}
        A dictionary specifying the shape of the differentiated vector for

    Returns
    -------
    shapes : list
        A list of the shapes of the parameters.
    """
    if isinstance(parameters, str):
        parameters = [parameters]
    shapes = [_get_shape(pytree, p, shape_dict) for p in parameters]
    lengths = [_shape_to_length(shape) for shape in shapes]
    return shapes, lengths
