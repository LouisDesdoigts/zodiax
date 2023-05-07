import zodiax
import jax.numpy as np
import jax.scipy as jsp
from jax import hessian, lax, Array
from typing import Union, List, Any


__all__ = ["poiss_loglike", "chi2_loglike", "fisher_matrix",
    "covariance_matrix", "self_fisher_matrix", "self_covariance_matrix"]


"""
Assumes all pytrees have a .model() function
TODO: Allow *args and **kwargs to be passed to pytree.model()
    -> Actually kind of hard, since we already have * args and ** kwargs for
    the loglike function
"""


Base = lambda: zodiax.base.Base
Params = Union[str, List[str]]


def poiss_loglike(pytree : Base(), data : Array) -> float:
    """
    Poissonian log likelihood of the pytree given the data. Assumes the pytree
    has a .model() function.

    Parameters
    ----------
    pytree : Base
        Pytree with a .model() function.
    data : Array
        Data to compare the model to.
    
    Returns
    -------
    log_likelihood : Array
        Log likelihood of the pytree given the data.
    """
    return jsp.stats.poisson.logpmf(pytree.model(), data).sum()


def chi2_loglike(pytree : Base(), data : Array, noise : float = 1) -> float:
    """
    Chi2 log likelihood of the pytree given the data. Assumes the pytree has a
    .model() function.

    Parameters
    ----------
    pytree : Base
        Pytree with a .model() function.
    data : Array
        Data to compare the model to.
    noise : float = 1
        The 'scale' parameter representing the noise in the data.
    
    Returns
    -------
    log_likelihood : Array
        Log likelihood of the pytree given the data.
    """
    return jsp.stats.chi2.logpdf(pytree.model(), data, scale=noise).sum()


def fisher_matrix(
        pytree           : Base(), 
        parameters       : Params,
        loglike_fn       : callable,
        *loglike_args    : Any,
        shape_dict       : dict = {},
        **loglike_kwargs : Any,
        ) -> Array:
    """
    Calculates the Fisher information matrix of the pytree parameters. The
    `shaped_dict` parameter is used to specify the shape of the differentiated
    vector for specific parameters. For example, if the parameter `param` is
    a 1D array of shape (5,) and we wanted to calculate the Fisher information
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
    *loglike_args : Any
        The args to pass to the log likelihood function.
    shape_dict : dict = {}
        A dictionary specifying the shape of the differentiated vector for
        specific parameters.
    **loglike_kwargs : Any
        The kwargs to pass to the log likelihood function.
    
    Returns
    -------
    fisher_matrix : Array
        The Fisher information matrix of the pytree parameters.
    """
    # Build X vec
    pytree = zodiax.tree.set_array(pytree, parameters)
    shapes, lengths = _shapes_and_lengths(pytree, parameters, shape_dict)
    X = np.zeros(_lengths_to_N(lengths))

    # Build function to calculate FIM and calculate
    @hessian
    def calc_fim(X):
        parametric_pytree = _perturb(X, pytree, parameters, shapes, lengths)
        return loglike_fn(parametric_pytree, *loglike_args, **loglike_kwargs)
    return calc_fim(X)


def covariance_matrix(
        pytree           : Base(), 
        parameters       : Params,
        loglike_fn       : callable,
        *loglike_args    : Any,
        shape_dict       : dict = {},
        **loglike_kwargs : Any
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
    *loglike_args : Any
        The args to pass to the log likelihood function.
    shape_dict : dict = {}
        A dictionary specifying the shape of the differentiated vector for
        specific parameters.
    **loglike_kwargs : Any
        The kwargs to pass to the log likelihood function.
    
    Returns
    -------
    covariance_matrix : Array
        The covariance matrix of the pytree parameters.
    """
    return -np.linalg.inv(fisher_matrix(pytree, parameters, loglike_fn, 
        *loglike_args, shape_dict=shape_dict, **loglike_kwargs))


def self_fisher_matrix(
        pytree           : Base(), 
        parameters       : Params, 
        loglike_fn       : callable,
        *loglike_args    : Any,
        shape_dict       : dict = {},
        **loglike_kwargs : Any
        ) -> Array:
    """
    Calculates the Fisher information matrix of the pytree parameters with
    respect to itself. This can be used to optimise the fisher information of
    the pytree model. Simply calcluates the data from the `.model()` method and
    passes it to the `fisher_matrix` function. The `shaped_dict` parameter is
    used to specify the shape of the differentiated vector for specific
    parameters. For example, if the parameter `param` is a 1D array of shape
    (5,) and we wanted to calculate the covariance matrix of the mean, we can
    pass in `shape_dict={'param': (1,)}`. This will differentiate the log
    likelihood with respect to the mean of the parameters.

    Parameters
    ----------
    pytree : Base
        Pytree with a .model() function.
    parameters : Union[str, list]
        A path or list of paths or list of nested paths.
    loglike_fn : callable
        The log likelihood function to differentiate.
    *loglike_args : Any
        The args to pass to the log likelihood function.
    shape_dict : dict = {}
        A dictionary specifying the shape of the differentiated vector for
        specific parameters.
    **loglike_kwargs : Any
        The kwargs to pass to the log likelihood function.
    
    Returns
    -------
    fisher_matrix : Array
        The Fisher information matrix of the pytree parameters.
    """
    # Build X vec and get data
    data = pytree.model()
    loglike_args = [pytree.model()] + list(loglike_args)
    return fisher_matrix(pytree, parameters, loglike_fn, *loglike_args, 
        shape_dict=shape_dict, **loglike_kwargs)


def self_covariance_matrix(
        pytree           : Base(), 
        parameters       : Params, 
        loglike_fn       : callable,
        *loglike_args    : Any,
        shape_dict       : dict = {},
        **loglike_kwargs : Any
        ) -> Array:
    """
    Calculates the covariance matrix of the pytree parameters with respect to
    itself. This can be used to optimise the fisher information of the pytree
    model. Simply calcluates the data from the `.model()` method and passes it
    to the `covariance_matrix` function. The `shaped_dict` parameter is used to
    specify the shape of the differentiated vector for specific parameters. For
    example, if the parameter `param` is a 1D array of shape (5,) and we wanted
    to calculate the covariance matrix of the mean, we can pass in
    `shape_dict={'param': (1,)}`. This will differentiate the log likelihood
    with respect to the mean of the parameters.

    Parameters
    ----------
    pytree : Base
        Pytree with a .model() function.
    parameters : Union[str, list]
        A path or list of paths or list of nested paths.
    loglike_fn : callable
        The log likelihood function to differentiate.
    *loglike_args : Any
        The args to pass to the log likelihood function.
    shape_dict : dict = {}
        A dictionary specifying the shape of the differentiated vector for
        specific parameters.
    **loglike_kwargs : Any
        The kwargs to pass to the log likelihood function.
    
    Returns
    -------
    covariance_matrix : Array
        The covariance matrix of the pytree parameters.
    """
    data = pytree.model()
    loglike_args = [pytree.model()] + list(loglike_args)
    return covariance_matrix(pytree, parameters, loglike_fn, *loglike_args, 
        shape_dict=shape_dict, **loglike_kwargs)


def _perturb(
        X          : Array, 
        pytree     : Base(), 
        parameters : Params, 
        shapes     : list, 
        lengths    : list
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
    for i, param, shape, length in zip(indexes, parameters, shapes, lengths):
        if length == 1:
            xs.append(X[i+n])
        else:
            xs.append(lax.dynamic_slice(X, (i+n,), (length,)).reshape(shape))
            n += length
    return pytree.add(parameters, xs)


# Functions for calculating lengths and shapes for 'dynamically' generated X
# Vectors for the hessian calculation.
def _lengths_to_N(lengths : list, N : int = 0) -> int:
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


def _shape_to_length(shape : Union[tuple, int], length : int = 0) -> int:
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
    return _shape_to_length(shape[1:], length=shape[0]*length)


def _get_shape(
        pytree     : Base(), 
        parameter  : str, 
        shape_dict : dict = {}) -> Union[tuple, int]:
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
        pytree     : Base(), 
        parameters : Params, 
        shape_dict : dict = {}) -> tuple:
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