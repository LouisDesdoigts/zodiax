import jax.numpy as np
from jax.scipy import stats
from jax.flatten_util import ravel_pytree
from jax import lax, Array

__all__ = [
    "z_score",
    "mv_z_score",
    "loglike",
    "mv_loglike",
    "chi2",
    "chi2r",
    "chi2r_from_z",
    "calc_entropy",
    "ddof",
    "check_symmetric",
    "check_positive_semi_definite",
    "gauss_hessian",
]


def z_score(x: Array, mean: Array, std: Array) -> Array:
    """
    Calculates the z-score $(\mu - x) / \sigma$ of a value given a mean and standard
    deviation.

    Parameters
    ----------
    x : Array
        The value to calculate the z-score of.
    mean : Array
        The mean to use in the z-score calculation.
    std : Array
        The standard deviation to use in the z-score calculation.

    Returns
    -------
    z_score : Array
        The z-score of the value.
    """
    return (mean - x) / std


def mv_z_score(x: Array, mean: Array, cov: Array) -> Array:
    """
    Calculates the squared Mahalanobis distance $(\mu - x)^\top\Sigma^{-1}(\mu - x)$
    of a value given a mean and covariance matrix.

    Parameters
    ----------
    x : Array
        The value to calculate the multivariate z-score of.
    mean : Array
        The mean vector to use in the calculation.
    cov : Array
        The covariance matrix to use in the calculation.

    Returns
    -------
    mv_z_score : Array
        The squared Mahalanobis distance of the value.
    """
    # res = x - mean
    res = mean - x
    return res @ np.linalg.inv(cov) @ res


def loglike(x: Array, mean: Array, std: Array) -> Array:
    """
    Calculates the element-wise log-likelihood under a univariate normal distribution.

    Parameters
    ----------
    x : Array
        The observed values.
    mean : Array
        The mean of the normal distribution.
    std : Array
        The standard deviation of the normal distribution.

    Returns
    -------
    loglike : Array
        The element-wise log-likelihood values.
    """
    return stats.norm.logpdf(x, mean, std)


def mv_loglike(x: Array, mean: Array, cov: Array) -> Array:
    """
    Calculates the log-likelihood under a multivariate normal distribution.

    Parameters
    ----------
    x : Array
        The observed vector.
    mean : Array
        The mean vector of the multivariate normal distribution.
    cov : Array
        The covariance matrix of the multivariate normal distribution.

    Returns
    -------
    mv_loglike : Array
        The multivariate normal log-likelihood value.
    """
    return stats.multivariate_normal.logpdf(x, mean, cov)


def chi2(x: Array, mean: Array, std: Array) -> Array:
    """
    Calculates the chi-squared statistic from values, means, and standard deviations.

    Parameters
    ----------
    x : Array
        The observed values.
    mean : Array
        The expected mean values.
    std : Array
        The standard deviations of the observations.

    Returns
    -------
    chi2 : Array
        The chi-squared statistic.
    """
    return np.square(z_score(x, mean, std)).sum()


def chi2r(x: Array, mean: Array, std: Array, ddof: int) -> Array:
    """
    Calculates the reduced chi-squared statistic.

    Parameters
    ----------
    x : Array
        The observed values.
    mean : Array
        The expected mean values.
    std : Array
        The standard deviations of the observations.
    ddof : int
        The degrees of freedom used to normalise the chi-squared value.

    Returns
    -------
    chi2r : Array
        The reduced chi-squared statistic.
    """
    return chi2(x, mean, std) / ddof


def chi2r_from_z(z_score: Array, ddof: int) -> Array:
    """
    Calculates the reduced chi-squared statistic directly from z-scores.

    Parameters
    ----------
    z_score : Array
        The z-score values.
    ddof : int
        The degrees of freedom used to normalise the chi-squared value.

    Returns
    -------
    chi2r : Array
        The reduced chi-squared statistic.
    """
    return np.square(z_score).sum() / ddof


def ddof(params: Array, data: Array) -> int:
    """
    Calculates the degrees of freedom as the difference between flattened data
    size and flattened parameter size.

    Parameters
    ----------
    params : Array
        The model parameters, provided as a pytree.
    data : Array
        The observed data, provided as a pytree.

    Returns
    -------
    ddof : int
        The number of degrees of freedom, computed as
        ``flatten(data).size - flatten(params).size``.
    """
    X, _ = ravel_pytree(params)
    Y, _ = ravel_pytree(data)
    return Y.size - X.size


def check_symmetric(mat: Array) -> bool:
    """
    Checks if a matrix is symmetric.

    Parameters
    ----------
    mat : Array
        The matrix to check.

    Returns
    -------
    is_symmetric : bool
        ``True`` if the matrix is symmetric within numerical tolerance,
        otherwise ``False``.
    """
    return np.allclose(mat, mat.T)


def check_positive_semi_definite(mat: Array) -> bool:
    """
    Checks if a matrix is positive semi-definite.

    Parameters
    ----------
    mat : Array
        The matrix to check.

    Returns
    -------
    is_positive_semi_definite : bool
        ``True`` if all eigenvalues are non-negative and the matrix contains no
        NaN values, otherwise ``False``.
    """
    return lax.cond(
        np.isnan(mat).any(),
        lambda x: False,
        lambda x: np.all(np.linalg.eigvals(mat) >= 0),
        mat,
    )


def gauss_hessian(J: Array, cov: Array = None) -> Array:
    """
    Calculates the Gauss-Newton Hessian approximation under a Gaussian likelihood model.

    Parameters
    ----------
    J : Array
        The Jacobian matrix.
    cov : Array, optional
        The covariance matrix used to weight the Jacobian. If ``None``, the
        identity weighting is assumed.

    Returns
    -------
    hessian : Array
        The Gauss-Newton Hessian approximation, either ``J.T @ J`` or
        ``J.T @ (cov^{-1} @ J)``.
    """
    # Gauss-Newton hessian approximation under the assumption of a multivariate normal
    if cov is None:
        return J.T @ J
    return J.T @ (np.linalg.inv(cov) @ J)


def calc_entropy(cov: Array) -> Array:
    """
    Calculates the entropy of a covariance matrix.

    Parameters
    ----------
    cov : Array
        The covariance matrix to calculate the entropy of.

    Returns
    -------
    entropy : Array
        The entropy of the covariance matrix.
    """
    sign, logdet = np.linalg.slogdet(cov)
    return 0.5 * (np.log(2 * np.pi * np.e) + (sign * logdet))
