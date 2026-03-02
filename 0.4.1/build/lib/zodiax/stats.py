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


def z_score(x, mean, std):
    # return (x - mean) / std
    return (mean - x) / std


def mv_z_score(x, mean, cov):
    # res = x - mean
    res = mean - x
    return res @ np.linalg.inv(cov) @ res


def loglike(x, mean, std):
    return stats.norm.logpdf(x, mean, std)


def mv_loglike(x, mean, cov):
    return stats.multivariate_normal.logpdf(x, mean, cov)


def chi2(x, mean, std):
    return np.square(z_score(x, mean, std)).sum()


def chi2r(x, mean, std, ddof):
    return chi2(x, mean, std) / ddof


def chi2r_from_z(z_score, ddof):
    return np.square(z_score).sum() / ddof


def ddof(params, data):
    X, _ = ravel_pytree(params)
    Y, _ = ravel_pytree(data)
    return Y.size - X.size


def check_symmetric(mat):
    """Checks if a matrix is symmetric"""
    return np.allclose(mat, mat.T)


def check_positive_semi_definite(mat):
    """Checks if a matrix is positive semi-definite"""
    return lax.cond(
        np.isnan(mat).any(),
        lambda x: False,
        lambda x: np.all(np.linalg.eigvals(mat) >= 0),
        mat,
    )


def gauss_hessian(J, cov=None):
    # Gauss-Newton hessian approximation under the assumption of a multivariate normal
    if cov is None:
        return J.T @ J
    return J.T @ (np.linalg.inv(cov) @ J)


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
