from __future__ import annotations
import zodiax

from jax import config, Array, scipy as jsp, numpy as np

config.update("jax_debug_nans", True)


# Paths
paths = [
    "param",
    "b.param",
    ["param", "b.param"],
]


def poiss_loglike(pytree, data: Array) -> float:
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


def test_all_fisher_matrices(create_base):
    pytree = create_base()
    data = pytree.model()
    loglike_fn = poiss_loglike
    shape_dict = {"param": (1,)}

    for param in paths:

        hess = zodiax.bayes.hessian(
            pytree, param, loglike_fn, data, shape_dict=shape_dict
        )

        fisher = zodiax.bayes.fisher_matrix(
            pytree, param, loglike_fn, data, shape_dict=shape_dict
        )

        cov = zodiax.bayes.covariance_matrix(
            pytree, param, loglike_fn, data, shape_dict=shape_dict
        )

        if not np.allclose(hess, -fisher):
            raise ValueError("Negative Hessian and Fisher matrix are not equal.")

        if not np.allclose(np.linalg.inv(fisher), cov):
            raise ValueError(
                "Inverse Fisher matrix and covariance matrix are not equal."
            )


def test_calc_entropy(create_base):
    pytree = create_base()
    data = pytree.model()
    loglike_fn = poiss_loglike
    shape_dict = {"param": (1,)}
    for param in paths:
        cov = zodiax.bayes.covariance_matrix(
            pytree, param, loglike_fn, data, shape_dict=shape_dict
        )
        zodiax.bayes.calc_entropy(cov)
