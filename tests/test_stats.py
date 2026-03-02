from __future__ import annotations
import zodiax
import jax.scipy as jsp
from jax import config

config.update("jax_debug_nans", True)


# Paths
paths = [
    "param",
    "b.param",
    ["param", "b.param"],
]


def poiss_loglike(pytree, data):
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


def test_calc_entropy(create_base):
    """
    tests the calc_entropy function
    """
    pytree = create_base()
    data = pytree.model()
    loglike_fn = poiss_loglike
    shape_dict = {"param": (1,)}
    for param in paths:
        cov = zodiax.fisher.covariance_matrix(
            pytree, param, loglike_fn, data, shape_dict=shape_dict
        )
        zodiax.stats.calc_entropy(cov)
