from __future__ import annotations
import pytest
import zodiax

from jax import config, Array, scipy as jsp, numpy as np

config.update("jax_debug_nans", True)


PATHS = [
    "param",
    "b.param",
    ["param", "b.param"],
]


def poiss_loglike(pytree, data: Array) -> float:
    return jsp.stats.poisson.logpmf(pytree.model(), data).sum()


@pytest.mark.parametrize("param", PATHS)
@pytest.mark.parametrize("save_memory", [True, False])
def test_all_fisher_matrices(create_base, param, save_memory):
    pytree = create_base()
    data = pytree.model()
    shape_dict = {"param": (1,)}

    hess = zodiax.fisher.hessian(
        pytree,
        param,
        poiss_loglike,
        data,
        shape_dict=shape_dict,
        save_memory=save_memory,
    )
    fisher = zodiax.fisher.fisher_matrix(
        pytree,
        param,
        poiss_loglike,
        data,
        shape_dict=shape_dict,
        save_memory=save_memory,
    )
    cov = zodiax.fisher.covariance_matrix(
        pytree,
        param,
        poiss_loglike,
        data,
        shape_dict=shape_dict,
        save_memory=save_memory,
    )

    assert np.allclose(hess, -fisher)
    assert np.allclose(np.linalg.inv(fisher), cov)
