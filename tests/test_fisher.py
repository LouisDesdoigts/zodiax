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

    hess = zodiax.fisher.hessian_old(
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


def test_hessian_single_parameter_container_path(create_base):
    pytree = create_base()
    data = pytree.model()
    out = zodiax.fisher.hessian_old(
        pytree,
        ("param",),
        poiss_loglike,
        data,
        shape_dict={"param": (1,)},
        save_memory=False,
    )
    assert out.shape == (1, 1)


class _VecBase(zodiax.base.Base):
    vec: Array

    def __init__(self, vec):
        self.vec = vec


def test_perturb_handles_vector_slices():
    pytree = _VecBase(np.array([1.0, 2.0]))
    out = zodiax.fisher._perturb(
        np.array([0.5, -1.5]),
        pytree,
        ["vec"],
        [(2,)],
        [2],
    )
    assert np.allclose(out.get("vec"), np.array([1.5, 0.5]))


def test_shape_to_length_singleton_tuple():
    assert zodiax.fisher._shape_to_length((3,)) == 3


def test_shape_to_length_recursive_branch():
    out = zodiax.fisher._shape_to_length((2, 3), length=1)
    assert out == 3
