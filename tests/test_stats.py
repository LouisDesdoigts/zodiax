from __future__ import annotations

import jax
import pytest
import jax.scipy as jsp
import zodiax
from jax import config, numpy as np

config.update("jax_debug_nans", True)


PATHS = [
    "param",
    "b.param",
    ["param", "b.param"],
]


def test_calc_entropy():
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])
    entropy = zodiax.stats.calc_entropy(cov)
    assert entropy is not None


def test_z_score_and_chi2_helpers():
    x = np.array([1.0, 3.0])
    mean = np.array([2.0, 1.0])
    std = np.array([1.0, 2.0])

    z = zodiax.stats.z_score(x, mean, std)
    assert np.allclose(z, np.array([1.0, -1.0]))
    assert np.allclose(zodiax.stats.chi2(x, mean, std), 2.0)
    assert np.allclose(zodiax.stats.chi2r(x, mean, std, ddof=2), 1.0)
    assert np.allclose(zodiax.stats.chi2r_from_z(z, ddof=2), 1.0)


def test_mv_scores_and_loglikes():
    x = np.array([0.0, 0.0])
    mean = np.array([1.0, 1.0])
    cov = np.eye(2)

    assert np.allclose(zodiax.stats.mv_z_score(x, mean, cov), 2.0)
    assert np.allclose(
        zodiax.stats.mv_loglike(x, mean, cov),
        jsp.stats.multivariate_normal.logpdf(x, mean, cov),
    )


def test_loglike():
    x = np.array([0.0, 1.0])
    mean = np.array([0.0, 0.0])
    std = np.array([1.0, 2.0])
    assert np.allclose(
        zodiax.stats.loglike(x, mean, std),
        jsp.stats.norm.logpdf(x, mean, std),
    )


def test_ddof_and_matrix_checks():
    params = {"a": np.ones((2,))}
    data = {"x": np.ones((3,)), "y": np.ones((2,))}
    assert zodiax.stats.ddof(params, data) == 3

    symmetric = np.array([[1.0, 0.5], [0.5, 1.0]])
    non_symmetric = np.array([[1.0, 2.0], [0.5, 1.0]])
    psd = np.array([[2.0, 0.0], [0.0, 1.0]])
    not_psd = np.array([[1.0, 0.0], [0.0, -1.0]])

    assert zodiax.stats.check_symmetric(symmetric)
    assert not zodiax.stats.check_symmetric(non_symmetric)
    assert zodiax.stats.check_positive_semi_definite(psd)
    assert not zodiax.stats.check_positive_semi_definite(not_psd)


def test_gauss_hessian():
    jacobian = np.array([[1.0, 2.0], [3.0, 4.0]])
    cov = np.array([[2.0, 0.0], [0.0, 1.0]])

    unweighted = zodiax.stats.gauss_hessian(jacobian)
    weighted = zodiax.stats.gauss_hessian(jacobian, cov)

    assert np.allclose(unweighted, jacobian.T @ jacobian)
    assert np.allclose(weighted, jacobian.T @ (np.linalg.inv(cov) @ jacobian))


@pytest.mark.parametrize(
    "matrix, expected",
    [
        ([[2.0, 0.5], [0.5, 1.0]], (True, True, True)),
        ([[1.0, 1.0], [1.0, 1.0]], (True, True, False)),
        ([[1.0, 2.0], [2.0, 1.0]], (True, False, False)),
        ([[1.0, 2.0], [0.0, 1.0]], (False, False, False)),
        ([[1.0, 2.0, 3.0]], (False, False, False)),
    ],
)
def test_public_matrix_predicates_have_known_results_under_jit(matrix, expected):
    predicates = (
        zodiax.is_symmetric,
        zodiax.is_positive_semidefinite,
        zodiax.is_positive_definite,
    )
    for predicate, result in zip(predicates, expected):
        assert bool(jax.jit(predicate)(np.asarray(matrix))) == result


def test_public_definiteness_tolerances_and_empty_matrices():
    matrix = np.diag(np.asarray([-1e-7, 1.0]))
    assert zodiax.is_positive_semidefinite(matrix)
    assert not zodiax.is_positive_semidefinite(matrix, rtol=0.0, atol=0.0)
    assert not zodiax.check_positive_semi_definite(matrix)
    assert not zodiax.is_positive_definite(np.diag(np.asarray([1e-7, 1.0])))
    assert zodiax.is_positive_definite(
        np.diag(np.asarray([1e-7, 1.0])), rtol=0.0, atol=0.0
    )
    assert not zodiax.is_positive_definite(np.empty((0, 0)))
    assert not zodiax.is_positive_semidefinite(np.empty((0, 0)))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_definiteness_predicates_reject_nonfinite_matrices(value):
    with jax.debug_nans(False):
        matrix = np.asarray([[value, 0.0], [0.0, 1.0]])
        for predicate in (zodiax.is_positive_definite, zodiax.is_positive_semidefinite):
            assert not bool(jax.jit(predicate)(matrix))


def test_definiteness_predicates_avoid_overflow_in_float32():
    matrix = np.diag(np.asarray([2e38, 1e38], dtype=np.float32))
    assert zodiax.is_positive_definite(matrix)
    assert zodiax.is_positive_semidefinite(matrix)
