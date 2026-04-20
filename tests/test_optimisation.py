from __future__ import annotations
import optax
import pytest
import zodiax
from jax import config, numpy as np

config.update("jax_debug_nans", True)


def test_debug_nan_check(create_base):
    pytree = create_base()
    out = zodiax.optimisation.debug_nan_check(pytree)
    assert out is pytree


def test_zero_nan_check(create_base):
    pytree = create_base(param=np.nan)
    out = zodiax.optimisation.zero_nan_check(pytree)
    assert float(out.get("param")) == 0.0


def test_delay_schedule():
    lr = 0.1
    start = 2
    length = 4
    schedule = zodiax.optimisation.delay(lr, start, length)
    assert np.allclose(schedule(0), 0.0)
    assert np.allclose(schedule(start), 0.0)
    assert np.allclose(schedule(start + length), lr)


def test_map_optimisers_non_strict():
    params = {"a": 1.0, "b": 2.0}
    optimisers = {"a": optax.sgd(1e-2), "c": optax.adam(1e-2)}
    optim, state = zodiax.optimisation.map_optimisers(
        params,
        optimisers,
        strict=False,
    )
    assert hasattr(optim, "update")
    assert state is not None


def test_map_optimisers_strict_mismatch_raises():
    params = {"a": 1.0}
    optimisers = {"b": optax.sgd(1e-2)}
    with pytest.raises(ValueError):
        zodiax.optimisation.map_optimisers(params, optimisers, strict=True)


@pytest.mark.parametrize("hermitian", [True, False])
def test_decompose_outputs_sorted(hermitian):
    matrix = np.array([[2.0, 0.5], [0.5, 1.0]])
    vals, vecs = zodiax.optimisation.decompose(matrix, hermitian=hermitian)
    assert vals.shape == (2,)
    assert vecs.shape == (2, 2)
    assert vals[0] >= vals[1]


def test_decompose_normalise():
    matrix = np.array([[3.0, 0.0], [0.0, 1.0]])
    vals, _ = zodiax.optimisation.decompose(matrix, normalise=True)
    assert np.allclose(vals[0], 1.0)


def test_eigen_projection_requires_input():
    with pytest.raises(ValueError):
        zodiax.optimisation.eigen_projection()


@pytest.mark.parametrize("key", ["fmat", "cov"])
def test_eigen_projection_output_shape(key):
    matrix = np.array([[2.0, 0.25], [0.25, 1.0]])
    kwargs = {key: matrix}
    projection = zodiax.optimisation.eigen_projection(**kwargs)
    assert projection.shape == matrix.shape
    assert np.isfinite(projection).all()
