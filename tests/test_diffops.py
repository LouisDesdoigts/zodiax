from __future__ import annotations

import pytest
import zodiax

from jax import numpy as np


def _make_x():
    return {"a": np.array([1.0, 2.0]), "b": np.array([3.0])}


def _vector_fn(x):
    return np.array(
        [
            x["a"][0] * x["a"][1] + x["b"][0],
            x["a"][0] ** 2 + 2.0 * x["b"][0],
        ]
    )


def _scalar_fn(x):
    return np.sum(x["a"] ** 2) + x["a"][0] * x["b"][0] + x["b"][0] ** 3


def test_get_batch_sizes_padding_and_shape():
    idx, total = zodiax.diffops._get_batch_sizes(5, 2)
    assert idx.shape == (2, 3)
    assert int(total) == 6
    assert int(idx[0, 0]) == 0


def test_get_batch_sizes_raises_for_too_many_batches():
    with pytest.raises(ValueError, match="nbatches"):
        zodiax.diffops._get_batch_sizes(3, 4)


@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize("checkpoint", [True, False])
def test_jacobian_batched_matches_unbatched(jit, checkpoint):
    x = _make_x()
    j_ref, unflatten_ref = zodiax.diffops.jacobian(
        _vector_fn,
        x,
        nbatches=1,
        jit=jit,
        checkpoint=checkpoint,
    )
    j_batched, unflatten_batched = zodiax.diffops.jacobian(
        _vector_fn,
        x,
        nbatches=2,
        jit=jit,
        checkpoint=checkpoint,
    )

    assert np.allclose(j_batched, j_ref)
    rebuilt = unflatten_batched(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(rebuilt["a"], np.array([1.0, 2.0]))
    assert np.allclose(rebuilt["b"], np.array([3.0]))
    assert np.allclose(
        unflatten_ref(np.array([1.0, 2.0, 3.0]))["a"],
        rebuilt["a"],
    )


@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize("checkpoint", [True, False])
def test_hessian_batched_matches_unbatched(jit, checkpoint):
    x = _make_x()
    h_ref, _ = zodiax.diffops.hessian(
        _scalar_fn,
        x,
        nbatches=1,
        jit=jit,
        checkpoint=checkpoint,
    )
    h_batched, _ = zodiax.diffops.hessian(
        _scalar_fn,
        x,
        nbatches=2,
        jit=jit,
        checkpoint=checkpoint,
    )

    assert np.allclose(h_batched, h_ref)


def test_hessian_raises_for_too_many_batches():
    x = _make_x()
    with pytest.raises(ValueError, match="nbatches"):
        zodiax.diffops.hessian(_scalar_fn, x, nbatches=4, jit=False)


def test_hessian_to_pytree_structure_and_values():
    x = _make_x()
    h_flat = np.arange(9.0).reshape(3, 3)

    h_tree = zodiax.diffops.hessian_to_pytree(h_flat, x)

    assert h_tree["a"]["a"].shape == (2, 2)
    assert h_tree["a"]["b"].shape == (2, 1)
    assert h_tree["b"]["a"].shape == (1, 2)
    assert h_tree["b"]["b"].shape == (1, 1)

    assert np.allclose(h_tree["a"]["a"], h_flat[:2, :2])
    assert np.allclose(h_tree["a"]["b"], h_flat[:2, 2:].reshape(2, 1))
    assert np.allclose(h_tree["b"]["a"], h_flat[2:, :2].reshape(1, 2))
    assert np.allclose(h_tree["b"]["b"], h_flat[2:, 2:].reshape(1, 1))


def test_hessian_to_pytree_shape_mismatch_raises():
    x = _make_x()
    with pytest.raises(ValueError, match="H has shape"):
        zodiax.diffops.hessian_to_pytree(np.zeros((2, 2)), x)
