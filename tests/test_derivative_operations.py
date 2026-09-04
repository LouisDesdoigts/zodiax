from __future__ import annotations

import equinox as eqx
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
    idx, total = zodiax.derivatives.operations._get_batch_sizes(5, 2)
    assert idx.shape == (2, 3)
    assert int(total) == 6
    assert int(idx[0, 0]) == 0


def test_get_batch_sizes_raises_for_too_many_batches():
    with pytest.raises(ValueError, match="nbatches"):
        zodiax.derivatives.operations._get_batch_sizes(3, 4)


@pytest.mark.parametrize("nbatches", [0, -1])
def test_get_batch_sizes_requires_positive_batches(nbatches):
    with pytest.raises(ValueError, match="positive integer"):
        zodiax.derivatives.operations._get_batch_sizes(3, nbatches)


def test_get_batch_sizes_requires_integer_batches():
    with pytest.raises(TypeError, match="positive integer"):
        zodiax.derivatives.operations._get_batch_sizes(3, 1.5)


@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize("checkpoint", [True, False])
def test_jacobian_batched_matches_unbatched(jit, checkpoint):
    x = _make_x()
    reference = zodiax.derivatives.jacobian(
        _vector_fn,
        x,
        nbatches=1,
        jit=jit,
        checkpoint=checkpoint,
    )
    batched = zodiax.derivatives.jacobian(
        _vector_fn,
        x,
        nbatches=2,
        jit=jit,
        checkpoint=checkpoint,
    )
    expected = np.asarray([[2.0, 1.0, 1.0], [2.0, 0.0, 2.0]])

    assert isinstance(reference, zodiax.Jacobian)
    assert isinstance(batched, zodiax.Jacobian)
    assert np.allclose(reference.matrix, expected)
    assert np.allclose(batched.matrix, reference.matrix)
    assert batched.layout.paths == ("a", "b")
    assert batched.layout.shapes == ((2,), (1,))


@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize("checkpoint", [True, False])
@pytest.mark.parametrize("method", ["fwd-rev", "rev-fwd"])
def test_hessian_batched_matches_unbatched(jit, checkpoint, method):
    x = _make_x()
    reference = zodiax.derivatives.hessian(
        _scalar_fn,
        x,
        nbatches=1,
        jit=jit,
        checkpoint=checkpoint,
        method=method,
    )
    batched = zodiax.derivatives.hessian(
        _scalar_fn,
        x,
        nbatches=2,
        jit=jit,
        checkpoint=checkpoint,
        method=method,
    )
    expected = np.asarray([[2.0, 0.0, 1.0], [0.0, 2.0, 0.0], [1.0, 0.0, 18.0]])

    assert isinstance(reference, zodiax.Hessian)
    assert isinstance(batched, zodiax.Hessian)
    assert np.allclose(reference.matrix, expected)
    assert np.allclose(batched.matrix, reference.matrix)
    assert batched.layout.paths == ("a", "b")
    assert np.allclose(batched.blocks()["a"]["b"], expected[:2, 2:].reshape(2, 1))


def test_hessian_raises_for_too_many_batches():
    x = _make_x()
    with pytest.raises(ValueError, match="nbatches"):
        zodiax.derivatives.hessian(_scalar_fn, x, nbatches=4, jit=False)

    with pytest.raises(ValueError, match="nbatches"):
        zodiax.derivatives.jacobian(_vector_fn, x, nbatches=4, jit=False)


def test_hessian_requires_scalar_output():
    with pytest.raises(ValueError, match="scalar"):
        zodiax.derivatives.hessian(_vector_fn, _make_x(), jit=False)


def test_hessian_rejects_unknown_method():
    with pytest.raises(ValueError, match="fwd-rev.*rev-fwd"):
        zodiax.hessian(_scalar_fn, _make_x(), method="forward", jit=False)


@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize("checkpoint", [True, False])
@pytest.mark.parametrize("weighting", ["cov", "inv_cov"])
def test_gauss_newton_matches_explicit_jacobian(jit, checkpoint, weighting):
    x = _make_x()
    jacobian = np.asarray([[2.0, 1.0, 1.0], [2.0, 0.0, 2.0]])
    cov = np.asarray([[2.0, 0.5], [0.5, 1.5]])
    expected = jacobian.T @ np.linalg.solve(cov, jacobian)
    kwargs = {weighting: cov if weighting == "cov" else np.linalg.inv(cov)}

    matrix = zodiax.gauss_newton(
        _vector_fn,
        x,
        nbatches=2,
        jit=jit,
        checkpoint=checkpoint,
        **kwargs,
    )

    assert isinstance(matrix, zodiax.GaussNewton)
    assert matrix.layout.paths == ("a", "b")
    assert np.allclose(matrix.matrix, expected)
    assert np.allclose(matrix.blocks()["a"]["b"], expected[:2, 2:])


def test_gauss_newton_identity_weighting_and_scalar_residual():
    x = _make_x()
    identity = zodiax.gauss_newton(_vector_fn, x, nbatches=2, jit=False)
    scalar = zodiax.gauss_newton(
        lambda values: values["a"][0] ** 2,
        x,
        nbatches=2,
        jit=False,
    )
    jacobian = np.asarray([[2.0, 1.0, 1.0], [2.0, 0.0, 2.0]])
    expected_scalar = np.asarray([[4.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    assert np.allclose(identity.matrix, jacobian.T @ jacobian)
    assert np.allclose(scalar.matrix, expected_scalar)


def test_gauss_newton_omits_nonlinear_residual_correction():
    x = {"value": np.asarray(2.0)}
    residual_fn = lambda values: values["value"] ** 2 - 1.0
    loss_fn = lambda values: 0.5 * residual_fn(values) ** 2

    approximation = zodiax.gauss_newton(residual_fn, x, jit=False)
    exact = zodiax.hessian(loss_fn, x, jit=False)

    assert np.allclose(approximation.matrix, np.asarray([[16.0]]))
    assert np.allclose(exact.matrix, np.asarray([[22.0]]))


def test_gauss_newton_validates_weighting():
    x = _make_x()
    cov = np.eye(2)

    with pytest.raises(ValueError, match="either cov or inv_cov"):
        zodiax.gauss_newton(
            _vector_fn,
            x,
            cov=cov,
            inv_cov=cov,
            jit=False,
        )
    with pytest.raises(ValueError, match="cov must have shape"):
        zodiax.gauss_newton(_vector_fn, x, cov=np.eye(3), jit=False)
    with pytest.raises(TypeError, match="inv_cov.*floating"):
        zodiax.gauss_newton(
            _vector_fn,
            x,
            inv_cov=np.eye(2, dtype=np.int32),
            jit=False,
        )


def test_gauss_newton_requires_floating_residuals():
    with pytest.raises(TypeError, match="floating residual"):
        zodiax.gauss_newton(
            lambda values: np.asarray([values["a"].size], dtype=np.int32),
            _make_x(),
            jit=False,
        )


def test_gauss_newton_result_crosses_outer_jit_boundary():
    calculate = eqx.filter_jit(
        lambda x, cov: zodiax.gauss_newton(
            _vector_fn,
            x,
            cov=cov,
            nbatches=2,
        )
    )
    matrix = calculate(_make_x(), np.eye(2))

    assert isinstance(matrix, zodiax.GaussNewton)
    assert matrix.matrix.shape == (3, 3)
    assert matrix.layout.paths == ("a", "b")


def test_jacobian_result_crosses_outer_jit_boundary():
    calculate = eqx.filter_jit(
        lambda x: zodiax.derivatives.jacobian(_vector_fn, x, nbatches=2)
    )
    jacobian = calculate(_make_x())

    assert isinstance(jacobian, zodiax.Jacobian)
    assert jacobian.matrix.shape == (2, 3)
    assert jacobian.layout.paths == ("a", "b")


def test_derivatives_reject_non_floating_parameter_leaves():
    values = {
        "coefficient": np.asarray([2.0, 3.0]),
        "power": np.asarray(2, dtype=np.int32),
    }

    with pytest.raises(TypeError, match="power.*floating"):
        zodiax.jacobian(
            lambda model: model["coefficient"] ** model["power"],
            values,
            jit=False,
        )


def test_hessian_to_pytree_structure_and_values():
    x = _make_x()
    matrix = np.arange(9.0).reshape(3, 3)
    h_flat = (matrix + matrix.T) / 2
    layout = zodiax.TreeLayout.from_tree(x)

    with pytest.warns(DeprecationWarning, match="Hessian.blocks"):
        h_tree = zodiax.derivatives.hessian_to_pytree(h_flat, x)
    with pytest.warns(DeprecationWarning, match="Hessian.blocks"):
        typed_tree = zodiax.derivatives.hessian_to_pytree(
            zodiax.Hessian(h_flat, layout),
            x,
        )

    assert h_tree["a"]["a"].shape == (2, 2)
    assert h_tree["a"]["b"].shape == (2, 1)
    assert h_tree["b"]["a"].shape == (1, 2)
    assert h_tree["b"]["b"].shape == (1, 1)

    assert np.allclose(h_tree["a"]["a"], h_flat[:2, :2])
    assert np.allclose(h_tree["a"]["b"], h_flat[:2, 2:].reshape(2, 1))
    assert np.allclose(h_tree["b"]["a"], h_flat[2:, :2].reshape(1, 2))
    assert np.allclose(h_tree["b"]["b"], h_flat[2:, 2:].reshape(1, 1))
    assert np.allclose(typed_tree["a"]["b"], h_tree["a"]["b"])


def test_hessian_to_pytree_shape_mismatch_raises():
    x = _make_x()
    with pytest.warns(DeprecationWarning, match="Hessian.blocks"):
        with pytest.raises(ValueError, match="H has shape"):
            zodiax.derivatives.hessian_to_pytree(np.zeros((2, 2)), x)

    layout = zodiax.TreeLayout.from_tree(x)
    incompatible = {"different": np.ones(3)}
    with pytest.warns(DeprecationWarning, match="Hessian.blocks"):
        with pytest.raises(ValueError, match="not compatible"):
            zodiax.derivatives.hessian_to_pytree(
                zodiax.Hessian(np.zeros((3, 3)), layout),
                incompatible,
            )


def test_hessian_to_pytree_supports_python_float_leaves():
    x = {"a": 1.0, "b": 2.0}
    hessian = np.asarray([[1.0, 2.0], [2.0, 3.0]])
    with pytest.warns(DeprecationWarning, match="Hessian.blocks"):
        tree = zodiax.derivatives.hessian_to_pytree(hessian, x)

    assert tree["a"]["a"].shape == ()
    assert tree["a"]["b"].shape == ()
    assert np.allclose(tree["b"]["a"], 2.0)
