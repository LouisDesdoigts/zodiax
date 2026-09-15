from __future__ import annotations

import equinox as eqx
import jax
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


@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize("checkpoint", [True, False])
@pytest.mark.parametrize("std", [2.0, np.asarray([2.0, 4.0])])
def test_gauss_newton_uncorrelated_noise_fast_path(jit, checkpoint, std):
    x = _make_x()
    jacobian = np.asarray([[2.0, 1.0, 1.0], [2.0, 0.0, 2.0]])
    standard_deviation = np.broadcast_to(std, (2,))
    expected = (jacobian / standard_deviation[:, None]).T @ (
        jacobian / standard_deviation[:, None]
    )

    matrix = zodiax.gauss_newton(
        _vector_fn,
        x,
        std=std,
        nbatches=2,
        jit=jit,
        checkpoint=checkpoint,
    )

    assert np.allclose(matrix.matrix, expected)


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

    with pytest.raises(ValueError, match="only one of std, cov, or inv_cov"):
        zodiax.gauss_newton(
            _vector_fn,
            x,
            cov=cov,
            inv_cov=cov,
            jit=False,
        )
    with pytest.raises(ValueError, match="only one of std, cov, or inv_cov"):
        zodiax.gauss_newton(
            _vector_fn,
            x,
            std=1.0,
            cov=cov,
            jit=False,
        )
    with pytest.raises(ValueError, match="cov must have shape"):
        zodiax.gauss_newton(_vector_fn, x, cov=np.eye(3), jit=False)
    with pytest.raises(ValueError, match="broadcast to output shape"):
        zodiax.gauss_newton(_vector_fn, x, std=np.ones(3), jit=False)


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

    independent = eqx.filter_jit(
        lambda x, std: zodiax.gauss_newton(
            _vector_fn,
            x,
            std=std,
            nbatches=2,
        )
    )(_make_x(), np.asarray([2.0, 4.0]))
    assert np.all(np.isfinite(independent.matrix))


def test_jacobian_result_crosses_outer_jit_boundary():
    calculate = eqx.filter_jit(
        lambda x: zodiax.derivatives.jacobian(_vector_fn, x, nbatches=2)
    )
    jacobian = calculate(_make_x())

    assert isinstance(jacobian, zodiax.Jacobian)
    assert jacobian.matrix.shape == (2, 3)
    assert jacobian.layout.paths == ("a", "b")


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


@pytest.mark.parametrize(
    "operation", [zodiax.jacobian, zodiax.hessian, zodiax.gauss_newton]
)
@pytest.mark.parametrize("nbatches", [0, -1, 1.5, True, 4])
def test_derivative_batch_configuration_is_structural(operation, nbatches):
    with pytest.raises((TypeError, ValueError), match="positive integer|nbatches"):
        operation(_scalar_fn, _make_x(), nbatches=nbatches)


@pytest.mark.parametrize(
    "operation", [zodiax.jacobian, zodiax.hessian, zodiax.gauss_newton]
)
@pytest.mark.parametrize("dtype", [int, bool, complex])
def test_trainable_trees_never_filter_or_coerce_nonfloating_leaves(operation, dtype):
    values = {
        "coefficient": np.asarray([2.0, 3.0]),
        "configuration": np.asarray(1, dtype=dtype),
    }
    with pytest.raises(TypeError, match="parameter leaf must be floating"):
        operation(lambda p: p["coefficient"].sum(), values)


@pytest.mark.parametrize("weighting", ["std", "cov", "inv_cov"])
def test_real_noise_data_are_converted_to_default_float(weighting):
    weights = {"std": 2, "cov": [[4, 0], [0, 4]], "inv_cov": [[1, 0], [0, 1]]}
    expected = np.asarray([[2.0, 1.0, 1.0], [2.0, 0.0, 2.0]])
    expected = expected.T @ expected
    if weighting != "inv_cov":
        expected = expected / 4
    result = zodiax.gauss_newton(
        _vector_fn, _make_x(), **{weighting: weights[weighting]}
    )
    assert np.allclose(result.matrix, expected)


@pytest.mark.parametrize("scale", [1e-20, 1e20])
@pytest.mark.parametrize("jit", [False, True])
def test_standard_deviation_weighting_avoids_variance_overflow(scale, jit, x64_context):
    # Exercise float32 even when the wider scientific suite uses x64.
    with x64_context(False):
        noise = np.asarray(scale, dtype=float)
        parameters = {"x": np.asarray([1.0, 2.0])}
        residuals = lambda p: noise * p["x"]
        result = zodiax.gauss_newton(
            residuals, parameters, std=noise, nbatches=2, jit=jit
        )
        fisher = zodiax.jacobian(residuals, parameters, jit=jit).fisher(std=noise)
        assert np.allclose(result.matrix, np.eye(2), rtol=2e-6)
        assert np.allclose(result.matrix, fisher.matrix, rtol=2e-6)


def test_multidimensional_residual_noise_broadcasts_before_flattening():
    parameters = np.asarray([1.0, 2.0, 3.0])
    noise = np.asarray([[2.0], [3.0]])

    def residuals(p):
        return np.asarray([[p[0] * p[1], p[2]], [p[0] ** 2, p[1] + p[2]]])

    jacobian = jax.jacfwd(residuals)(parameters)
    whitened = (jacobian / noise[..., None]).reshape(4, 3)
    result = zodiax.gauss_newton(residuals, parameters, std=noise, nbatches=2)
    assert np.allclose(result.matrix, whitened.T @ whitened)


def test_gauss_newton_supports_outer_derivatives_of_point_and_noise():
    def calculate(point, noise):
        parameters = {"x": point}
        return zodiax.gauss_newton(lambda p: p["x"] ** 2, parameters, std=noise).matrix[
            0, 0
        ]

    point, noise = np.asarray(3.0), np.asarray(2.0)
    point_gradient, noise_gradient = jax.grad(calculate, argnums=(0, 1))(point, noise)
    assert np.allclose(point_gradient, 8 * point / noise**2)
    assert np.allclose(noise_gradient, -8 * point**2 / noise**3)


def test_hessian_legacy_conversion_preserves_sequence_containers_under_jit():
    parameters = (np.asarray(1.0), [np.asarray([2.0, 3.0])])
    matrix = np.arange(9.0).reshape(3, 3)
    with pytest.warns(DeprecationWarning):
        result = jax.jit(lambda m: zodiax.hessian_to_pytree(m, parameters))(matrix)
    assert isinstance(result, tuple)
    assert isinstance(result[1], list)
    assert np.array_equal(result[0][1][0], matrix[0, 1:])
    assert np.array_equal(result[1][0][1][0], matrix[1:, 1:])
