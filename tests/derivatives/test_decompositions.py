"""Public decomposition contracts from derivatives/overview.md.

Reference identities use the represented matrix and explicit coordinate maps.
"""

from __future__ import annotations

from io import BytesIO

import equinox as eqx
import jax
import pytest
import zodiax as zdx
from jax import numpy as np
from jax.flatten_util import ravel_pytree


class Calibration(zdx.Module):
    offset: jax.Array
    weights: jax.Array
    name: str = eqx.field(static=True, default="calibration")


def _fisher():
    layout = zdx.TreeLayout(("a", "b"), ((2,), ()))
    matrix = np.asarray(
        [
            [9.0, 1.5, -0.5],
            [1.5, 4.0, 0.75],
            [-0.5, 0.75, 2.0],
        ]
    )
    return zdx.Fisher(matrix, layout)


def test_decomposition_types_are_public():
    names = [
        "Decomposition",
        "EigenDecomposition",
        "CholeskyDecomposition",
        "ParameterProjection",
        "LocalParameterisation",
    ]
    for name in names:
        assert getattr(zdx, name) is getattr(zdx.derivatives, name)
        assert name in zdx.__all__
        assert name in zdx.derivatives.__all__


@pytest.mark.parametrize("kind", ["eigh", "cholesky", "projection", "parameterise"])
def test_decomposition_family_preserves_fields_and_immutable_updates(kind):
    fisher = _fisher()
    if kind == "parameterise":
        result = fisher.parameterise({"a": np.zeros(2), "b": np.zeros((), dtype=float)})
        path = "projection.matrix"
    else:
        result = getattr(fisher, kind)()
        path = {"eigh": "values", "cholesky": "factor", "projection": "matrix"}[kind]

    assert isinstance(result, zdx.Decomposition)
    original_array = result.get(path)
    updated = result.set(path, 2 * original_array)
    assert type(updated) is type(result)
    assert np.array_equal(updated.get(path), 2 * original_array)
    assert np.array_equal(result.get(path), original_array)

    file = BytesIO()
    zdx.save(file, result)
    restored = zdx.load(file)
    assert isinstance(restored, zdx.Decomposition)
    assert eqx.tree_equal(restored, result, typematch=True)


def test_eigh_uses_numpy_orientation_and_reconstructs_matrix():
    fisher = _fisher()
    decomposition = fisher.eigh()

    assert np.all(decomposition.values[:-1] <= decomposition.values[1:])
    assert np.allclose(
        fisher.matrix @ decomposition.vectors,
        decomposition.vectors * decomposition.values[None, :],
    )
    assert np.allclose(decomposition.reconstruct().matrix, fisher.matrix)
    assert decomposition.layout.compatible(fisher.layout)


def test_cholesky_reconstructs_solves_and_calculates_log_determinant():
    fisher = _fisher()
    decomposition = fisher.cholesky()
    right_hand_side = zdx.TreeVector(np.asarray([1.0, -2.0, 0.5]), fisher.layout)

    assert np.allclose(decomposition.reconstruct().matrix, fisher.matrix)
    assert np.allclose(
        fisher.matrix @ decomposition.solve(right_hand_side).vector,
        right_hand_side.vector,
    )
    assert np.allclose(
        decomposition.log_determinant,
        np.linalg.slogdet(fisher.matrix)[1],
    )


@pytest.mark.parametrize("matrix_type", [zdx.Hessian, zdx.GaussNewton, zdx.Fisher])
def test_all_symmetric_matrix_types_expose_decompositions(matrix_type):
    fisher = _fisher()
    matrix = matrix_type(fisher.matrix, fisher.layout)

    assert isinstance(matrix.eigh(), zdx.EigenDecomposition)
    assert isinstance(matrix.cholesky(), zdx.CholeskyDecomposition)
    assert isinstance(matrix.projection(), zdx.ParameterProjection)


def test_full_rank_eigen_projection_whitens_metric_and_recovers_inverse():
    fisher = _fisher()
    projection = fisher.projection(rtol=0.0)
    identity = np.eye(fisher.layout.size)

    assert projection.method == "eigh"
    assert np.all(projection.retained)
    assert int(projection.rank) == fisher.layout.size
    assert np.allclose(
        projection.matrix.T @ fisher.matrix @ projection.matrix,
        identity,
        rtol=2e-5,
        atol=2e-5,
    )
    assert np.allclose(
        projection.covariance.matrix,
        np.linalg.inv(fisher.matrix),
        rtol=2e-5,
        atol=2e-5,
    )

    step = projection(np.ones(fisher.layout.size))
    assert isinstance(step, zdx.TreeVector)
    assert step.flat_dict()["a"].shape == (2,)


def test_projection_supports_optimisation_coordinate_operations():
    fisher = _fisher()
    projection = fisher.projection(rtol=0.0)
    latent = np.asarray([0.25, -1.0, 2.0])
    gradient = zdx.TreeVector(np.asarray([1.5, -0.5, 3.0]), fisher.layout)

    step = projection.apply(latent)
    pulled_back = projection.pullback(gradient)
    natural = projection.natural_gradient(gradient)

    assert np.allclose(projection.solve(step), latent, rtol=2e-5, atol=2e-5)
    assert np.allclose(pulled_back, projection.matrix.T @ gradient.vector)
    assert np.allclose(
        natural.vector,
        projection.covariance.matrix @ gradient.vector,
    )
    assert np.allclose(
        projection.natural_norm(gradient),
        np.sqrt(gradient.vector @ natural.vector),
    )


def test_matrix_parameterisation_keeps_pytree_bookkeeping_internal():
    origin = {
        "a": np.asarray([1.0, -2.0]),
        "b": np.asarray(0.5, dtype=np.float32),
    }
    fisher = _fisher()
    geometry = fisher.parameterise(origin, rtol=0.0)
    latent = np.asarray([0.25, -1.0, 2.0])

    at_origin = geometry(geometry.zeros)
    parameters = geometry(latent)
    step = geometry.step(latent)
    flat_origin, _ = ravel_pytree(origin)
    flat_parameters, _ = ravel_pytree(parameters)
    flat_step, _ = ravel_pytree(step)
    variance, _ = ravel_pytree(geometry.variance)
    uncertainty, _ = ravel_pytree(geometry.standard_deviation)

    assert isinstance(geometry, zdx.LocalParameterisation)
    assert jax.tree.all(jax.tree.map(np.allclose, at_origin, origin))
    assert np.allclose(flat_step, geometry.projection.matrix @ latent)
    assert np.allclose(flat_parameters, flat_origin + flat_step)
    assert np.allclose(geometry.encode(parameters), latent, rtol=2e-5, atol=2e-5)
    assert np.allclose(variance, np.diag(geometry.projection.covariance.matrix))
    assert np.allclose(uncertainty**2, variance)


def test_local_parameterisation_accepts_gradient_pytrees_and_jit():
    origin = {
        "a": np.asarray([1.0, -2.0]),
        "b": np.asarray(0.5, dtype=np.float32),
    }
    gradient = {
        "a": np.asarray([1.5, -0.5]),
        "b": np.asarray(3.0),
    }
    geometry = _fisher().projection(rtol=0.0).bind(origin)
    gradient_vector, _ = ravel_pytree(gradient)
    natural = geometry.natural_gradient(gradient)
    natural_vector, _ = ravel_pytree(natural)

    assert np.allclose(
        geometry.pullback(gradient),
        geometry.projection.matrix.T @ gradient_vector,
    )
    assert np.allclose(
        natural_vector,
        geometry.projection.covariance.matrix @ gradient_vector,
    )
    assert np.allclose(
        geometry.natural_norm(gradient),
        np.sqrt(gradient_vector @ natural_vector),
    )

    result = jax.jit(lambda latent: geometry(latent))(np.ones(3))
    assert jax.tree.all(jax.tree.map(lambda value: np.all(np.isfinite(value)), result))


def test_realised_tree_matrix_can_derive_its_layout_from_parameters():
    parameters = {"x": np.ones(2), "y": np.asarray(0.0)}
    matrix = np.eye(3)

    fisher = zdx.Fisher.from_tree(matrix, parameters)

    assert isinstance(fisher, zdx.Fisher)
    assert fisher.layout.compatible(zdx.TreeLayout.from_tree(parameters))


def test_cholesky_projection_whitens_full_rank_metric():
    fisher = _fisher()
    projection = fisher.projection(method="cholesky")

    assert projection.method == "cholesky"
    assert projection.eigenvalues is None
    assert np.all(projection.retained)
    assert np.allclose(
        projection.matrix.T @ fisher.matrix @ projection.matrix,
        np.eye(fisher.layout.size),
        rtol=2e-5,
        atol=2e-5,
    )


def test_cholesky_projection_damping_supports_singular_metrics():
    layout = zdx.TreeLayout(("x",), ((2,),))
    fisher = zdx.Fisher(np.asarray([[1.0, 1.0], [1.0, 1.0]]), layout)
    projection = fisher.projection(method="cholesky", damping=1e-3)

    assert np.all(np.isfinite(projection.matrix))
    assert np.allclose(
        projection.matrix.T @ (fisher.matrix + 1e-3 * np.eye(2)) @ projection.matrix,
        np.eye(2),
        rtol=2e-5,
        atol=2e-5,
    )


def test_eigen_projection_discards_unresolved_modes_without_changing_shape():
    layout = zdx.TreeLayout(("x",), ((3,),))
    fisher = zdx.Fisher(np.diag(np.asarray([4.0, 1e-8, 0.0])), layout)
    projection = fisher.projection(rtol=1e-5, equilibrate=False)

    assert projection.matrix.shape == fisher.matrix.shape
    assert np.array_equal(projection.retained, np.asarray([False, False, True]))
    assert int(projection.rank) == 1
    assert np.all(np.isfinite(projection.matrix))
    assert np.allclose(projection.matrix[:, ~projection.retained], 0)


def test_projection_equilibration_is_invariant_to_coordinate_units():
    fisher = _fisher()
    coordinate_scale = np.diag(np.asarray([1e3, 1e-2, 5.0]))
    rescaled = zdx.Fisher(
        coordinate_scale @ fisher.matrix @ coordinate_scale,
        fisher.layout,
    )
    original = fisher.projection(rtol=1e-5)
    transformed = rescaled.projection(rtol=1e-5)

    assert np.allclose(original.eigenvalues, transformed.eigenvalues)
    assert np.array_equal(original.retained, transformed.retained)
    assert np.allclose(
        coordinate_scale @ transformed.covariance.matrix @ coordinate_scale,
        original.covariance.matrix,
        rtol=3e-5,
        atol=3e-5,
    )


def test_decompositions_and_projection_support_filtered_jit():
    fisher = _fisher()
    layout = fisher.layout

    function = eqx.filter_jit(
        lambda matrix: (
            zdx.Fisher(matrix, layout).eigh(),
            zdx.Fisher(matrix, layout).cholesky(),
            zdx.Fisher(matrix, layout).projection(),
        )
    )
    eigen, cholesky, projection = function(fisher.matrix)

    assert np.allclose(eigen.reconstruct().matrix, fisher.matrix)
    assert np.allclose(cholesky.reconstruct().matrix, fisher.matrix)
    assert np.all(np.isfinite(projection.matrix))


def test_eigh_remains_differentiable_for_distinct_eigenvalues():
    layout = zdx.TreeLayout(("x",), ((2,),))

    def largest(value):
        matrix = np.asarray([[value, 0.25], [0.25, 1.0]])
        return zdx.Hessian(matrix, layout).eigh().values[-1]

    assert np.isfinite(jax.grad(largest)(np.asarray(2.0)))


def test_projection_roundtrips_through_archive():
    original = _fisher().projection()
    file = BytesIO()
    zdx.save(file, original)
    file.seek(0)
    loaded = zdx.load(file)

    assert isinstance(loaded, zdx.ParameterProjection)
    assert loaded.method == original.method
    assert loaded.layout.compatible(original.layout)
    assert np.array_equal(loaded.retained, original.retained)
    assert np.allclose(loaded.matrix, original.matrix)
    assert np.allclose(loaded.eigenvalues, original.eigenvalues)


def test_local_parameterisation_roundtrips_through_archive():
    origin = {
        "a": np.asarray([1.0, -2.0]),
        "b": np.asarray(0.5, dtype=np.float32),
    }
    original = _fisher().parameterise(origin)
    file = BytesIO()
    zdx.save(file, original)
    file.seek(0)
    loaded = zdx.load(file)

    assert isinstance(loaded, zdx.LocalParameterisation)
    assert jax.tree.all(
        jax.tree.map(np.allclose, loaded(np.ones(3)), original(np.ones(3)))
    )


def test_decomposition_and_projection_validate_inputs():
    fisher = _fisher()
    incompatible = zdx.TreeVector(np.ones(2), zdx.TreeLayout(("x",), ((2,),)))

    with pytest.raises(ValueError):
        fisher.cholesky().solve(incompatible)
    with pytest.raises(ValueError):
        fisher.projection(method="svd")
    with pytest.raises(TypeError):
        fisher.projection(rtol=np.asarray(1e-5))
    with pytest.raises(ValueError):
        fisher.projection(damping=-1.0)
    with pytest.raises(ValueError):
        fisher.projection(rtol=float("inf"))
    with pytest.raises(TypeError):
        fisher.projection(equilibrate=1)
    with pytest.raises(TypeError):
        zdx.EigenDecomposition.from_matrix(zdx.TreeMatrix(fisher.matrix, fisher.layout))
    with pytest.raises(ValueError):
        fisher.cholesky().solve(np.ones(2))
    with pytest.raises(ValueError):
        fisher.projection().apply(np.ones(2))

    with pytest.raises(TypeError):
        zdx.LocalParameterisation(object(), {"a": np.ones(2), "b": np.ones(())})
    with pytest.raises(ValueError):
        fisher.parameterise({"x": np.ones(3)})

    geometry = fisher.parameterise({"a": np.ones(2), "b": np.ones(())})
    with pytest.raises(ValueError):
        geometry.encode({"x": np.ones(3)})
    integer_gradient = {"a": np.ones(2, dtype=int), "b": np.ones((), dtype=int)}
    assert np.allclose(
        geometry.pullback(integer_gradient),
        geometry.projection.matrix.T @ np.ones(3),
    )


def test_eager_check_rejects_invalid_metric_decompositions():
    layout = zdx.TreeLayout(("x",), ((2,),))
    indefinite = zdx.Hessian(np.asarray([[1.0, 2.0], [2.0, 1.0]]), layout)

    with pytest.raises(ValueError):
        indefinite.check("cholesky")
    with pytest.raises(ValueError):
        indefinite.check("projection")


def test_tree_matrix_diagnostics_are_jittable_and_tolerance_aware():
    layout = zdx.TreeLayout(("x",), ((2,),))
    positive = np.asarray([[2.0, 0.25], [0.25, 1.0]])
    semidefinite = np.asarray([[1.0, 1.0], [1.0, 1.0]])
    indefinite = np.asarray([[1.0, 2.0], [2.0, 1.0]])
    asymmetric = np.asarray([[1.0, 1.0], [0.0, 1.0]])

    @jax.jit
    def diagnostics(matrix):
        value = zdx.TreeMatrix(matrix, layout)
        return (
            value.is_finite(),
            value.is_symmetric(),
            value.is_positive_semidefinite(),
            value.is_positive_definite(),
        )

    assert all(bool(value) for value in diagnostics(positive))
    assert tuple(bool(value) for value in diagnostics(semidefinite)) == (
        True,
        True,
        True,
        False,
    )
    assert tuple(bool(value) for value in diagnostics(indefinite)) == (
        True,
        True,
        False,
        False,
    )
    assert tuple(bool(value) for value in diagnostics(asymmetric)) == (
        True,
        False,
        False,
        False,
    )

    nearly_psd = zdx.TreeMatrix(np.diag(np.asarray([-1e-7, 1.0])), layout)
    assert nearly_psd.is_positive_semidefinite(rtol=1e-5)
    assert not nearly_psd.is_positive_semidefinite(rtol=0.0, atol=0.0)


def test_eager_check_accepts_valid_contracts_and_reports_failures():
    layout = zdx.TreeLayout(("x",), ((2,),))
    positive = zdx.TreeMatrix(np.asarray([[2.0, 0.25], [0.25, 1.0]]), layout)
    semidefinite = zdx.TreeMatrix(np.asarray([[1.0, 1.0], [1.0, 1.0]]), layout)
    asymmetric = zdx.TreeMatrix(np.asarray([[1.0, 1.0], [0.0, 1.0]]), layout)
    nonfinite = zdx.TreeMatrix(np.asarray([[1.0, np.inf], [np.inf, 1.0]]), layout)
    empty = zdx.TreeMatrix(
        np.empty((0, 0)),
        zdx.TreeLayout(("x",), ((0,),)),
    )

    assert positive.check("eigh") is positive
    assert positive.check("cholesky") is positive
    assert semidefinite.check("projection") is semidefinite

    with pytest.raises(ValueError):
        positive.check("svd")
    with pytest.raises(TypeError):
        positive.check(rtol=True)
    with pytest.raises(TypeError):
        positive.check(atol=np.asarray(1e-8))
    with pytest.raises(ValueError):
        positive.check(rtol=np.inf)
    with pytest.raises(ValueError):
        positive.check(atol=-1.0)
    with pytest.raises(ValueError):
        empty.check()
    with pytest.raises(ValueError):
        nonfinite.check()
    with pytest.raises(ValueError):
        asymmetric.check()
    with pytest.raises(ValueError):
        semidefinite.check("cholesky")

    with pytest.raises(TypeError):
        jax.jit(lambda matrix: zdx.TreeMatrix(matrix, layout).check())(positive.matrix)

    assert not empty.is_positive_semidefinite()
    assert not empty.is_positive_definite()


def test_decomposition_constructors_validate_storage():
    layout = zdx.TreeLayout(("x",), ((2,),))

    with pytest.raises(TypeError):
        zdx.EigenDecomposition(np.ones(2), np.eye(2), object())
    with pytest.raises(ValueError):
        zdx.EigenDecomposition(np.ones(3), np.eye(2), layout)
    with pytest.raises(ValueError):
        zdx.EigenDecomposition(np.ones(2), np.ones((2, 3)), layout)
    with pytest.raises(TypeError):
        zdx.CholeskyDecomposition(np.eye(2), object())
    with pytest.raises(ValueError):
        zdx.CholeskyDecomposition(np.ones((2, 3)), layout)

    with pytest.raises(TypeError):
        zdx.ParameterProjection(
            np.eye(2),
            np.ones(2, dtype=bool),
            object(),
            method="eigh",
            eigenvalues=np.ones(2),
        )
    with pytest.raises(ValueError):
        zdx.ParameterProjection(
            np.eye(2),
            np.ones(2, dtype=bool),
            layout,
            method="svd",
        )
    with pytest.raises(ValueError):
        zdx.ParameterProjection(
            np.ones((2, 3)),
            np.ones(2, dtype=bool),
            layout,
            method="cholesky",
        )
    converted = zdx.ParameterProjection(
        np.eye(2), np.ones(2), layout, method="cholesky"
    )
    assert converted.retained.dtype == np.dtype(bool)
    with pytest.raises(ValueError):
        zdx.ParameterProjection(
            np.eye(2),
            np.ones(3, dtype=bool),
            layout,
            method="cholesky",
        )
    with pytest.raises(ValueError):
        zdx.ParameterProjection(
            np.eye(2),
            np.ones(2, dtype=bool),
            layout,
            method="eigh",
            eigenvalues=np.ones(3),
        )
    with pytest.raises(ValueError):
        zdx.ParameterProjection(
            np.eye(2),
            np.ones(2, dtype=bool),
            layout,
            method="eigh",
        )
    with pytest.raises(ValueError):
        zdx.ParameterProjection(
            np.eye(2),
            np.ones(2, dtype=bool),
            layout,
            method="cholesky",
            eigenvalues=np.ones(2),
        )


@pytest.mark.parametrize("scale", [1e-16, 1.0, 1e16])
def test_projection_solve_preserves_modes_across_parameter_units(scale):
    origin = {"x": np.zeros(2)}
    metric = zdx.Fisher.from_tree(np.diag(np.array([scale, 1.0])), origin)
    projection = metric.projection()
    latent = np.array([2.0, -3.0])

    assert int(projection.rank) == 2
    recovered = eqx.filter_jit(lambda p, u: p.solve(p.apply(u)))(projection, latent)
    assert np.allclose(recovered, latent, rtol=2e-5, atol=2e-5)


def test_zero_rank_projection_returns_zero_coordinates_and_steps():
    origin = {"x": np.zeros(2)}
    geometry = zdx.Fisher.from_tree(np.zeros((2, 2)), origin).parameterise(origin)
    projection = geometry.projection

    assert int(projection.rank) == 0
    assert np.array_equal(projection.solve(np.ones(2)), np.zeros(2))
    assert np.array_equal(geometry.encode(origin), np.zeros(2))
    assert np.array_equal(projection.apply(np.ones(2)).vector, np.zeros(2))
    derivative = jax.jacrev(projection.solve)(np.ones(2))
    assert np.array_equal(derivative, np.zeros((2, 2)))


def test_projection_solve_is_least_squares_for_out_of_range_steps():
    layout = zdx.TreeLayout(("x",), ((3,),))
    projection = zdx.ParameterProjection(
        np.array([[1.0, 0, 0], [1.0, 0, 0], [0, 0, 2.0]]),
        np.array([True, False, True]),
        layout,
        method="eigh",
        eigenvalues=np.array([1.0, 0, 1.0]),
    )
    value = np.array([2.0, 4.0, 8.0])
    latent = projection.solve(value)

    assert np.allclose(latent, np.array([3.0, 0, 4.0]), atol=1e-6)
    residual = projection.apply(latent).vector - value
    assert np.allclose(projection.matrix.T @ residual, 0, atol=1e-6)
    assert np.allclose(
        jax.jacfwd(projection.solve)(value),
        np.array([[0.5, 0.5, 0], [0, 0, 0], [0, 0, 0.5]]),
        atol=1e-6,
    )


def test_projection_rank_can_change_without_changing_compiled_shapes():
    layout = zdx.TreeLayout(("x",), ((2,),))

    @jax.jit
    def recover(diagonal):
        projection = zdx.Fisher(np.diag(diagonal), layout).projection(equilibrate=False)
        return projection.rank, projection.solve(np.ones(2))

    for diagonal, expected_rank in [([0.0, 0.0], 0), ([0.0, 2.0], 1), ([1.0, 2.0], 2)]:
        rank, latent = recover(np.array(diagonal))
        assert int(rank) == expected_rank
        assert latent.shape == (2,)
        assert np.all(np.isfinite(latent))


@pytest.mark.parametrize("method", ["eigh", "cholesky"])
def test_projection_damping_follows_its_documented_coordinates(method):
    layout = zdx.TreeLayout(("x",), ((2,),))
    metric = zdx.Fisher(np.diag(np.array([4.0, 9.0])), layout)
    damping = 0.2
    projection = metric.projection(method=method, damping=damping, equilibrate=False)
    diagonal_shift = damping * 9.0 if method == "eigh" else damping
    expected = np.diag(1 / (np.array([4.0, 9.0]) + diagonal_shift))
    assert np.allclose(projection.covariance.matrix, expected)


def test_cholesky_covariance_has_finite_metric_derivatives():
    layout = zdx.TreeLayout(("x",), ((2,),))
    metric = np.diag(np.array([3.0, 2.0]))

    def covariance_sum(value):
        projection = zdx.Fisher(value, layout).projection(method="cholesky")
        return projection.covariance.matrix.sum()

    inverse_ones = np.linalg.solve(metric, np.ones(2))
    expected = -np.outer(inverse_ones, inverse_ones)
    assert np.allclose(jax.grad(covariance_sum)(metric), expected, rtol=2e-5)


def test_decompositions_do_not_overflow_when_symmetry_is_already_established():
    # Keep this case in float32 even when the main suite enables x64.
    with jax.experimental.disable_x64():
        diagonal = np.array([2e38, 1.0], dtype=np.float32)
        metric = zdx.Fisher.from_tree(np.diag(diagonal), {"x": np.zeros(2)})
        eigen = metric.eigh()
        cholesky = metric.cholesky()

        assert np.all(np.isfinite(eigen.values))
        assert np.allclose(eigen.values, diagonal[::-1])
        assert np.allclose(np.diag(cholesky.factor), np.sqrt(diagonal))
        for method in ("eigh", "cholesky"):
            projection = metric.projection(method=method, equilibrate=False, rtol=0)
            assert int(projection.rank) == 2
            assert np.all(np.isfinite(projection.matrix))


def test_local_uncertainties_reconstruct_mixed_input_dtypes():
    narrow = np.dtype("float32")
    wide = np.dtype("float64") if jax.config.x64_enabled else narrow
    origin = {"a": np.array(1.0, dtype=narrow), "b": np.array(2.0, dtype=wide)}
    metric = zdx.Fisher.from_tree(np.diag(np.array([4.0, 9.0])), origin)
    geometry = metric.parameterise(origin)

    assert np.allclose(geometry.variance["a"], 0.25)
    assert np.allclose(geometry.variance["b"], 1 / 9)
    assert np.allclose(geometry.standard_deviation["a"], 0.5)
    assert np.allclose(geometry.standard_deviation["b"], 1 / 3)
    assert all(
        leaf.dtype == np.asarray(0.0).dtype
        for leaf in jax.tree.leaves(geometry.variance)
    )


def test_local_parameterisation_preserves_module_class_aliases_and_static_metadata():
    origin = Calibration(
        offset=np.array(2.0, dtype=float),
        weights=np.array([3.0, 4.0]),
        alias=(("zero", "offset"),),
    )
    metric = zdx.Fisher.from_tree(np.diag(np.array([4.0, 9.0, 16.0])), origin)
    geometry = metric.parameterise(origin)
    result = eqx.filter_jit(lambda operation, u: operation(u))(geometry, np.ones(3))

    assert isinstance(result, Calibration)
    assert result.name == "calibration"
    assert result.alias == origin.alias
    assert result.get("zero") == result.offset
    assert np.allclose(result.weights, origin.weights + np.array([1 / 3, 1 / 4]))
    assert np.array_equal(origin.weights, np.array([3.0, 4.0]))
    assert isinstance(geometry.variance, Calibration)
    assert np.allclose(geometry.variance.weights, np.array([1 / 9, 1 / 16]))

    file = BytesIO()
    zdx.save(file, geometry)
    restored = zdx.load(file)
    assert isinstance(restored.origin, Calibration)
    assert jax.tree.all(jax.tree.map(np.allclose, restored(np.ones(3)), result))


@pytest.mark.parametrize("method", ["eigh", "cholesky", "projection"])
def test_decomposing_an_empty_parameter_space_has_a_clear_boundary(method):
    metric = zdx.Fisher.from_tree(np.empty((0, 0)), {"x": np.empty(0)})
    with pytest.raises(ValueError):
        getattr(metric, method)()
