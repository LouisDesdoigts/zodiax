from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import pytest
import zodiax as zdx
from jax import numpy as np


class NumericalModel(zdx.Module):
    output: Any


def _layout_tree():
    return {
        "z": np.asarray(7.0, dtype=np.float32),
        "group": {
            "vector": np.asarray([5.0, 6.0], dtype=np.float32),
            "matrix": np.arange(4.0, dtype=np.float32).reshape(2, 2),
        },
    }


def test_public_exports():
    names = [
        "TreeLayout",
        "TreeVector",
        "TreeMatrix",
        "Jacobian",
        "Hessian",
        "GaussNewton",
        "Fisher",
    ]

    for name in names:
        assert getattr(zdx, name) is getattr(zdx.derivatives, name)
        assert name in zdx.__all__
        assert name in zdx.derivatives.__all__

    assert zdx.Jacobian is not zdx.jacobian
    assert zdx.jacobian is zdx.derivatives.jacobian
    assert zdx.hessian is zdx.derivatives.hessian
    assert zdx.gauss_newton is zdx.derivatives.gauss_newton


def test_tree_layout_roundtrip_uses_canonical_jax_order():
    tree = _layout_tree()
    layout = zdx.TreeLayout.from_tree(tree)

    assert layout.paths == ("group.matrix", "group.vector", "z")
    assert layout.shapes == ((2, 2), (2,), ())
    assert layout.sizes == (4, 2, 1)
    assert layout.starts == (0, 4, 6)
    assert layout.size == 7

    vector = layout.flatten(tree)
    expected = np.asarray([0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0])
    assert np.allclose(vector, expected)

    flat = layout.unflatten(vector)
    nested = layout.unflatten(vector, nested=True)
    assert tuple(flat) == layout.paths
    assert flat["group.matrix"].shape == (2, 2)
    assert flat["z"].shape == ()
    assert np.allclose(nested["group"]["vector"], tree["group"]["vector"])
    assert np.allclose(layout.flatten(nested), vector)


def test_tree_layout_from_paths_preserves_selected_order():
    tree = {"a": np.ones(2), "nested": {"value": np.asarray(3.0)}}
    layout = zdx.TreeLayout.from_paths(tree, ("nested.value", "a"))

    assert layout.paths == ("nested.value", "a")
    assert np.allclose(layout.flatten(tree), np.asarray([3.0, 1.0, 1.0]))


@pytest.mark.parametrize("constructor", ["from_tree", "from_paths"])
def test_tree_layout_rejects_literal_dotted_mapping_keys(constructor):
    tree = {"nested.value": np.asarray(1.0)}

    with pytest.raises(ValueError, match="reserved as structural path"):
        if constructor == "from_tree":
            zdx.TreeLayout.from_tree(tree)
        else:
            zdx.TreeLayout.from_paths(tree, "nested.value")


def test_tree_layout_selects_unrealised_leaf_through_alias():
    latent = np.asarray([0.2, -0.1])
    model = NumericalModel(
        zdx.Exp(x=zdx.Map(x=latent, s=2.0, b=0.5)),
        alias=(("latent", "output.x.x"),),
    )
    layout = zdx.TreeLayout.from_paths(model, "latent")

    assert layout.paths == ("latent",)
    assert np.allclose(layout.flatten(model), latent)


def test_tree_layout_maps_zodiax_module_paths(create_base):
    model = create_base()
    layout = zdx.TreeLayout.from_tree(model)

    assert layout.paths == ("param", "b.param")
    assert np.allclose(layout.flatten(model), np.asarray([1.0, 2.0]))
    assert np.allclose(layout.unflatten(layout.flatten(model))["b.param"], 2.0)


def test_tree_layout_uses_the_flat_vector_dtype():
    tree = {
        "low_precision": np.asarray([1, 2], dtype=np.float16),
        "real": np.asarray(3.0, dtype=np.float32),
    }
    layout = zdx.TreeLayout.from_tree(tree)
    vector = layout.flatten(tree)
    restored = layout.unflatten(vector)
    compatible = zdx.TreeLayout.from_tree(
        {
            "low_precision": np.ones(2, dtype=np.float32),
            "real": np.ones((), dtype=np.float16),
        }
    )

    assert not hasattr(layout, "dtypes")
    assert restored["low_precision"].dtype == vector.dtype
    assert restored["real"].dtype == vector.dtype
    assert layout.compatible(compatible)


@pytest.mark.parametrize("dtype", [np.int32, np.bool_, np.complex64])
def test_tree_layout_rejects_non_floating_leaves(dtype):
    with pytest.raises(TypeError, match="floating"):
        zdx.TreeLayout.from_tree({"value": np.ones(2, dtype=dtype)})

    layout = zdx.TreeLayout(("value",), ((2,),))
    with pytest.raises(TypeError, match="floating"):
        layout.flatten({"value": np.ones(2, dtype=dtype)})


def test_tree_layout_from_tree_rejects_mixed_parameter_leaves():
    tree = {
        "coefficient": np.asarray([1.0, 2.0]),
        "index": np.asarray(3, dtype=np.int32),
        "label": "fixed",
    }

    with pytest.raises(TypeError, match="index.*floating"):
        zdx.TreeLayout.from_tree(tree)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (((), ()), "at least one"),
        ((("a", "a"), ((), ())), "unique"),
        (
            (("a", "a.b"), ((), (1,))),
            "parent",
        ),
        ((("a",), ((), (1,))), "equal lengths"),
        ((("a",), ((-1,),)), "negative"),
    ],
)
def test_tree_layout_rejects_invalid_metadata(args, message):
    with pytest.raises((TypeError, ValueError), match=message):
        zdx.TreeLayout(*args)


@pytest.mark.parametrize("shape", [(1.9,), ("2",), (True,)])
def test_tree_layout_rejects_non_integer_shape_dimensions(shape):
    with pytest.raises(TypeError, match="integer"):
        zdx.TreeLayout(("a",), (shape,))


def test_tree_layout_rejects_incompatible_values_and_axes():
    layout = zdx.TreeLayout(("a",), ((2,),))

    with pytest.raises(ValueError, match="does not contain"):
        layout.flatten({"b": np.ones(2)})
    with pytest.raises(ValueError, match="shape"):
        layout.flatten({"a": np.ones(3)})
    with pytest.raises(ValueError, match="shape"):
        layout.unflatten(np.ones(3))
    with pytest.raises(ValueError, match="expected 2"):
        layout.split_axis(np.ones((3, 4)), 0)
    with pytest.raises(ValueError, match="axis"):
        layout.split_axis(np.ones((2, 4)), 2)


def test_tree_layout_roundtrip_is_jittable_and_differentiable():
    layout = zdx.TreeLayout.from_tree(_layout_tree())
    vector = np.arange(layout.size, dtype=float)

    @eqx.filter_jit
    def roundtrip(value):
        return layout.flatten(layout.unflatten(value, nested=True))

    assert np.allclose(roundtrip(vector), vector)
    gradient = jax.grad(lambda value: np.sum(roundtrip(value) ** 2))(vector)
    assert np.allclose(gradient, 2 * vector)


def test_tree_vector_views_and_validation():
    tree = _layout_tree()
    vector = zdx.TreeVector.from_tree(tree)

    assert np.allclose(vector.vector, vector.layout.flatten(tree))
    assert vector.flat_dict()["group.vector"].shape == (2,)
    assert vector.nested_dict()["group"]["matrix"].shape == (2, 2)

    with pytest.raises(ValueError, match="vector must have shape"):
        zdx.TreeVector(np.ones((1, vector.layout.size)), vector.layout)


def test_tree_matrix_axis_and_block_views():
    layout = zdx.TreeLayout(("a", "z"), ((2,), ()))
    matrix = np.arange(9.0, dtype=np.float32).reshape(3, 3)
    tree_matrix = zdx.TreeMatrix(matrix, layout)

    row_view = tree_matrix.rows()
    column_view = tree_matrix.columns()
    blocks = tree_matrix.blocks()

    assert row_view["a"].shape == (2, 3)
    assert row_view["z"].shape == (3,)
    assert np.allclose(row_view["a"], matrix[:2])
    assert column_view["a"].shape == (3, 2)
    assert column_view["z"].shape == (3,)
    assert np.allclose(column_view["a"], matrix[:, :2])
    assert blocks["a"]["a"].shape == (2, 2)
    assert blocks["a"]["z"].shape == (2,)
    assert np.allclose(blocks["z"]["a"], matrix[2, :2])

    with pytest.raises(ValueError, match="matrix must have shape"):
        zdx.TreeMatrix(np.ones((3, 4)), layout)


def test_tree_matrix_nested_blocks_follow_both_layouts():
    layout = zdx.TreeLayout(
        ("source.position", "flux"),
        ((2,), ()),
    )
    matrix = np.arange(9.0).reshape(3, 3)
    blocks = zdx.TreeMatrix(matrix, layout).blocks(nested=True)

    assert np.allclose(
        blocks["source"]["position"]["source"]["position"],
        matrix[:2, :2],
    )
    assert np.allclose(blocks["flux"]["source"]["position"], matrix[2, :2])


def test_tree_matrix_diagonal_and_norms_map_to_the_correct_axes():
    layout = zdx.TreeLayout(("a", "z"), ((2,), ()))
    matrix = np.arange(9.0, dtype=np.float32).reshape(3, 3)
    tree_matrix = zdx.TreeMatrix(matrix, layout)

    diagonal = tree_matrix.diagonal().flat_dict()
    row_norms = tree_matrix.row_norms().vector
    column_norms = tree_matrix.column_norms().vector

    assert np.allclose(diagonal["a"], np.asarray([0.0, 4.0]))
    assert np.allclose(diagonal["z"], 8.0)
    assert np.allclose(row_norms, np.linalg.norm(matrix, axis=1))
    assert np.allclose(column_norms, np.linalg.norm(matrix, axis=0))
    assert not np.allclose(row_norms, column_norms)


def test_tree_matrix_views_are_jittable_and_differentiable():
    layout = zdx.TreeLayout(("a", "z"), ((2,), ()))
    matrix = np.arange(9.0, dtype=np.float32).reshape(3, 3)

    @eqx.filter_jit
    def diagonal(value):
        return zdx.TreeMatrix(value, layout).diagonal().vector

    assert np.allclose(diagonal(matrix), np.diag(matrix))
    gradient = jax.grad(lambda value: diagonal(value).sum())(matrix)
    assert np.allclose(gradient, np.eye(3))


def test_jacobian_maps_the_final_parameter_axis():
    parameters = {
        "z": np.asarray(2.0),
        "a": np.asarray([1.0, 3.0]),
    }

    def function(values):
        a = values["a"]
        z = values["z"]
        return np.asarray(
            [
                [a[0] ** 2 + 2 * z, a[1] * z],
                [a[0] * a[1] + z, z**2],
            ]
        )

    jacobian = zdx.jacobian(
        function,
        parameters,
        nbatches=2,
        checkpoint=True,
    )
    expected = np.asarray(
        [
            [[2.0, 0.0, 2.0], [0.0, 2.0, 3.0]],
            [[3.0, 1.0, 1.0], [0.0, 0.0, 4.0]],
        ]
    )

    assert jacobian.output_shape == (2, 2)
    assert jacobian.layout.paths == ("a", "z")
    assert np.allclose(jacobian.matrix, expected)

    columns = jacobian.columns(nested=True)
    assert columns["a"].shape == (2, 2, 2)
    assert columns["z"].shape == (2, 2)
    assert np.allclose(columns["a"], expected[..., :2])
    assert np.allclose(columns["z"], expected[..., 2])

    assert np.allclose(jacobian.row_norms(), np.linalg.norm(expected, axis=-1))
    column_norms = jacobian.column_norms()
    assert np.allclose(
        column_norms.vector,
        np.linalg.norm(expected.reshape(-1, 3), axis=0),
    )
    assert column_norms.flat_dict()["a"].shape == (2,)
    assert column_norms.flat_dict()["z"].shape == ()


def test_jacobian_rejects_non_array_like_mapping_outputs():
    parameters = {"a": np.ones(2)}

    with pytest.raises(TypeError, match="array-like"):
        zdx.jacobian(lambda values: {"output": values["a"]}, parameters)


def test_jacobian_accepts_coercible_array_like_outputs():
    parameters = {"a": np.asarray([1.0, 2.0])}
    jacobian = zdx.jacobian(
        lambda values: [values["a"][0], 2 * values["a"][1]],
        parameters,
        jit=False,
    )

    assert jacobian.output_shape == (2,)
    assert np.allclose(jacobian.matrix, np.asarray([[1.0, 0.0], [0.0, 2.0]]))


def test_jacobian_supports_scalar_array_outputs_with_padding():
    parameters = {"a": np.asarray([1.0, 3.0]), "z": np.asarray(2.0)}
    jacobian = zdx.jacobian(
        lambda values: np.sum(values["a"] ** 2) + 3 * values["z"],
        parameters,
        nbatches=2,
        jit=False,
    )

    assert jacobian.output_shape == ()
    assert jacobian.matrix.shape == (3,)
    assert np.allclose(jacobian.matrix, np.asarray([2.0, 6.0, 3.0]))
    assert jacobian.columns()["a"].shape == (2,)
    assert jacobian.columns()["z"].shape == ()


def test_jacobian_preserves_array_output_shape():
    parameters = {"a": np.asarray([1.0, 2.0])}

    def function(values):
        a = values["a"]
        return np.asarray([[a[0] ** 2, a[0] * a[1]], [a[1], a.sum()]])

    jacobian = zdx.jacobian(function, parameters, jit=False)

    expected = np.asarray(
        [
            [[2.0, 0.0], [2.0, 1.0]],
            [[0.0, 1.0], [1.0, 1.0]],
        ]
    )

    assert jacobian.matrix.shape == (2, 2, 2)
    assert np.allclose(jacobian.matrix, expected)
    assert jacobian.layout.paths == ("a",)
    assert jacobian.columns()["a"].shape == (2, 2, 2)


def test_jacobian_supports_higher_derivatives():
    def derivative(value):
        parameters = {"a": np.asarray([value])}
        function = lambda values: values["a"] ** 2
        return zdx.jacobian(
            function,
            parameters,
            jit=False,
        ).matrix[0, 0]

    assert np.allclose(jax.grad(derivative)(np.asarray(1.0)), 2.0)


def test_hessian_supports_higher_derivatives():
    def derivative(value):
        parameters = {"a": np.asarray([value])}
        function = lambda values: values["a"][0] ** 3
        return zdx.hessian(function, parameters, jit=False).matrix[0, 0]

    assert np.allclose(jax.grad(derivative)(np.asarray(2.0)), 6.0)


def test_hessian_preserves_matrix_coordinate_labels():
    parameters = {
        "z": np.asarray(2.0),
        "a": np.asarray([1.0, 3.0]),
    }

    def function(values):
        a = values["a"]
        z = values["z"]
        return a[0] ** 2 + 3 * a[1] ** 2 + 5 * a[0] * z - z**2

    hessian = zdx.hessian(function, parameters, nbatches=2)
    expected = np.asarray([[2.0, 0.0, 5.0], [0.0, 6.0, 0.0], [5.0, 0.0, -2.0]])

    assert hessian.layout.paths == ("a", "z")
    assert np.allclose(hessian.matrix, expected)
    assert np.allclose(hessian.blocks()["a"]["z"], np.asarray([5.0, 0.0]))
    assert np.allclose(hessian.diagonal().flat_dict()["z"], -2.0)


def test_fisher_from_jacobian_matches_gaussian_information():
    layout = zdx.TreeLayout(("a", "z"), ((2,), ()))
    jacobian = np.asarray([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]])
    covariance = np.asarray([[2.0, 0.0], [0.0, 4.0]])

    unweighted = zdx.Fisher.from_jacobian(jacobian, layout)
    weighted = zdx.Fisher.from_jacobian(
        jacobian,
        layout,
        covariance=covariance,
    )
    tree_jacobian = zdx.Jacobian(jacobian.reshape(1, 2, 3), layout)
    from_object = zdx.Fisher.from_jacobian(
        tree_jacobian,
        covariance=covariance,
    )

    assert np.allclose(unweighted.matrix, jacobian.T @ jacobian)
    assert np.allclose(
        weighted.matrix,
        jacobian.T @ np.linalg.inv(covariance) @ jacobian,
    )
    assert weighted.layout.compatible(layout)
    assert weighted.diagonal().flat_dict()["a"].shape == (2,)
    assert np.allclose(from_object.matrix, weighted.matrix)


def test_fisher_from_jacobian_validates_shapes():
    layout = zdx.TreeLayout(("a",), ((2,),))

    with pytest.raises(ValueError, match="final axis"):
        zdx.Fisher.from_jacobian(np.ones((3, 3)), layout)
    with pytest.raises(ValueError, match="covariance must have shape"):
        zdx.Fisher.from_jacobian(
            np.ones((3, 2)),
            layout,
            covariance=np.eye(2),
        )
    with pytest.raises(TypeError, match="layout must be a TreeLayout"):
        zdx.Fisher.from_jacobian(np.ones((3, 2)))


@pytest.mark.parametrize(
    "matrix_type",
    [zdx.Hessian, zdx.GaussNewton, zdx.Fisher],
)
def test_symmetric_matrix_types_enforce_symmetry_under_jit(matrix_type):
    layout = zdx.TreeLayout(("a",), ((2,),))
    matrix = np.asarray([[1.0, 3.0], [1.0, 2.0]])
    expected = np.asarray([[1.0, 2.0], [2.0, 2.0]])

    construct = eqx.filter_jit(lambda value: matrix_type(value, layout))

    assert np.allclose(matrix_type(matrix, layout).matrix, expected)
    assert np.allclose(construct(matrix).matrix, expected)
