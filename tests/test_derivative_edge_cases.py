from __future__ import annotations

from copy import copy

import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

import zodiax as zdx
from zodiax.derivatives.containers import _path_string, _static_shape
from zodiax.derivatives.operations import _get_batch_sizes, _square_matrix


def test_layout_supports_scalar_sequence_and_flat_mapping_paths():
    scalar = zdx.TreeLayout.from_tree(jnp.asarray(2.0))
    sequence = zdx.TreeLayout.from_tree((jnp.ones(2), jnp.ones(())))
    flat = zdx.TreeLayout(("group.value",), ((2,),))

    assert scalar.paths == ("value",)
    assert scalar.flatten(jnp.asarray(3.0)).shape == (1,)
    assert zdx.TreeLayout.from_paths(jnp.asarray(3.0), "value").paths == ("value",)
    assert sequence.paths == ("0", "1")
    assert sequence.flatten((jnp.ones(2), jnp.ones(()))).shape == (3,)
    assert jnp.array_equal(flat.flatten({"group.value": jnp.ones(2)}), jnp.ones(2))


def test_path_and_static_shape_validation():
    with pytest.raises(ValueError, match="reserved"):
        _path_string((jtu.DictKey("bad.key"),))
    with pytest.raises(TypeError, match="supports module attributes"):
        zdx.TreeLayout.from_tree({1: jnp.asarray(1.0)})

    with pytest.raises(TypeError, match="sequence of integers"):
        _static_shape(1)
    with pytest.raises(TypeError, match="dimension must be an integer"):
        _static_shape((True,))
    with pytest.raises(TypeError, match="dimension must be an integer"):
        _static_shape((1.5,))


@pytest.mark.parametrize(
    ("paths", "shapes", "message"),
    [
        ((), (), "at least one"),
        ((1,), ((),), "non-empty string"),
        (("",), ((),), "non-empty string"),
        (("a..b",), ((),), "empty components"),
        (("a", "a"), ((), ()), "unique"),
        (("a", "a.b"), ((), ()), "parent of another"),
        (("a",), ((), ()), "equal lengths"),
        (("a",), ((-1,),), "negative"),
    ],
)
def test_layout_constructor_rejects_invalid_metadata(paths, shapes, message):
    with pytest.raises((TypeError, ValueError), match=message):
        zdx.TreeLayout(paths, shapes)


def test_layout_tree_and_path_selection_validation():
    with pytest.raises(ValueError, match="at least one numerical leaf"):
        zdx.TreeLayout.from_tree({})
    with pytest.raises(ValueError, match="does not contain"):
        zdx.TreeLayout.from_paths({"a": jnp.ones(2)}, "missing")

    shared = jnp.ones(2)
    tree = {"a": shared, "b": shared}
    with pytest.raises(ValueError, match="same leaf"):
        zdx.TreeLayout.from_paths(tree, ("a", "b"))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("paths", ["a"], "must be tuples"),
        ("shapes", ((), ()), "equal lengths"),
        ("shapes", ([1],), "shape must be a tuple"),
        ("shapes", ((1.0,),), "dimension must be an integer"),
        ("shapes", ((-1,),), "negative"),
    ],
)
def test_layout_archive_validation(field, value, message):
    layout = copy(zdx.TreeLayout(("a",), ((1,),)))
    object.__setattr__(layout, field, value)
    with pytest.raises((TypeError, ValueError), match=message):
        layout.__zodiax_validate__()


def test_layout_flatten_unflatten_and_split_axis_validation():
    layout = zdx.TreeLayout(("a",), ((2,),))
    layout.__zodiax_validate__()

    with pytest.raises(TypeError, match="floating"):
        layout.flatten({"a": jnp.ones(2, dtype=int)})
    with pytest.raises(TypeError, match="floating"):
        layout.unflatten(jnp.ones(2, dtype=int))
    with pytest.raises(TypeError, match="axis must be an integer"):
        layout.split_axis(jnp.ones((2, 2)), 0.0)
    assert layout.split_axis(jnp.ones((3, 2)), -1, nested=True)["a"].shape == (3, 2)


def test_vector_constructor_and_archive_validation():
    layout = zdx.TreeLayout(("a",), ((2,),))
    vector = zdx.TreeVector.from_tree({"a": jnp.ones(2)}, layout=layout)
    vector.__zodiax_validate__()

    with pytest.raises(TypeError, match="layout must be a TreeLayout"):
        zdx.TreeVector(jnp.ones(2), object())

    invalid_layout = copy(vector)
    object.__setattr__(invalid_layout, "layout", object())
    with pytest.raises(TypeError, match="layout must be a TreeLayout"):
        invalid_layout.__zodiax_validate__()

    invalid_vector = copy(vector)
    object.__setattr__(invalid_vector, "vector", jnp.ones(3))
    with pytest.raises(ValueError, match="vector must have shape"):
        invalid_vector.__zodiax_validate__()


def test_matrix_constructor_and_archive_validation():
    layout = zdx.TreeLayout(("a",), ((2,),))
    matrix = zdx.TreeMatrix(jnp.eye(2), layout)
    matrix.__zodiax_validate__()

    with pytest.raises(TypeError, match="layout must be a TreeLayout"):
        zdx.TreeMatrix(jnp.eye(2), object())

    invalid_layout = copy(matrix)
    object.__setattr__(invalid_layout, "layout", object())
    with pytest.raises(TypeError, match="layout must be a TreeLayout"):
        invalid_layout.__zodiax_validate__()

    invalid_matrix = copy(matrix)
    object.__setattr__(invalid_matrix, "matrix", jnp.ones((2, 3)))
    with pytest.raises(ValueError, match="matrix must have shape"):
        invalid_matrix.__zodiax_validate__()

    asymmetric = copy(zdx.Hessian(jnp.eye(2), layout))
    object.__setattr__(asymmetric, "matrix", jnp.asarray([[1.0, 2.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="matrix must be symmetric"):
        asymmetric.__zodiax_validate__()


def test_jacobian_constructor_and_archive_validation():
    layout = zdx.TreeLayout(("a",), ((2,),))
    jacobian = zdx.Jacobian(jnp.eye(2), layout)
    jacobian.__zodiax_validate__()

    with pytest.raises(TypeError, match="layout must be a TreeLayout"):
        zdx.Jacobian(jnp.eye(2), object())
    with pytest.raises(ValueError, match="final axis"):
        zdx.Jacobian(jnp.ones(()), layout)

    invalid_layout = copy(jacobian)
    object.__setattr__(invalid_layout, "layout", object())
    with pytest.raises(TypeError, match="layout must be a TreeLayout"):
        invalid_layout.__zodiax_validate__()

    invalid_matrix = copy(jacobian)
    object.__setattr__(invalid_matrix, "matrix", jnp.ones(3))
    with pytest.raises(ValueError, match="final axis"):
        invalid_matrix.__zodiax_validate__()


def test_fisher_rejects_explicit_incompatible_layouts_for_jacobian_objects():
    layout = zdx.TreeLayout(("a",), ((2,),))
    jacobian = zdx.Jacobian(jnp.eye(2), layout)

    with pytest.raises(TypeError, match="TreeLayout or None"):
        zdx.Fisher.from_jacobian(jacobian, object())
    with pytest.raises(ValueError, match="compatible"):
        zdx.Fisher.from_jacobian(
            jacobian,
            zdx.TreeLayout(("other",), ((2,),)),
        )


def test_derivative_operation_conversion_and_batch_validation():
    with pytest.raises(TypeError, match="positive integer"):
        _get_batch_sizes(2, True)
    with pytest.raises(TypeError, match="floating array-like matrix"):
        _square_matrix(object(), "cov", 2)

    parameters = {"x": jnp.asarray([1.0, 2.0])}
    with pytest.raises(TypeError, match=r"f\(x\).*floating"):
        zdx.jacobian(lambda value: object(), parameters, jit=False)
    with pytest.raises(TypeError, match=r"f\(x\).*floating"):
        zdx.jacobian(
            lambda value: jnp.asarray([1, 2], dtype=jnp.int32),
            parameters,
            jit=False,
        )
    with pytest.raises(TypeError, match=r"f\(x\).*numerical scalar"):
        zdx.hessian(lambda value: object(), parameters, jit=False)
    with pytest.raises(TypeError, match=r"residual_fn\(x\).*floating"):
        zdx.gauss_newton(lambda value: object(), parameters, jit=False)
