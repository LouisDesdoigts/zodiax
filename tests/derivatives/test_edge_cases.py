"""Public structural and domain contracts for derivative inputs."""

import jax.numpy as jnp
import pytest

import zodiax as zdx


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


@pytest.mark.parametrize(
    ("paths", "shapes", "message"),
    [
        ((), (), "at least one"),
        ((1,), ((),), "non-empty string"),
        (("",), ((),), "non-empty string"),
        (("a..b",), ((),), "empty components"),
        (("a", "a"), ((), ()), "unique"),
        (("a.b", "a"), ((), ()), "parent of another"),
        (("a", "a-b", "a.b"), ((), (), ()), "parent of another"),
        (("a",), ((), ()), "equal lengths"),
        (("a",), ((-1,),), "negative"),
        (("a",), ((1.5,),), "integer"),
        (("a",), ((True,),), "integer"),
    ],
)
def test_layout_rejects_ambiguous_paths_and_invalid_shapes(paths, shapes, message):
    with pytest.raises((TypeError, ValueError), match=message):
        zdx.TreeLayout(paths, shapes)


def test_layout_path_selection_checks_missing_and_duplicate_leaves():
    with pytest.raises(ValueError, match="at least one numerical leaf"):
        zdx.TreeLayout.from_tree({})
    with pytest.raises(ValueError, match="does not contain"):
        zdx.TreeLayout.from_paths({"a": jnp.ones(2)}, "missing")
    with pytest.raises(TypeError, match="supports module attributes"):
        zdx.TreeLayout.from_tree({1: jnp.ones(2)})

    shared = jnp.ones(2)
    with pytest.raises(ValueError, match="same leaf"):
        zdx.TreeLayout.from_paths({"a": shared, "b": shared}, ("a", "b"))

    # Equal Python scalars remain independent coordinates despite interning.
    layout = zdx.TreeLayout.from_paths({"a": 1.0, "b": 1.0}, ("a", "b"))
    assert layout.size == 2


def test_axis_views_accept_negative_axes_and_reject_invalid_axes():
    layout = zdx.TreeLayout(("a",), ((2,),))
    assert layout.split_axis(jnp.ones((3, 2)), -1, nested=True)["a"].shape == (3, 2)
    with pytest.raises(TypeError, match="axis must be an integer"):
        layout.split_axis(jnp.ones((2, 2)), 0.0)
    with pytest.raises(ValueError, match="axis"):
        layout.split_axis(jnp.ones((2, 2)), 2)


@pytest.mark.parametrize("constructor", [zdx.TreeVector, zdx.TreeMatrix, zdx.Jacobian])
def test_container_constructors_require_a_coordinate_layout(constructor):
    with pytest.raises(TypeError, match="layout must be a TreeLayout"):
        constructor(jnp.eye(2), object())


def test_jacobian_requires_a_final_parameter_axis():
    layout = zdx.TreeLayout(("a",), ((2,),))
    with pytest.raises(ValueError, match="final axis"):
        zdx.Jacobian(jnp.ones(()), layout)
    with pytest.raises(ValueError, match="final axis"):
        zdx.Jacobian(jnp.ones((3, 3)), layout)


def test_empty_coordinate_views_remain_well_defined():
    tree = {"x": jnp.empty((0,))}
    layout = zdx.TreeLayout.from_tree(tree)
    matrix = zdx.TreeMatrix.from_tree(jnp.empty((0, 0)), tree)
    assert layout.flatten(tree).shape == (0,)
    assert layout.unflatten(jnp.empty(0))["x"].shape == (0,)
    assert matrix.blocks()["x"]["x"].shape == (0, 0)
    assert not matrix.is_positive_semidefinite()
    assert not matrix.is_positive_definite()


@pytest.mark.parametrize("operation", [zdx.jacobian, zdx.hessian, zdx.gauss_newton])
def test_derivative_calculations_require_a_parameter_coordinate(operation):
    with pytest.raises(ValueError, match="at least one coordinate"):
        operation(lambda p: p["x"].sum(), {"x": jnp.empty(0)})


@pytest.mark.parametrize("operation", [zdx.jacobian, zdx.hessian, zdx.gauss_newton])
@pytest.mark.parametrize(
    "output", [jnp.asarray(1), jnp.asarray(True), jnp.asarray(1j), {}]
)
def test_derivative_outputs_are_floating_numerical_values(operation, output):
    with pytest.raises(TypeError, match="floating array-like"):
        operation(lambda p: output, {"x": jnp.asarray(1.0)})
