"""Overview 12.1: numerical conversion, configured dtypes, and deferred inputs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import zodiax as zdx


@pytest.mark.parametrize(
    "name", ["as_array", "Expression", "Transform", "Map", "Mask", "Operation"]
)
def test_public_conversion_and_transform_entry_points_are_exported(name):
    assert getattr(zdx, name) is getattr(zdx.numerics, name)
    assert name in zdx.__all__


@pytest.mark.parametrize("value", [2, 2.5, 2 + 3j, True])
def test_python_scalars_become_strongly_typed_scalar_arrays(value):
    array = zdx.as_array(value)

    assert isinstance(array, jax.Array)
    assert array.shape == ()
    assert array.dtype == jnp.asarray(value).dtype
    assert not array.weak_type
    np.testing.assert_array_equal(array, value)


@pytest.mark.parametrize(
    "value",
    [
        [[1, 2], [3, 4]],
        ((1.0, 2.0), (3.0, 4.0)),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        jnp.array([[1 + 2j, 3 - 1j], [2j, 4 + 0j]]),
    ],
)
def test_array_like_inputs_preserve_shape_and_inferred_dtype(value):
    array = zdx.as_array(value)

    assert isinstance(array, jax.Array)
    assert array.shape == (2, 2)
    assert array.dtype == jnp.asarray(value).dtype
    assert not array.weak_type
    np.testing.assert_array_equal(array, value)


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (float, [0.0, 1.0, 2.0]),
        (int, [0, 1, 2]),
        (complex, [0j, 1 + 0j, 2 + 0j]),
        (bool, [False, True, True]),
    ],
)
def test_explicit_numerical_types_use_jax_configured_default_precision(dtype, expected):
    array = zdx.as_array([0, 1, 2], dtype=dtype)

    # JAX, configured before the suite starts, owns the default width for each type.
    assert array.dtype == jnp.asarray(0, dtype=dtype).dtype
    assert array.shape == (3,)
    assert not array.weak_type
    np.testing.assert_array_equal(array, expected)


@pytest.mark.parametrize("dtype", [None, float, int, complex, bool])
def test_none_and_expressions_pass_through_without_evaluation_or_future_cast(dtype):
    definition = zdx.Map(x=[1, 2], s=2, b=1)

    assert zdx.as_array(None, dtype=dtype) is None
    assert zdx.as_array(definition, dtype=dtype) is definition
    result = definition.resolve()
    assert jnp.issubdtype(result.dtype, jnp.integer)
    np.testing.assert_array_equal(result, [3, 5])


def test_existing_weak_scalar_becomes_strong_without_changing_its_dtype():
    scalar = jnp.asarray(2.5)
    assert scalar.weak_type

    converted = zdx.as_array(scalar)

    assert not converted.weak_type
    assert converted.dtype == scalar.dtype
    np.testing.assert_array_equal(converted, 2.5)


@pytest.mark.parametrize("values", [[1, 2], [1.0, 2.0], [1 + 2j, 3 - 1j]])
def test_generic_constructor_boundaries_preserve_integer_float_and_complex_inputs(
    values,
):
    definition = zdx.Map(x=values)

    assert definition.x.dtype == jnp.asarray(values).dtype
    assert not definition.x.weak_type
    result = definition.resolve()
    assert result.dtype == definition.x.dtype
    np.testing.assert_array_equal(result, values)


def test_conversion_remains_usable_inside_a_differentiated_compiled_calculation():
    @jax.jit
    def energy(values):
        converted = zdx.as_array(values, dtype=float)
        return jnp.sum(converted**2)

    values = jnp.array([2.0, 3.0])
    np.testing.assert_allclose(energy(values), 13.0)
    np.testing.assert_allclose(jax.grad(energy)(values), [4.0, 6.0])
