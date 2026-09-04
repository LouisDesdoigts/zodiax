from __future__ import annotations

from collections import namedtuple
from copy import copy
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import zodiax as zdx
from zodiax.numerics.arrays import (
    _as_float_array,
    _as_inexact_array,
    _validate_float_array,
    _validate_inexact_array,
)
from zodiax.numerics.expressions import _runtime_path
from zodiax.numerics.normalisation import _axes
from zodiax.numerics.parametric.bases import (
    _solve as solve_basis,
    _validate_x as validate_basis_x,
)
from zodiax.numerics.parametric.polynomials import _validate_powers_array
from zodiax.numerics.parametric.polynomials import _polynomial_terms
from zodiax.numerics.transforms import matmul
from zodiax.numerics.state import _leaf_signature, _normalise_values


class ContextValue(zdx.Expression):
    name: str = eqx.field(static=True)

    def evaluate(self, **context: Any):
        return context[self.name]


class NestedExpression(zdx.Expression):
    child: Any

    def evaluate(self, **context: Any):
        del context
        return self.child


class RuntimeContainers(zdx.Module):
    sequence: Any = zdx.field(runtime=True)
    mapping: Any = zdx.field(runtime=True)


@pytest.fixture
def restore_units():
    units = zdx.get_units()
    yield
    zdx.set_units(units)


def test_array_conversion_preserves_special_values_and_reports_invalid_inputs():
    expression = zdx.Exp()

    assert zdx.as_array(None) is None
    assert zdx.as_array(expression) is expression
    assert zdx.as_array([1, 2]).dtype == jnp.int32

    with pytest.raises(TypeError, match="array-like"):
        zdx.as_array(object())
    with pytest.raises(TypeError, match="floating or complex"):
        _as_inexact_array([1, 2], dtype=jnp.int32)
    with pytest.raises(TypeError, match="floating and array-like"):
        _as_float_array(object())
    with pytest.raises(TypeError, match="floating dtype"):
        _as_float_array(jnp.asarray([1]))


@pytest.mark.parametrize(
    ("validator", "value", "message"),
    [
        (_validate_inexact_array, 1.0, "JAX array"),
        (_validate_inexact_array, jnp.asarray(1.0), "strongly typed inexact"),
        (_validate_inexact_array, jnp.asarray(1, dtype=jnp.int32), "strongly typed"),
        (_validate_float_array, 1.0, "JAX array"),
        (_validate_float_array, jnp.asarray(1.0), "strongly typed floating"),
        (
            _validate_float_array,
            jnp.asarray(1 + 1j, dtype=jnp.complex64),
            "strongly typed floating",
        ),
    ],
)
def test_array_storage_validators_reject_weak_or_wrong_values(
    validator, value, message
):
    with pytest.raises(TypeError, match=message):
        validator(value, "sample")


def test_expression_call_runtime_containers_and_depth_limit():
    expression = ContextValue("coords")
    assert jnp.allclose(expression(coords=jnp.asarray(2.0)), 2.0)

    model = RuntimeContainers(sequence=(expression,), mapping={"value": expression})
    protected = model.resolve(coords=3.0)
    assert isinstance(protected.sequence[0], ContextValue)
    assert isinstance(protected.mapping["value"], ContextValue)

    realised = model.resolve(runtime=True, coords=3.0)
    assert jnp.allclose(realised.sequence[0], 3.0)
    assert jnp.allclose(realised.mapping["value"], 3.0)

    with pytest.raises(TypeError, match="runtime must be a bool"):
        zdx.resolve(expression, runtime=1)

    nested: Any = jnp.asarray(1.0)
    for _ in range(101):
        nested = NestedExpression(nested)
    with pytest.raises(ValueError, match="exceeded 100"):
        nested.resolve()

    assert not _runtime_path(object(), (object(),))

    nested_runtime = (RuntimeContainers(sequence=(expression,), mapping={}),)
    protected_nested = zdx.resolve(nested_runtime, coords=4.0)
    assert isinstance(protected_nested[0].sequence[0], ContextValue)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: zdx.Grid(n=(), d=1.0), "at least one size"),
        (lambda: zdx.Grid(n=True, d=1.0), "contain integers"),
        (lambda: zdx.Grid(n=1.5, d=1.0), "contain integers"),
        (lambda: zdx.Grid(n=0, d=1.0), "positive sizes"),
        (lambda: zdx.Grid(n=(2, 2), d=(1.0, 2.0, 3.0)), "axis of size 2"),
        (lambda: zdx.Grid(n=2, d=1.0, unit=1), "unit string"),
        (lambda: zdx.Grid(n=2, d=1.0, unit=zdx.Unit(1.0, "m")), "unbound"),
        (lambda: zdx.Grid(n=2, d=1.0, unit="m^2"), "simple Cartesian"),
        (lambda: zdx.Grid(n=2, d=1.0, unit="photon"), "Cartesian or angular"),
    ],
)
def test_grid_constructor_validation(factory, message):
    with pytest.raises((TypeError, ValueError), match=message):
        factory()


def test_grid_metadata_axes_resize_broadcast_and_from_axes():
    grid = zdx.Grid(n=(2, 3), d=(1.0, 2.0), c=None, alias=("step", "d"))

    assert grid.ndim == 2
    assert grid.shape == (2, 3)
    assert grid.batch_shape == ()
    assert [axis.shape for axis in grid.axes] == [(2,), (3,)]
    assert jnp.allclose(grid.fov, jnp.asarray([2.0, 6.0]))
    assert jnp.allclose(grid.measure, 2.0)
    assert jnp.array_equal(grid.coordinates, grid.resolve())
    assert grid.resize((4, 5)).shape == (4, 5)
    assert grid.match_shape((2, 3)).shape == (2, 3)

    one = zdx.Grid(n=3, d=jnp.asarray([2.0]), c=jnp.asarray([1.0]))
    broadcast = one.broadcast(3)
    assert broadcast.shape == (3, 3, 3)
    assert broadcast.d.shape == (3,)

    rebuilt = zdx.Grid.from_axes(
        (jnp.asarray([-1.0, 1.0]), jnp.asarray([2.0, 5.0, 8.0])),
        alias=("spacing", "d"),
    )
    assert rebuilt.n == (2, 3)
    assert jnp.allclose(rebuilt.d, jnp.asarray([2.0, 3.0]))
    assert jnp.allclose(rebuilt.c, jnp.asarray([0.0, 5.0]))
    assert rebuilt.alias == (("spacing", "d"),)
    rebuilt.__zodiax_validate__()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("n", [], "non-empty tuple"),
        ("n", (0,), "positive Python integers"),
        ("d", jnp.asarray([1], dtype=jnp.int32), "strongly typed inexact"),
        ("c", jnp.asarray([1.0, 2.0]), "axis of size 1"),
    ],
)
def test_grid_archive_validation_rejects_noncanonical_storage(field, value, message):
    grid = copy(zdx.Grid(n=2, d=1.0))
    object.__setattr__(grid, field, value)
    with pytest.raises((TypeError, ValueError), match=message):
        grid.__zodiax_validate__()


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda grid: grid.axes_for(3), "dimensionality"),
        (lambda grid: grid.broadcast(True), "positive integer"),
        (lambda grid: grid.broadcast(1.5), "positive integer"),
        (lambda grid: grid.broadcast(0), "positive integer"),
        (lambda grid: grid.broadcast(3), "cannot broadcast"),
        (lambda grid: grid.match_shape((3, 2)), "must match"),
        (lambda grid: grid.resize(2), "dimensionality"),
    ],
)
def test_grid_shape_operation_validation(operation, message):
    with pytest.raises((TypeError, ValueError), match=message):
        operation(zdx.Grid(n=(2, 3), d=1.0))


@pytest.mark.parametrize(
    "axes",
    [(), (jnp.asarray(1.0),), (jnp.asarray([1.0]),)],
)
def test_grid_from_axes_rejects_missing_or_short_axes(axes):
    with pytest.raises(ValueError, match="axes|at least two"):
        zdx.Grid.from_axes(axes)


def test_grid_supports_batched_spacing_centres_and_contextual_leaves():
    grid = zdx.Grid(
        n=(2, 2),
        d=ContextValue("spacing"),
        c=ContextValue("centre"),
    )
    coords = grid.resolve(
        spacing=jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
        centre=jnp.asarray([0.0, 1.0]),
    )

    assert coords.shape == (2, 2, 2, 2)
    with pytest.raises(Exception, match="positive"):
        grid.resolve(spacing=jnp.asarray([-1.0, 2.0]), centre=0.0)


@pytest.mark.parametrize("mode", [None, 1, "unknown"])
def test_norm_rejects_invalid_modes(mode):
    with pytest.raises((TypeError, ValueError), match="mode"):
        zdx.Norm(mode=mode)


@pytest.mark.parametrize("axis", [(), (0, 0), True, 1.5])
def test_norm_rejects_invalid_static_axes(axis):
    with pytest.raises((TypeError, ValueError), match="axis"):
        zdx.MeanNorm(axis=axis)


def test_norm_axes_weights_complex_values_and_targets():
    x = jnp.asarray([[1.0, 3.0], [2.0, 6.0]])
    assert jnp.allclose(zdx.MeanNorm(x, axis=-1).resolve().mean(-1), 1.0)
    assert jnp.allclose(zdx.SumNorm(x, axis=0).resolve().sum(0), 1.0)
    assert jnp.allclose(
        jnp.sqrt(jnp.mean(jnp.abs(zdx.RMSNorm(x + 1j * x).resolve()) ** 2)),
        1.0,
    )
    weighted_rms = zdx.RMSNorm(x, w=jnp.asarray([[1.0, 0.0], [1.0, 0.0]])).resolve()
    assert jnp.allclose(jnp.sqrt(jnp.sum(weighted_rms[:, 0] ** 2) / 2), 1.0)

    with pytest.raises(ValueError, match="axis entries"):
        zdx.MeanNorm(x, axis=2).resolve()
    with pytest.raises(TypeError, match="w must be real"):
        zdx.MeanNorm(x, w=1j).resolve()
    with pytest.raises(ValueError, match="broadcast"):
        zdx.MeanNorm(x, w=jnp.ones(3)).resolve()
    with pytest.raises(Exception, match="nonnegative"):
        zdx.MeanNorm(x, w=jnp.asarray([-1.0, 1.0])).resolve()
    with pytest.raises(Exception, match="positive total weight"):
        zdx.MeanNorm(x, w=0.0).resolve()
    with pytest.raises(Exception, match="measure must be nonzero"):
        zdx.SumNorm(jnp.zeros(2)).resolve()
    with pytest.raises(ValueError, match="without changing"):
        zdx.MeanNorm(x, s=jnp.ones((2, 1, 1))).resolve()
    with pytest.raises(ValueError, match="unique dimensions"):
        _axes((0, -2), 2)

    generic = zdx.Norm(x, mode="mean", axis=0, alias=("input", "x"))
    assert "mode='mean'" in repr(generic)
    assert "axis=(0,)" in repr(generic)
    generic.__zodiax_validate__()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mode", "MEAN", "canonical spelling"),
        ("axis", [0], "canonical tuple"),
    ],
)
def test_norm_archive_validation_rejects_noncanonical_storage(field, value, message):
    transform = copy(zdx.MeanNorm())
    object.__setattr__(transform, field, value)
    with pytest.raises(ValueError, match=message):
        transform.__zodiax_validate__()


@pytest.mark.parametrize("axes", [True, 1.5, 0, -1])
def test_basis_rejects_invalid_axes(axes):
    with pytest.raises((TypeError, ValueError), match="positive integer"):
        zdx.Basis(M=jnp.eye(2), axes=axes)


def test_basis_validation_inference_batches_and_solve():
    matrix = jnp.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    coefficients = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    transform = zdx.Basis(M=matrix)
    values = transform(coefficients)

    assert values.shape == (2, 3)
    assert jnp.allclose(transform.solve(values), coefficients, atol=1e-5)
    assert jnp.allclose(solve_basis(jnp.asarray(2.0), jnp.asarray([2.0]), 1), 1.0)

    with pytest.raises(ValueError, match="at least one dimension"):
        zdx.Basis(M=1.0)
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        zdx.Basis(M=jnp.ones(2), axes=2)
    with pytest.raises(ValueError, match="nonzero dimensions"):
        zdx.Basis(M=jnp.empty((0, 2)))
    with pytest.raises(ValueError, match="No trailing x dimensions"):
        transform(jnp.ones(4))
    with pytest.raises(ValueError, match="ambiguous"):
        zdx.Basis(M=jnp.ones((2, 2)), x=jnp.ones((2, 2)))
    with pytest.raises(ValueError, match="coefficient shape"):
        zdx.Basis(M=jnp.ones((2, 3)), axes=1)(jnp.ones(4))
    with pytest.raises(ValueError, match="output shape"):
        transform.solve(jnp.ones(4))
    with pytest.raises(ValueError, match="ambiguous while solving"):
        zdx.Basis(M=jnp.ones((2, 2, 2))).solve(jnp.ones((2, 2)))
    with pytest.raises(ValueError, match="output shape"):
        zdx.Basis(M=jnp.ones((2, 3)), axes=1).solve(jnp.ones(4))

    transform.__zodiax_validate__()
    with pytest.raises(ValueError, match="coefficient shape"):
        validate_basis_x(jnp.ones(3), jnp.ones((2, 3)), 1)


def test_basis_resolves_contextual_matrix_and_reports_none():
    transform = zdx.Basis(x=jnp.asarray([1.0, 2.0]), M=ContextValue("basis"))
    assert jnp.allclose(transform.resolve(basis=jnp.eye(2)), jnp.asarray([1.0, 2.0]))
    with pytest.raises(ValueError, match="must not resolve to None"):
        transform.resolve(basis=None)
    with pytest.raises(ValueError, match="must not resolve to None"):
        transform.solve(jnp.ones(2), basis=None)

    noncanonical = copy(zdx.Basis(M=jnp.eye(2), axes=1))
    object.__setattr__(noncanonical, "axes", np.int64(1))
    with pytest.raises(TypeError, match="positive Python integer"):
        noncanonical.__zodiax_validate__()

    bad_x = copy(zdx.Basis(M=jnp.eye(2)))
    object.__setattr__(bad_x, "x", jnp.ones(3))
    with pytest.raises(ValueError, match="No trailing x dimensions"):
        bad_x.__zodiax_validate__()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": 1}, "method must be a string"),
        ({"method": " "}, "must not be empty"),
        ({"extrap": object()}, "booleans or real"),
        ({"extrap": (0.0,)}, "lower and upper"),
        ({"extrap": jnp.inf}, "finite"),
        ({"period": True}, "positive real"),
        ({"period": 0.0}, "finite and positive"),
    ],
)
def test_interpolation_option_validation(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        zdx.Interpolation(knots=[0.0, 1.0], values=[0.0, 1.0], **kwargs)


@pytest.mark.parametrize(
    ("knots", "values", "message"),
    [
        (jnp.asarray([0 + 0j, 1 + 0j]), [0.0, 1.0], "knots must be real"),
        (jnp.ones((2, 2)), [0.0, 1.0], "one-dimensional"),
        ([0.0], [0.0], "at least two"),
        ([0.0, 1.0], 1.0, "leading axis"),
        ([0.0, 1.0], [1.0, 2.0, 3.0], "leading axis"),
    ],
)
def test_interpolation_data_validation(knots, values, message):
    with pytest.raises((TypeError, ValueError), match=message):
        zdx.Interpolation(knots=knots, values=values)


def test_interpolation_shapes_extrapolation_period_and_integration():
    knots = jnp.asarray([0.0, 1.0, 2.0])
    values = jnp.asarray([[0.0, 0.0], [2.0, 4.0], [0.0, 0.0]])
    transform = zdx.Interpolation(knots=knots, values=values, extrap=(0.0, True))
    query = jnp.asarray([[0.5, 1.5]])
    output = transform(query)

    assert output.shape == (1, 2, 2)
    assert jnp.allclose(output[0, 0], jnp.asarray([1.0, 2.0]))

    periodic = zdx.Interpolation(
        knots=knots,
        values=jnp.asarray([0.0, 1.0, 0.0]),
        period=2.0,
    )
    assert jnp.allclose(periodic(2.5), periodic(0.5))

    stored = zdx.Interpolation(
        x=0.5,
        knots=knots,
        values=jnp.asarray([0.0, 1.0, 0.0]),
        method=" LINEAR ",
    )
    assert jnp.allclose(stored.resolve(), 0.5)
    stored.__zodiax_validate__()

    linear = zdx.Interpolation(knots=knots, values=jnp.asarray([0.0, 2.0, 0.0]))
    assert jnp.allclose(linear.integrate(0.0, 2.0), 2.0)
    assert jnp.allclose(linear.integrate(2.0, 0.0), -2.0)
    assert jnp.allclose(
        linear.integrate(jnp.asarray([-1.0, 0.0]), jnp.asarray([0.0, 1.0])),
        jnp.asarray([0.0, 1.0]),
    )


def test_interpolation_integration_rejects_unsupported_contracts():
    knots = [0.0, 1.0, 2.0]
    values = [0.0, 1.0, 0.0]
    with pytest.raises(NotImplementedError, match="method='linear'"):
        zdx.Interpolation(knots=knots, values=values, method="cubic").integrate(0, 1)
    with pytest.raises(NotImplementedError, match="zero extrapolation"):
        zdx.Interpolation(knots=knots, values=values, extrap=True).integrate(0, 1)
    with pytest.raises(NotImplementedError, match="period"):
        zdx.Interpolation(knots=knots, values=values, period=2.0).integrate(0, 1)
    with pytest.raises(NotImplementedError, match="scalar sampled"):
        zdx.Interpolation(knots=knots, values=jnp.ones((3, 2))).integrate(0, 1)
    with jax.debug_nans(False):
        with pytest.raises(Exception, match="lower must not be NaN"):
            zdx.Interpolation(knots=knots, values=values).integrate(jnp.nan, 1)
        with pytest.raises(Exception, match="upper must not be NaN"):
            zdx.Interpolation(knots=knots, values=values).integrate(0, jnp.nan)


def test_interpolation_accepts_expression_data_independently():
    with_expression_knots = zdx.Interpolation(
        knots=ContextValue("sample_knots"), values=jnp.asarray([0.0, 2.0])
    )
    with_expression_values = zdx.Interpolation(
        knots=jnp.asarray([0.0, 1.0]), values=ContextValue("sample_values")
    )
    assert jnp.allclose(
        with_expression_knots(0.5, sample_knots=jnp.asarray([0.0, 1.0])), 1
    )
    assert jnp.allclose(
        with_expression_values(0.5, sample_values=jnp.asarray([0.0, 2.0])), 1
    )
    with_expression_knots.__zodiax_validate__()
    with_expression_values.__zodiax_validate__()

    with pytest.raises(ValueError, match="leading knot axis"):
        zdx.Interpolation(knots=ContextValue("knots"), values=1.0)

    invalid = copy(with_expression_knots)
    object.__setattr__(invalid, "values", jnp.asarray(1.0, dtype=jnp.float32))
    with pytest.raises(ValueError, match="leading knot axis"):
        invalid.__zodiax_validate__()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("method", "LINEAR", "stored canonically"),
        ("extrap", [0.0, 0.0], "stored canonically"),
        ("period", 0, "finite and positive"),
    ],
)
def test_interpolation_archive_validation_rejects_noncanonical_storage(
    field, value, message
):
    transform = copy(zdx.Interpolation(knots=[0.0, 1.0], values=[0.0, 1.0]))
    object.__setattr__(transform, field, value)
    with pytest.raises((TypeError, ValueError), match=message):
        transform.__zodiax_validate__()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"degree": True}, "degree must be an integer"),
        ({"degree": 1.5}, "degree must be an integer"),
        ({"degree": -1}, "at least 0"),
        ({"degree": 1, "ndim": 0}, "at least 1"),
        ({"p": jnp.ones(2)}, "integer dtype"),
        ({"p": jnp.empty((0, 2), dtype=int)}, "must have shape"),
        ({"p": [[-1, 0]]}, "nonnegative"),
        ({"degree": 1, "p": [[0, 1]]}, "mutually exclusive"),
        ({"degree": 1, "degrees": [0]}, "only one"),
        ({}, "Provide one"),
        ({"degrees": [0.0]}, "integer dtype"),
        ({"degrees": jnp.empty((0,), dtype=int)}, "at least one"),
        ({"degrees": [-1]}, "nonnegative"),
    ],
)
def test_polynomial_definition_validation(kwargs, message):
    with pytest.raises(Exception, match=message):
        zdx.Polynomial(**kwargs)


def test_polynomial_selected_degrees_explicit_p_and_shape_errors():
    selected = zdx.Polynomial(jnp.ones(4), degrees=[0, 2], ndim=2)
    assert jnp.array_equal(selected.p.sum(0), jnp.asarray([0, 2, 2, 2]))

    explicit = zdx.Polynomial(jnp.asarray([1.0, 2.0]), p=jnp.asarray([0, 1]))
    assert jnp.allclose(
        explicit(coords=jnp.asarray([0.0, 1.0])), jnp.asarray([1.0, 3.0])
    )

    with pytest.raises(ValueError, match="coefficient axis"):
        zdx.Polynomial(jnp.ones(3), degree=1)
    with pytest.raises(ValueError, match="coefficient axis"):
        zdx.Polynomial(degree=1)(jnp.ones(3), coords=jnp.ones(2))
    with pytest.raises(ValueError, match="coordinate shape"):
        zdx.Polynomial(degree=1).solve(jnp.ones(3), coords=jnp.ones(2))

    assert _polynomial_terms(jnp.asarray([0.0, 1.0]), explicit.p).shape == (2, 2)

    explicit.__zodiax_validate__()
    invalid = copy(explicit)
    object.__setattr__(invalid, "x", jnp.ones(3))
    with pytest.raises(ValueError, match="stored monomial term count"):
        invalid.__zodiax_validate__()


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([[0, 1]], "JAX array"),
        (jnp.asarray([[0.0, 1.0]]), "strongly typed integer"),
        (jnp.asarray([0, 1]), "must have shape"),
        (jnp.empty((1, 0), dtype=int), "must have shape"),
        (jnp.asarray([[-1, 0]]), "nonnegative"),
    ],
)
def test_polynomial_archive_power_validation(value, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _validate_powers_array(value)


@pytest.mark.parametrize(
    ("cov", "mean", "message"),
    [
        (jnp.asarray([[1 + 0j]]), None, "cov must be real"),
        (jnp.ones(2), None, "square matrix"),
        (jnp.eye(2), jnp.ones(3), "axis of size 2"),
    ],
)
def test_gaussian_constructor_validation(cov, mean, message):
    with pytest.raises((TypeError, ValueError), match=message):
        zdx.Gaussian(cov=cov, mean=mean)


def test_gaussian_stored_coordinates_batches_and_invalid_data():
    coords = zdx.Grid(n=(2, 2), d=1.0).resolve()
    transform = zdx.Gaussian(x=coords, cov=jnp.eye(2), mean=0.0)
    assert transform.resolve().shape == (2, 2)
    assert zdx.Gaussian(cov=jnp.eye(2), mean=jnp.asarray([0.0])).resolve(
        coords=coords
    ).shape == (2, 2)
    transform.__zodiax_validate__()

    batched_cov = jnp.stack((jnp.eye(2), 2 * jnp.eye(2)))
    assert zdx.Gaussian(cov=batched_cov).resolve(coords=coords).shape == (2, 2, 2)

    with pytest.raises(TypeError, match="coords must be real"):
        zdx.Gaussian(cov=jnp.eye(2))(coords.astype(complex))
    with pytest.raises(Exception, match="finite and symmetric"):
        zdx.Gaussian(cov=jnp.asarray([[1.0, 2.0], [0.0, 1.0]]))(coords)
    with pytest.raises(Exception, match="positive definite"):
        zdx.Gaussian(cov=-jnp.eye(2))(coords)
    with pytest.raises(TypeError, match="mean must be real"):
        zdx.Gaussian(cov=jnp.eye(2), mean=1j)(coords)
    with pytest.raises(ValueError, match="must not resolve to None"):
        zdx.Gaussian(cov=ContextValue("cov"))(coords, cov=None)


def test_state_normalises_nested_pytrees_and_validates_names():
    Pair = namedtuple("Pair", "left right")
    first = ContextValue("first")
    second = ContextValue("second")
    state = zdx.State(
        values={
            "nested": {"value": 1.0},
            "tupled": (first, second),
            "listed": [first, second],
            "named": Pair(first, second),
            "module": zdx.Add(b=1.0),
        }
    )

    assert jnp.allclose(state.value("nested.value"), 1.0)
    assert state.value("tupled.1") is second
    assert state.value("listed.0") is first
    assert isinstance(state.value("named"), Pair)
    assert isinstance(state.value("module"), zdx.Add)

    with pytest.raises(TypeError, match="must be a mapping"):
        zdx.State(values=1)
    with pytest.raises(ValueError, match="provided twice"):
        zdx.State(values={"value": 1.0}, value=2.0)
    with pytest.raises(TypeError, match="keys must be strings"):
        zdx.State(values={1: 1.0})
    with pytest.raises(ValueError, match="public Python identifier"):
        zdx.State(values={"not-valid": 1.0})
    with pytest.raises(ValueError, match="collides with the State API"):
        zdx.State(values={"step": 1.0})
    with pytest.raises(ValueError, match="must not be None"):
        zdx.State(sample=None)
    with pytest.raises(TypeError, match="must contain arrays"):
        zdx.State(sample=object())
    with pytest.raises(TypeError, match="State values must be a mapping"):
        _normalise_values([], zdx.State)
    assert _leaf_signature(object()) == ("python", object)


@pytest.mark.parametrize("path", [1, "", ".value", "value.", "_private", "class"])
def test_state_ref_path_validation(path):
    with pytest.raises((TypeError, ValueError), match="StateRef"):
        zdx.StateRef(path)


def test_state_value_ref_and_state_validation_failures():
    state = zdx.State(sample=1.0)

    with pytest.raises(KeyError, match="no value"):
        state.value("step")
    with pytest.raises(KeyError, match="no value"):
        state.value("missing")
    with pytest.raises(KeyError, match="no value"):
        state.ref("missing")
    with pytest.raises(TypeError, match="zodiax State"):
        zdx.StateRef("value").evaluate(state={"value": 1.0})
    with pytest.raises(TypeError, match="zodiax State"):
        zdx.validate_state(zdx.StateRef("value"), {"value": 1.0})


def test_state_step_all_transition_and_validation_paths():
    assert zdx.State(index=0).step().value("index") == 1
    assert jnp.allclose(zdx.State(time=0.0).step(0.25).value("time"), 0.25)
    assert jnp.allclose(zdx.State(time=0.0, dt=0.5).step().value("time"), 0.5)

    with pytest.raises(TypeError, match="own state"):
        zdx.State(index=0).step(state=zdx.State())
    with pytest.raises(ValueError, match="no time entry"):
        zdx.State(index=0).step(0.1)
    with pytest.raises(ValueError, match="requires dt"):
        zdx.State(time=0.0).step()
    with pytest.raises(ValueError, match="no index or time"):
        zdx.State(sample=1.0).step()
    with pytest.raises(TypeError, match="scalar integer"):
        zdx.State(index=jnp.ones(2, dtype=int)).step()
    with pytest.raises(TypeError, match="scalar integer"):
        zdx.State(index=0.0).step()
    with pytest.raises(TypeError, match="real floating"):
        zdx.State(time=jnp.asarray(0, dtype=int)).step(1.0)
    with pytest.raises(TypeError, match="State dt"):
        zdx.State(time=0.0).step(jnp.asarray(1, dtype=int))
    with pytest.raises(ValueError, match="without changing"):
        zdx.State(time=jnp.zeros(2)).step(jnp.ones((2, 1)))

    state = zdx.State(sample=1.0, alias=("entry", "values.sample"))
    assert "alias=" in repr(state)
    state.__zodiax_validate__()
    zdx.StateRef("sample").__zodiax_validate__()


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([], "stored as a dict"),
        ({"nested": {"b": 1.0, "a": 2.0}}, "stored canonically"),
        ({"step": 1.0}, "collides with the State API"),
    ],
)
def test_state_archive_validation_rejects_invalid_storage(values, message):
    state = copy(zdx.State(sample=1.0))
    object.__setattr__(state, "values", values)
    with pytest.raises((TypeError, ValueError), match=message):
        state.__zodiax_validate__()


def test_state_archive_validation_rejects_mapping_subclasses_and_bad_ref_path():
    class MappingSubclass(dict):
        pass

    state = copy(zdx.State(sample=1.0))
    object.__setattr__(state, "values", {"nested": MappingSubclass(sample=1.0)})
    with pytest.raises(TypeError, match="stored as dicts"):
        state.__zodiax_validate__()

    reference = copy(zdx.StateRef("sample"))
    object.__setattr__(reference, "path", "sample.")
    with pytest.raises(ValueError, match="canonical dotted path"):
        reference.__zodiax_validate__()


def test_state_ref_and_interpolation_canonical_mismatch_guards(monkeypatch):
    import zodiax.numerics.parametric.interpolation as interpolation_module
    import zodiax.numerics.state as state_module

    reference = zdx.StateRef("sample")
    with monkeypatch.context() as patch:
        patch.setattr(state_module, "_validate_path", lambda path: f"other.{path}")
        with pytest.raises(ValueError, match="stored canonically"):
            reference.__zodiax_validate__()

    transform = zdx.Interpolation(
        knots=jnp.asarray([0.0, 1.0]),
        values=jnp.asarray([0.0, 1.0]),
        period=1.0,
    )
    with monkeypatch.context() as patch:
        patch.setattr(interpolation_module, "_period", lambda value: 2.0)
        with pytest.raises(ValueError, match="period must be stored canonically"):
            transform.__zodiax_validate__()


def test_transform_identity_paths_initialise_projection_and_errors():
    values = jnp.asarray([1.0, 2.0])

    assert jnp.array_equal(zdx.Add().inv(values), values)
    assert jnp.array_equal(zdx.Mul().inv(values), values)
    assert jnp.allclose(zdx.Log().inv(jnp.log(values)), values)
    assert jnp.allclose(zdx.Exp().inv(jnp.exp(values)), values)

    transform = zdx.Exp(x=zdx.Mul(x=zdx.Add(b=1.0), s=2.0))
    initialised = transform.initialise(jnp.exp(jnp.asarray([4.0, 6.0])))
    assert jnp.allclose(initialised.x.x.x, jnp.asarray([1.0, 2.0]))
    assert jnp.allclose(transform.project(transform(values)), transform(values))

    with pytest.raises(NotImplementedError, match="does not define inv"):
        zdx.Pow(p=2).inv(values)
    with pytest.raises(ValueError, match="through an Expression"):
        zdx.Exp(x=ContextValue("value")).initialise(values)
    with pytest.raises(Exception, match="must be nonzero"):
        zdx.Mul(s=0.0).inv(values)

    assert jnp.allclose(zdx.Pow(p=2)(values), values**2)
    assert jnp.allclose(zdx.Exp(base=10)(values), 10**values)
    assert jnp.allclose(zdx.Log()(values), jnp.log(values))
    assert jnp.allclose(zdx.Log(base=10)(values), jnp.log10(values))
    assert "alias=" in repr(zdx.Add(b=1, alias=("bias", "b")))


def test_transform_shape_matrix_base_and_mask_validation():
    values = jnp.ones(2)
    with pytest.raises(ValueError, match="without changing"):
        zdx.Add(b=jnp.ones((2, 1)))(values)
    with pytest.raises(ValueError, match="without changing"):
        zdx.Mul(s=jnp.ones((2, 1)))(values)
    with pytest.raises(TypeError, match="base must be real"):
        zdx.Exp(base=1j)(values)
    with pytest.raises(Exception, match="positive and not equal"):
        zdx.Exp(base=1.0)(values)
    with pytest.raises(ValueError, match="p must not be None"):
        zdx.Pow()
    with pytest.raises(ValueError, match="at least one axis"):
        zdx.MatMul(M=1.0)
    with pytest.raises(ValueError, match="nonzero dimensions"):
        zdx.MatMul(M=jnp.empty((2, 0)))
    with pytest.raises(ValueError, match="final axis"):
        zdx.MatMul(M=jnp.ones((3, 2)))(values)
    with pytest.raises(ValueError, match="end in shape"):
        zdx.MatMul(M=jnp.ones((2, 2, 2))).solve(jnp.ones(3))
    with pytest.raises(ValueError, match="at least one axis"):
        matmul(values, ContextValue("matrix"), matrix=jnp.asarray(1.0))
    with pytest.raises(ValueError, match="nonzero dimensions"):
        matmul(values, ContextValue("matrix"), matrix=jnp.empty((2, 0)))

    symbolic_matrix = zdx.MatMul(M=ContextValue("matrix"))
    with pytest.raises(ValueError, match="must not resolve to None"):
        symbolic_matrix(values, matrix=None)
    with pytest.raises(ValueError, match="must not resolve to None"):
        symbolic_matrix.solve(values, matrix=None)
    with pytest.raises(ValueError, match="at least one axis"):
        symbolic_matrix.solve(values, matrix=jnp.asarray(1.0))
    with pytest.raises(ValueError, match="nonzero dimensions"):
        symbolic_matrix.solve(values, matrix=jnp.empty((2, 0)))

    with pytest.raises(TypeError, match="boolean dtype"):
        zdx.Mask(mask=[1, 0])
    with pytest.raises(ValueError, match="at least one axis"):
        zdx.Mask(mask=True)
    with pytest.raises(ValueError, match="select at least one"):
        zdx.Mask(mask=[False, False])
    mask = zdx.Mask(mask=[True, False, True])
    with pytest.raises(ValueError, match="final axis"):
        mask(jnp.ones(3))
    with pytest.raises(ValueError, match="end in shape"):
        mask.solve(jnp.ones(2))


def test_large_mask_repr_and_map_partial_reverse_paths():
    mask = zdx.Mask(mask=jnp.ones(10, dtype=bool))
    assert "10@" in repr(mask)

    values = jnp.asarray([2.0, 4.0])
    assert jnp.allclose(zdx.Map(b=1.0).solve(values + 1), values)
    assert jnp.allclose(zdx.Map(M=jnp.eye(2)).solve(values), values)
    assert jnp.allclose(zdx.Map(s=2.0).solve(2 * values), values)
    with pytest.raises(Exception, match="must be nonzero"):
        zdx.Map(s=0.0).solve(values)

    for transform in (
        zdx.Add(),
        zdx.Mul(),
        zdx.Pow(p=2),
        zdx.Exp(),
        zdx.Log(),
        zdx.MatMul(M=jnp.eye(2)),
        mask,
        zdx.Map(M=jnp.eye(2)),
    ):
        transform.__zodiax_validate__()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("shape", (), "non-empty tuple"),
        ("shape", (0,), "positive integers"),
        ("indices", [0], "JAX array"),
        ("indices", jnp.asarray([0.0]), "integer JAX array"),
        ("indices", jnp.asarray([], dtype=jnp.int32), "non-empty one-dimensional"),
        ("indices", jnp.asarray([1, 0], dtype=jnp.int32), "strictly increasing"),
        ("indices", jnp.asarray([0, 2], dtype=jnp.int32), "lie within shape"),
    ],
)
def test_mask_archive_validation_rejects_invalid_storage(field, value, message):
    transform = copy(zdx.Mask(mask=[True, False]))
    object.__setattr__(transform, field, value)
    with pytest.raises((TypeError, ValueError), match=message):
        transform.__zodiax_validate__()


def test_unit_catalogue_conversion_and_global_configuration(restore_units):
    assert zdx.canonical_unit("microns") == "um"
    assert zdx.canonical_unit("1/mm") == "1/mm"
    assert zdx.canonical_unit("mm^2") == "mm^2"
    assert zdx.canonical_unit("m^1") == "m"
    assert zdx.conversion_factor("m", "mm") == 1000.0
    assert zdx.convert(2.0, "m", "mm") == 2000.0

    configured = zdx.set_units({"length": "um", "angle": "mas", "photons": "photon"})
    assert configured == {"cartesian": "um", "angular": "mas", "photon": "photon"}
    snapshot = zdx.get_units()
    snapshot["cartesian"] = "m"
    assert zdx.get_units()["cartesian"] == "um"

    unit = zdx.Unit(unit="mm", to="um")
    assert unit.category == "cartesian"
    assert unit.factor == pytest.approx(1000.0)
    unit.__zodiax_validate__()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("unit", "microns", "canonical spellings"),
        ("to", "rad", "compatible physical"),
        ("_scale", np.float64(1000.0), "finite Python float"),
        ("_scale", float("inf"), "finite Python float"),
        ("_scale", 0.0, "does not match"),
        ("_scale", 2.0, "does not match"),
    ],
)
def test_unit_archive_validation_rejects_invalid_storage(
    field, value, message, restore_units
):
    unit = copy(zdx.Unit(unit="mm", to="um"))
    object.__setattr__(unit, field, value)
    with pytest.raises((TypeError, ValueError), match=message):
        unit.__zodiax_validate__()


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda: zdx.canonical_unit(1), "unit must be a string"),
        (lambda: zdx.canonical_unit(""), "must not be empty"),
        (lambda: zdx.canonical_unit("x" * 129), "at most 128"),
        (lambda: zdx.canonical_unit("unknown"), "Unknown unit"),
        (lambda: zdx.canonical_unit("m^2^3"), "Unknown unit"),
        (lambda: zdx.canonical_unit("m^x"), "Unknown unit"),
        (lambda: zdx.canonical_unit("m^0"), "nonzero integers"),
        (lambda: zdx.canonical_unit("m^9"), "nonzero integers"),
        (lambda: zdx.canonical_unit("rad^2"), "Only Cartesian"),
        (lambda: zdx.conversion_factor("m", "rad"), "Cannot convert"),
        (lambda: zdx.set_units(1), "must be a mapping"),
        (lambda: zdx.set_units({1: "m"}), "categories must be strings"),
        (lambda: zdx.set_units({"unknown": "m"}), "Unknown unit-system category"),
        (
            lambda: zdx.set_units({"length": "m", "cartesian": "mm"}),
            "supplied twice",
        ),
        (lambda: zdx.set_units({"cartesian": "rad"}), "not a simple"),
        (lambda: zdx.Unit(unit="m", to="rad"), "Cannot convert"),
    ],
)
def test_unit_validation_errors(operation, message, restore_units):
    with pytest.raises((TypeError, ValueError), match=message):
        operation()


def test_link_generation_evaluation_validation_and_alias_printing():
    generated = zdx.Linked(2.0)
    assert len(generated.key) == 32
    assert "…" in repr(generated)
    assert jnp.allclose(generated.resolve(), 2.0)

    aliased = zdx.Linked(2.0, key="scale", alias=("owned", "value"))
    assert "alias=" in repr(aliased)
    aliased.__zodiax_validate__()
    aliased.defer().__zodiax_validate__()

    with pytest.raises(TypeError, match="key must be a string"):
        zdx.Deferred(1)


def test_coordinate_context_validation_and_optional_context():
    from zodiax.numerics.parametric._context import (
        coordinate_axis,
        coordinate_context,
    )

    with pytest.raises(ValueError, match="must have shape"):
        coordinate_axis(jnp.ones((2, 3)), 2)
    with pytest.raises(ValueError, match="only one"):
        coordinate_context(1.0, 2.0, {}, required=True)
    with pytest.raises(ValueError, match="Provide coords"):
        coordinate_context(None, None, {}, required=True)
    assert coordinate_context(None, None, {"time": 1.0}, required=False) == (
        None,
        {"time": 1.0},
    )
