from __future__ import annotations

import inspect
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import zodiax as zdx


class CoordinateScale(zdx.Expression):
    scale: Any

    def evaluate(self, *, coords, **context):
        del context
        return coords * self.scale


class RuntimePair(zdx.Module):
    ordinary: Any
    local: Any = zdx.field(runtime=True)


class RuntimeStage(zdx.Module):
    response: Any = zdx.field(runtime=True)
    gain: Any

    def __call__(self, coords, **context):
        realised = self.resolve(runtime=True, coords=coords, **context)
        return realised.gain * realised.response


@pytest.fixture
def restore_units():
    units = zdx.get_units()
    yield
    zdx.set_units(units)


def test_public_numerics_exports_are_available_from_package_root():
    names = (
        "Expression",
        "Transform",
        "Add",
        "Mul",
        "Pow",
        "Exp",
        "Log",
        "MatMul",
        "Mask",
        "Map",
        "Norm",
        "MeanNorm",
        "RMSNorm",
        "SumNorm",
        "Unit",
        "Linked",
        "Deferred",
        "State",
        "StateRef",
        "Grid",
        "Basis",
        "Polynomial",
        "Gaussian",
        "Interpolation",
        "realise",
    )

    for name in names:
        assert getattr(zdx, name) is getattr(zdx.numerics, name)
        assert name in zdx.__all__


def test_transform_constructors_consistently_accept_x_first():
    classes = (
        zdx.Add,
        zdx.Mul,
        zdx.Pow,
        zdx.Exp,
        zdx.Log,
        zdx.MatMul,
        zdx.Mask,
        zdx.Map,
        zdx.Norm,
        zdx.MeanNorm,
        zdx.RMSNorm,
        zdx.SumNorm,
        zdx.Unit,
        zdx.Basis,
        zdx.Polynomial,
        zdx.Gaussian,
        zdx.Interpolation,
    )

    for cls in classes:
        first = next(iter(inspect.signature(cls).parameters.values()))
        assert first.name == "x"
        assert first.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


def test_transform_call_and_resolve_forms_are_equivalent():
    latent = jnp.asarray([-1.0, 0.0, 1.0])

    direct = zdx.Exp()(latent)
    explicit = zdx.Exp().apply(latent)
    stored_call = zdx.Exp(x=latent)()
    stored_resolve = zdx.Exp(x=latent).resolve()

    assert jnp.allclose(direct, jnp.exp(latent))
    assert jnp.allclose(explicit, direct)
    assert jnp.allclose(stored_call, direct)
    assert jnp.allclose(stored_resolve, direct)
    with pytest.raises(ValueError, match="no stored x"):
        zdx.Exp().resolve()


def test_nested_transform_spine_uses_deepest_explicit_input():
    transform = zdx.Exp(x=zdx.Mul(x=zdx.Add(b=1.0), s=2.0))
    latent = jnp.asarray([-1.0, 0.5])

    value = transform(latent)

    assert jnp.allclose(value, jnp.exp(2 * (latent + 1)))
    assert "x=None" not in repr(transform)
    assert "Add(b=f32[])" in repr(transform)


def test_reversible_transform_spine_roundtrips():
    transform = zdx.Exp(x=zdx.Mul(x=zdx.Add(b=0.5), s=2.0))
    latent = jnp.asarray([-0.25, 0.5, 1.0])

    value = transform.decode(latent)
    recovered = transform.encode(value)

    assert jnp.allclose(recovered, latent)
    assert jnp.allclose(zdx.Exp(base=10.0).inv(10.0), 1.0)
    assert jnp.allclose(zdx.Log(base=10.0).inv(2.0), 100.0)


def test_map_has_fixed_scale_matmul_bias_order_and_solve():
    x = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    scale = jnp.asarray([2.0, 0.5])
    matrix = jnp.asarray([[1.0, 2.0, 0.0], [0.0, -1.0, 3.0]])
    bias = jnp.asarray([1.0, 2.0, -1.0])
    mapping = zdx.Map(s=scale, M=matrix, b=bias)

    expected = (x * scale) @ matrix + bias
    value = mapping(x)
    recovered = mapping.solve(value)

    assert jnp.allclose(value, expected)
    assert jnp.allclose(recovered, x, atol=1e-5)
    assert jnp.allclose(zdx.Map()(x), x)


def test_mask_scatter_solve_and_projection_support_batches():
    mask = jnp.asarray([[True, False], [False, True]])
    compact = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    transform = zdx.Mask(mask=mask)
    expected = jnp.asarray(
        [
            [[1.0, 0.0], [0.0, 2.0]],
            [[3.0, 0.0], [0.0, 4.0]],
        ]
    )

    full = jax.jit(lambda item, values: item.apply(values))(transform, compact)

    assert jnp.allclose(full, expected)
    assert jnp.allclose(transform.solve(full), compact)
    assert jnp.allclose(transform.project(full), full)
    assert "idx=(0, 3)" in repr(transform)
    assert isinstance(transform.indices, jax.Array)


def test_runtime_fields_are_protected_per_occurrence():
    expression = CoordinateScale(jnp.asarray(2.0))
    model = RuntimePair(ordinary=expression, local=expression)
    coords = jnp.asarray([1.0, 3.0])

    context_resolved = model.resolve(coords=coords)

    assert jnp.allclose(context_resolved.ordinary, 2 * coords)
    assert isinstance(context_resolved.local, CoordinateScale)
    assert context_resolved.local is expression

    realised = model.resolve(runtime=True, coords=coords)
    assert jnp.allclose(realised.ordinary, 2 * coords)
    assert jnp.allclose(realised.local, 2 * coords)


def test_runtime_owner_resolves_with_its_authoritative_local_context():
    stage = RuntimeStage(
        response=CoordinateScale(jnp.asarray(3.0)),
        gain=jnp.asarray(2.0),
    )
    coords = jnp.asarray([-1.0, 0.5, 2.0])

    protected = stage.resolve()
    output = jax.jit(lambda model, value: model(value))(stage, coords)

    assert isinstance(protected.response, CoordinateScale)
    assert jnp.allclose(output, 6 * coords)


def test_runtime_protected_subtree_is_not_partially_resolved():
    state = zdx.State(offset=2.0)
    expression = zdx.Add(
        x=CoordinateScale(jnp.asarray(3.0)),
        b=state.ref("offset"),
    )
    stage = RuntimeStage(response=expression, gain=1.0)

    protected = stage.resolve(state=state)

    assert isinstance(protected.response, zdx.Add)
    assert isinstance(protected.response.x, CoordinateScale)
    assert isinstance(protected.response.b, zdx.StateRef)
    realised = stage.resolve(
        runtime=True,
        state=state,
        coords=jnp.asarray([1.0, 2.0]),
    )
    assert jnp.allclose(realised.response, jnp.asarray([5.0, 8.0]))


def test_runtime_field_declaration_rejects_static_fields():
    with pytest.raises(ValueError, match="cannot also be static"):
        zdx.field(runtime=True, static=True)


def test_state_ref_is_a_deferred_expression_leaf():
    state = zdx.State(index=0, time=0.0, dt=0.1, key=jr.key(0))
    time = state.ref("time")
    scheduled = zdx.Add(x=zdx.Mul(x=time, s=2.0), b=1.0)

    assert isinstance(time, zdx.Expression)
    assert repr(time) == "StateRef(path='time')"
    assert "StateRef(path='time')" in repr(scheduled)
    assert jnp.allclose(scheduled.resolve(state=state), 1.0)

    following = scheduled.resolve(state=state.step())
    assert jnp.allclose(following, 1.2)


def test_state_step_is_jittable_and_preserves_root_key():
    state = zdx.State(index=0, time=1.0, dt=0.25, key=jr.key(17))

    stepped = jax.jit(lambda value: value.step())(state)

    assert stepped.value("index") == 1
    assert jnp.allclose(stepped.value("time"), 1.25)
    assert jnp.array_equal(
        jr.key_data(stepped.value("key")),
        jr.key_data(state.value("key")),
    )


def test_state_set_preserves_loop_carry_signature():
    state = zdx.State(index=jnp.asarray(0, dtype=jnp.int32), sample=jnp.ones(2))

    updated = state.set(index=jnp.asarray(3, dtype=jnp.int32))
    assert updated.value("index") == 3

    with pytest.raises(ValueError, match="shape, dtype, or weak type"):
        state.set(sample=jnp.ones(3))
    with pytest.raises(ValueError, match="shape, dtype, or weak type"):
        state.set(index=jnp.asarray(3.0, dtype=jnp.float32))
    with pytest.raises(ValueError, match="structure"):
        state.set(values={"replacement": jnp.asarray(1.0)})


def test_state_ref_validation_reports_missing_paths():
    model = zdx.Add(x=zdx.StateRef("missing"), b=1.0)
    state = zdx.State(time=0.0)

    with pytest.raises(ValueError, match="'missing'"):
        zdx.validate_state(model, state)


def test_normalisation_variants_use_the_whole_array_by_default():
    values = jnp.asarray([1.0, 2.0, 3.0])

    mean = zdx.MeanNorm(values).resolve()
    rms = zdx.RMSNorm(values, s=2.0).resolve()
    total = zdx.SumNorm(values).resolve()

    assert jnp.allclose(jnp.mean(mean), 1.0)
    assert jnp.allclose(jnp.sqrt(jnp.mean(jnp.abs(rms) ** 2)), 2.0)
    assert jnp.allclose(jnp.sum(total), 1.0)
    assert "axis" not in repr(zdx.MeanNorm())


def test_weighted_normalisation_and_nested_composition():
    values = jnp.asarray([1.0, 3.0, 9.0])
    weights = jnp.asarray([1.0, 1.0, 0.0])
    definition = zdx.Add(x=zdx.MeanNorm(x=values, w=weights), b=1.0)

    output = definition.resolve()

    assert jnp.allclose(output, values / 2 + 1)


def test_unit_transform_captures_global_output_and_is_reversible(restore_units):
    zdx.set_units({"cartesian": "um", "angular": "mas"})
    conversion = zdx.Unit(2.0, "mm")

    zdx.set_units({"cartesian": "m", "angular": "rad"})

    assert conversion.to == "um"
    assert jnp.allclose(conversion.resolve(), 2000.0)
    assert jnp.allclose(conversion.inv(2000.0), 2.0)
    assert jnp.allclose(zdx.Unit(unit="mm", to="m")(2.0), 0.002)


def test_grid_uses_dimension_generic_ij_indexing():
    grid = zdx.Grid(n=(2, 3), d=(2.0, 10.0), c=(1.0, -2.0))

    coords = jax.jit(lambda value: value.resolve())(grid)

    expected_first = jnp.asarray([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    expected_second = jnp.asarray([[-12.0, -2.0, 8.0], [-12.0, -2.0, 8.0]])
    assert coords.shape == (2, 2, 3)
    assert jnp.allclose(coords[0], expected_first)
    assert jnp.allclose(coords[1], expected_second)
    assert jnp.allclose(grid.measure, 20.0)


def test_grid_applies_one_unit_conversion_to_spacing_and_centre(restore_units):
    zdx.set_units({"cartesian": "um"})
    grid = zdx.Grid(n=3, d=1.0, c=2.0, unit="mm")

    coords = grid.resolve()

    assert coords.shape == (1, 3)
    assert jnp.allclose(coords, jnp.asarray([[1000.0, 2000.0, 3000.0]]))


def test_basis_infers_single_and_multi_axis_coefficient_shapes():
    matrix = jnp.arange(6.0).reshape(2, 3)
    coefficients = jnp.asarray([1.0, 2.0])
    single = zdx.Basis(M=matrix, x=coefficients).resolve()
    assert jnp.allclose(single, coefficients @ matrix)

    modes = jnp.arange(24.0).reshape(2, 3, 4)
    coefficients = jnp.arange(6.0).reshape(2, 3)
    expected = jnp.tensordot(coefficients, modes, axes=((0, 1), (0, 1)))
    assert jnp.allclose(zdx.Basis(M=modes)(coefficients), expected)


def test_polynomial_consumes_contextual_coordinates():
    coords = jnp.asarray(
        [
            [[-1.0, 0.0], [1.0, 2.0]],
            [[0.0, 1.0], [2.0, 3.0]],
        ]
    )
    coefficients = jnp.asarray([1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    polynomial = zdx.Polynomial(degree=2, ndim=2, x=coefficients)

    value = polynomial.resolve(coords=coords)

    assert jnp.allclose(value, 1 + 2 * coords[0])


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_polynomial_pairs_grid_and_coefficient_batches_in_any_dimension(ndim):
    spacing = jnp.stack((jnp.ones(ndim), 2 * jnp.ones(ndim)))
    coords = zdx.Grid(n=(2,) * ndim, d=spacing).resolve()
    coefficients = jnp.stack((jnp.arange(ndim + 1.0), 2 * jnp.arange(ndim + 1.0)))
    polynomial = zdx.Polynomial(coefficients, degree=1, ndim=ndim)

    values = jax.jit(lambda model, grid: model.resolve(coords=grid))(polynomial, coords)

    expected = jnp.stack(
        tuple(
            zdx.Polynomial(coefficients[index], p=polynomial.p).resolve(
                coords=coords[index]
            )
            for index in range(2)
        )
    )
    recovered = polynomial.solve(values, coords=coords)
    assert values.shape == (2,) + (2,) * ndim
    assert jnp.allclose(values, expected)
    assert jnp.allclose(recovered, coefficients, atol=1e-5)

    coordinate_broadcast = zdx.Polynomial(coefficients[0], p=polynomial.p).resolve(
        coords=coords
    )
    coefficient_broadcast = polynomial.resolve(coords=coords[0])
    assert coordinate_broadcast.shape == values.shape
    assert coefficient_broadcast.shape == values.shape


def test_polynomial_retains_bare_one_variable_sample_arrays():
    samples = jnp.linspace(-1.0, 1.0, 5)
    polynomial = zdx.Polynomial(jnp.asarray([1.0, 2.0]), degree=1)

    assert jnp.allclose(polynomial.resolve(coords=samples), 1 + 2 * samples)


def test_gaussian_uses_covariance_dimension_and_grid_convention():
    coords = zdx.Grid(n=(3, 3), d=1.0).resolve()
    gaussian = zdx.Gaussian(cov=jnp.eye(2))

    values = gaussian.resolve(coords=coords)

    assert values.shape == (3, 3)
    assert jnp.allclose(values[1, 1], 1.0)
    assert jnp.all(values <= 1.0)


def test_interpolation_uses_state_ref_as_ordinary_query_expression():
    state = zdx.State(time=0.5, dt=0.5)
    interpolation = zdx.Interpolation(
        knots=jnp.asarray([0.0, 1.0, 2.0]),
        values=jnp.asarray([0.0, 2.0, 0.0]),
        x=state.ref("time"),
    )

    first = interpolation.resolve(state=state)
    second = interpolation.resolve(state=state.step())

    assert jnp.allclose(first, 1.0)
    assert jnp.allclose(second, 2.0)


def test_full_numerical_definition_is_differentiable():
    definition = zdx.Exp(x=zdx.Map(x=jnp.asarray([0.2, -0.1]), s=2.0, b=0.5))

    gradient = jax.grad(lambda model: jnp.sum(model.resolve()))(definition)

    expected = 2 * jnp.exp(2 * definition.x.x + 0.5)
    assert jnp.allclose(gradient.x.x, expected)


def test_compact_prints_show_only_active_operands():
    assert repr(zdx.Exp()) == "Exp()"
    assert repr(zdx.Add(b=1.0)) == "Add(b=f32[])"
    assert repr(zdx.Map(M=jnp.eye(2))) == "Map(M=f32[2,2])"
    assert repr(zdx.Grid(n=(2, 3), d=1.0)) == "Grid(n=(2, 3), d=f32[2])"


def test_numerics_objects_keep_equinox_repr_dispatch():
    values = (
        zdx.Exp(),
        zdx.Add(b=1.0),
        zdx.Map(M=jnp.eye(2)),
        zdx.State(time=0.0),
    )

    for value in values:
        assert "__repr__" not in type(value).__dict__
        assert type(value).__repr__ is eqx.Module.__repr__
        assert repr(value) == eqx.tree_pformat(value)
