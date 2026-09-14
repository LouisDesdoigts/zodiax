"""Overview §12.2: downstream expressions and staged scientific context.

The small classes below are implementation examples. Assertions concern their
values, retained definitions, and JAX behaviour rather than resolver internals.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import zodiax as zdx


class Sampled(zdx.Expression):
    """Prepare a coefficient now and sample it when coordinates arrive."""

    coefficient: jax.Array | zdx.Expression

    def evaluate(self, *, sample_coordinates, **context):
        coefficient = zdx.resolve(self.coefficient, **context)
        return coefficient * sample_coordinates


class GaussianProfile(zdx.Expression):
    """The overview's unit-peak Gaussian with a positive scalar scale."""

    scale: jax.Array | zdx.Expression

    def __init__(self, scale):
        self.alias = None
        self.scale = zdx.as_array(scale)

    def evaluate(self, *, coordinates, **context):
        scale = zdx.resolve(self.scale, **context)
        return jnp.exp(-0.5 * (coordinates / scale) ** 2)


class LinearBasis(zdx.Expression):
    def evaluate(self, *, basis_coordinates, **context):
        return jnp.stack([jnp.ones_like(basis_coordinates), basis_coordinates])


class BasisExpansion(zdx.Expression):
    """Resolve coefficient and basis definitions independently."""

    coefficients: jax.Array | zdx.Expression
    basis: jax.Array | zdx.Expression

    def evaluate(self, **context):
        coefficients = zdx.resolve(self.coefficients, **context)
        basis = zdx.resolve(self.basis, **context)
        return zdx.matmul(coefficients, basis)


class Model(zdx.Module):
    value: Any
    other: Any = None


class Shifted(zdx.Expression):
    """Incoming coordinates and derived sample coordinates have distinct names."""

    child: Sampled
    shift: jax.Array | zdx.Expression

    def evaluate(self, *, coordinates, **context):
        shift = zdx.resolve(self.shift, **context)
        return self.child.resolve(sample_coordinates=coordinates - shift, **context)


class ContextScale(zdx.Transform):
    """A field and fwd suffice: Equinox generates the constructor."""

    gain: Any = 1.0

    def fwd(self, x, *, offset, **context):
        gain = zdx.resolve(self.gain, **context)
        return gain * x + offset


class DefaultOffset(zdx.Transform):
    def fwd(self, x, *, offset=2.0, **context):
        return x + offset


class ApplyOperator(zdx.Expression):
    operator: zdx.Transform
    value: jax.Array
    unused: zdx.Expression

    def evaluate(self, **context):
        return self.operator.apply(self.value, **context)


class ReturnModel(zdx.Expression):
    model: Model

    def evaluate(self, **context):
        return self.model.resolve(**context)


class SampleModel(zdx.Expression):
    """An ordinary Module can remain partial inside another calculation."""

    model: Model

    def evaluate(self, *, coordinates, **context):
        prepared = self.model.resolve(**context)
        samples = prepared.value.resolve(sample_coordinates=coordinates, **context)
        return samples + prepared.other


class ReturnDefinitions(zdx.Expression):
    value: jax.Array

    def evaluate(self, **context):
        return {
            "values": [zdx.Map(x=self.value, b=1.0)],
            "samples": (Sampled(self.value),),
            "model": Model(zdx.Map(x=self.value, s=2.0)),
        }


class TwoSamples(zdx.Expression):
    """Reuse one definition with two different sets of sampling coordinates."""

    child: Sampled

    def evaluate(self, *, first_coordinates, second_coordinates, **context):
        first = self.child.resolve(sample_coordinates=first_coordinates, **context)
        second = self.child.resolve(sample_coordinates=second_coordinates, **context)
        return first, second


class CheckedCalculation(zdx.Expression):
    value: Any

    def evaluate(self, *, error_type=None, **context):
        if error_type is not None:
            raise error_type("calculation failed")
        return zdx.resolve(self.value, **context)


class RecursiveCalculation(zdx.Expression):
    value: jax.Array

    def evaluate(self, *, recursive=False, **context):
        if recursive:
            return self.resolve(recursive=True, **context)
        return self.value


class CompiledThenDirect(zdx.Expression):
    """A nested JAX calculation must not leave temporary tracers for later use."""

    definition: zdx.Expression

    def evaluate(self, **context):
        compiled = jax.jit(lambda: self.definition.resolve(**context))()
        direct = self.definition.resolve(**context)
        return compiled + direct


def test_missing_context_keeps_module_type_and_prepares_independent_fields():
    model = Model(Sampled(zdx.Map(x=2.0, s=3.0)), zdx.Map(x=1.0, b=4.0))

    prepared = model.resolve()

    assert type(prepared) is Model
    assert isinstance(prepared.value, Sampled)
    np.testing.assert_allclose(prepared.value.coefficient, 6.0)
    np.testing.assert_allclose(prepared.other, 5.0)
    assert isinstance(model.value.coefficient, zdx.Map)
    assert isinstance(model.other, zdx.Map)

    realised = prepared.resolve(sample_coordinates=jnp.array([1.0, 2.0]))
    assert type(realised) is Model
    np.testing.assert_allclose(realised.value, [6.0, 12.0])


def test_gaussian_prepares_scale_before_sampling_and_preserves_parameter_gradients():
    definition = GaussianProfile(zdx.Map(x=0.5, s=2.0))
    coordinates = jnp.array([-1.0, 0.0, 1.0])

    prepared = definition.resolve()

    assert isinstance(prepared, GaussianProfile)
    np.testing.assert_allclose(prepared.scale, 1.0)
    compiled = eqx.filter_jit(lambda model: model.resolve(coordinates=coordinates))
    np.testing.assert_allclose(
        compiled(prepared), [np.exp(-0.5), 1.0, np.exp(-0.5)], rtol=1e-6
    )
    gradient = jax.grad(lambda model: compiled(model).sum())(definition)
    # Two off-centre samples each contribute exp(-1/2), and Map scales by 2.
    np.testing.assert_allclose(gradient.scale.x, 4 * np.exp(-0.5), rtol=1e-6)


def test_basis_coefficients_prepare_before_sampling_and_remain_differentiable():
    definition = BasisExpansion(zdx.Map(x=[1.0, 2.0], s=2.0, b=1.0), LinearBasis())
    coordinates = jnp.array([-1.0, 0.0, 1.0])

    prepared = definition.resolve()

    assert isinstance(prepared, BasisExpansion)
    assert isinstance(prepared.basis, LinearBasis)
    np.testing.assert_allclose(prepared.coefficients, [3.0, 5.0])
    assert isinstance(definition.coefficients, zdx.Map)
    compiled = eqx.filter_jit(
        lambda model: model.resolve(basis_coordinates=coordinates)
    )
    np.testing.assert_allclose(compiled(prepared), [-2.0, 3.0, 8.0])
    np.testing.assert_allclose(compiled(definition), [-2.0, 3.0, 8.0])

    # Weights sum to 6 and their coordinate-weighted sum is 2; Map scales by 2.
    weights = jnp.array([1.0, 2.0, 3.0])
    gradient = jax.grad(lambda model: jnp.vdot(weights, compiled(model)))(definition)
    np.testing.assert_allclose(gradient.coefficients.x, [12.0, 4.0])


@pytest.mark.parametrize("mapped_coefficients", [False, True])
def test_basis_expansion_also_accepts_prepared_arrays(mapped_coefficients):
    coefficients = (
        zdx.Map(x=[1.0, 2.0], s=2.0, b=1.0)
        if mapped_coefficients
        else zdx.as_array([3.0, 5.0])
    )
    definition = BasisExpansion(
        coefficients, zdx.as_array([[1.0, 1.0, 1.0], [-1.0, 0.0, 1.0]])
    )

    np.testing.assert_allclose(definition.resolve(), [-2.0, 3.0, 8.0])


def test_unresolved_operand_keeps_parent_and_prepares_available_siblings():
    expression = zdx.Map(x=Sampled(zdx.Map(x=2.0, s=3.0)), b=zdx.StateRef("offset"))

    prepared = expression.resolve(state=zdx.State(offset=1.0))

    assert isinstance(prepared, zdx.Map)
    assert isinstance(prepared.x, Sampled)
    np.testing.assert_allclose(prepared.x.coefficient, 6.0)
    np.testing.assert_allclose(prepared.b, 1.0)
    np.testing.assert_allclose(
        prepared.resolve(sample_coordinates=jnp.array([1.0, 2.0])), [7.0, 13.0]
    )


def test_resolve_children_prepares_fields_without_computing_the_owner():
    definition = zdx.Map(x=zdx.Map(x=2.0, s=3.0), b=1.0)

    prepared = definition.resolve_children()

    assert isinstance(prepared, zdx.Map)
    np.testing.assert_allclose(prepared.x, 6.0)
    np.testing.assert_allclose(prepared.resolve(), 7.0)
    assert isinstance(definition.x, zdx.Map)


def test_nested_module_request_can_return_a_partial_model_for_later_calls():
    model = Model(Sampled(zdx.Map(x=2.0, s=3.0)), zdx.Map(x=1.0, b=4.0))

    prepared = ReturnModel(model).resolve()

    assert type(prepared) is Model
    assert isinstance(prepared.value, Sampled)
    np.testing.assert_allclose(prepared.value.coefficient, 6.0)
    np.testing.assert_allclose(prepared.other, 5.0)
    np.testing.assert_allclose(SampleModel(model).resolve(coordinates=2.0), 17.0)


def test_returned_definitions_are_resolved_recursively_inside_containers():
    definition = ReturnDefinitions(zdx.as_array(2.0))

    result = definition.resolve(sample_coordinates=jnp.array([3.0, 4.0]))

    assert isinstance(result, dict)
    assert isinstance(result["values"], list)
    assert isinstance(result["samples"], tuple)
    assert type(result["model"]) is Model
    np.testing.assert_allclose(result["values"][0], 3.0)
    np.testing.assert_allclose(result["samples"][0], [6.0, 8.0])
    np.testing.assert_allclose(result["model"].value, 4.0)


@pytest.mark.parametrize("stored_input", [None, 100.0])
def test_evaluate_uses_its_operator_before_child_preparation(stored_input):
    expression = ApplyOperator(
        zdx.Map(x=stored_input, s=2.0), zdx.as_array(3.0), zdx.StateRef("unused")
    )

    # Preparing children first would replace a bound operator with its array,
    # or try to read the deliberately unused state reference.
    np.testing.assert_allclose(expression.resolve(state=zdx.State()), 6.0)


def test_distinct_context_names_allow_parent_control_of_local_coordinates():
    expression = Shifted(Sampled(zdx.Map(x=2.0, s=3.0)), zdx.StateRef("shift"))
    coordinates = jnp.array([3.0, 5.0])

    partial = expression.resolve(coordinates=coordinates)

    assert isinstance(partial, Shifted)
    assert isinstance(partial.child, Sampled)
    np.testing.assert_allclose(partial.child.coefficient, 6.0)
    state = zdx.State(shift=1.0)
    compiled = eqx.filter_jit(
        lambda model, coords: model.resolve(coordinates=coords, state=state)
    )
    np.testing.assert_allclose(compiled(partial, coordinates), [12.0, 24.0])
    np.testing.assert_allclose(
        expression(coordinates=coordinates, state=state), [12.0, 24.0]
    )


def test_reused_expression_reads_each_context_and_each_outer_call_independently():
    definition = TwoSamples(Sampled(zdx.as_array(2.0)))
    first_coordinates = jnp.array([-1.0, 1.0])
    second_coordinates = jnp.array([2.0, 3.0])

    first, second = definition.resolve(
        first_coordinates=first_coordinates, second_coordinates=second_coordinates
    )

    np.testing.assert_allclose(first, [-2.0, 2.0])
    np.testing.assert_allclose(second, [4.0, 6.0])
    changed = definition.resolve(
        first_coordinates=second_coordinates, second_coordinates=first_coordinates
    )
    np.testing.assert_allclose(changed[0], [4.0, 6.0])
    np.testing.assert_allclose(changed[1], [-2.0, 2.0])


def test_required_fwd_context_and_declared_fields_use_the_generated_constructor():
    transform = ContextScale(x=zdx.Map(x=2.0, s=3.0), gain=zdx.Map(x=2.0, b=1.0))

    partial = transform.resolve()

    assert isinstance(partial, ContextScale)
    np.testing.assert_allclose(partial.x, 6.0)
    np.testing.assert_allclose(partial.gain, 3.0)
    np.testing.assert_allclose(partial.resolve(offset=4.0), 22.0)
    np.testing.assert_allclose(ContextScale(gain=2.0).apply(3.0, offset=4.0), 10.0)


@pytest.mark.parametrize(("context", "expected"), [({}, 5.0), ({"offset": 4.0}, 7.0)])
def test_default_fwd_context_is_optional(context, expected):
    np.testing.assert_allclose(DefaultOffset(x=3.0).resolve(**context), expected)


@pytest.mark.parametrize("error_type", [ValueError, TypeError, RuntimeError])
def test_calculation_errors_propagate_without_poisoning_later_resolution(error_type):
    definition = CheckedCalculation(zdx.Map(x=2.0, b=3.0))

    with pytest.raises(error_type, match="calculation failed"):
        definition.resolve(error_type=error_type)

    np.testing.assert_allclose(definition.resolve(), 5.0)
    assert isinstance(Sampled(zdx.as_array(2.0)).resolve(), Sampled)


def test_self_dependency_reports_a_cycle_and_restores_later_resolution():
    definition = RecursiveCalculation(zdx.as_array(3.0))

    with pytest.raises(ValueError):
        definition.resolve(recursive=True)

    np.testing.assert_allclose(definition.resolve(), 3.0)
    assert isinstance(Sampled(zdx.as_array(2.0)).resolve(), Sampled)


def test_nested_jax_calculation_does_not_leak_tracers_between_evaluations():
    def calculate(value):
        return CompiledThenDirect(zdx.Map(x=value, s=3.0)).resolve()

    compiled = jax.jit(calculate)
    np.testing.assert_allclose(compiled(jnp.array(2.0)), 12.0)
    np.testing.assert_allclose(compiled(jnp.array(4.0)), 24.0)
    np.testing.assert_allclose(jax.grad(calculate)(jnp.array(2.0)), 6.0)
    np.testing.assert_allclose(jax.vmap(calculate)(jnp.array([1.0, 2.0])), [6.0, 12.0])


def test_contextual_sampling_supports_gradients_and_vmap():
    def calculate(coefficient, coordinates):
        return Shifted(Sampled(coefficient), jnp.array(1.0)).resolve(
            coordinates=coordinates
        )

    gradient = jax.grad(calculate)(jnp.array(2.0), jnp.array(3.0))
    samples = jax.vmap(calculate, in_axes=(None, 0))(
        jnp.array(2.0), jnp.array([1.0, 3.0, 5.0])
    )

    np.testing.assert_allclose(gradient, 2.0)
    np.testing.assert_allclose(samples, [0.0, 4.0, 8.0])
