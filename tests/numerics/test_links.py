"""Overview §12.4: one owner, structural population, and shared gradients."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import zodiax as zdx


class Sampled(zdx.Expression):
    coefficient: jax.Array | zdx.Expression

    def evaluate(self, *, sample_coordinates, **context):
        coefficient = zdx.resolve(self.coefficient, **context)
        return coefficient * sample_coordinates


class Measurements(zdx.Module):
    """The references deliberately precede their owner's storage slot."""

    first: Any
    second: Any
    owner: Any


class LocalSamples(zdx.Expression):
    definition: zdx.Expression
    coordinates: jax.Array

    def evaluate(self, **context):
        return self.definition.resolve(sample_coordinates=self.coordinates, **context)


class FailedCalculation(zdx.Expression):
    def evaluate(self, **context):
        raise RuntimeError("numerical calculation failed")


def test_references_share_identity_without_duplicating_parameter_leaves():
    owner = zdx.Linked(2.0, key="gain")
    model = Measurements(owner.defer(), owner.defer(), owner)

    assert jax.tree.leaves(owner.defer()) == []
    leaves = jax.tree.leaves(model)
    assert len(leaves) == 1
    assert leaves[0] is owner.value
    assert model.first.key == model.second.key == owner.key

    populated = model.populate()
    assert populated.first is populated.second is populated.owner
    np.testing.assert_allclose(populated.first, 2.0)
    assert isinstance(model.first, zdx.Deferred)
    assert isinstance(model.owner, zdx.Linked)


def test_population_connects_definitions_without_preparing_their_calculations():
    owner = zdx.Linked(Sampled(zdx.Map(x=2.0, s=3.0)), key="response")
    model = Measurements(owner.defer(), owner.defer(), owner)

    populated = model.populate()

    assert isinstance(populated.first, Sampled)
    assert isinstance(populated.first.coefficient, zdx.Map)
    assert populated.first is populated.second is populated.owner

    prepared = model.resolve()
    assert isinstance(prepared.first, Sampled)
    np.testing.assert_allclose(prepared.first.coefficient, 6.0)
    realised = prepared.resolve(sample_coordinates=2.0)
    np.testing.assert_allclose(
        [realised.first, realised.second, realised.owner], [12.0, 12.0, 12.0]
    )


def test_topology_validation_does_not_require_a_successful_numerical_calculation():
    owner = zdx.Linked(FailedCalculation(), key="response")
    model = Measurements(owner.defer(), None, owner)

    assert zdx.validate_links(model) is None
    assert isinstance(model.populate().first, FailedCalculation)
    with pytest.raises(RuntimeError, match="numerical calculation failed"):
        model.resolve()


def test_owners_nested_inside_other_owned_values_are_available_to_outer_references():
    base = zdx.Linked(2.0, key="base")
    derived = zdx.Linked(zdx.Map(x=base, s=3.0), key="derived")
    model = Measurements(base.defer(), derived.defer(), derived)

    populated = model.populate()
    np.testing.assert_allclose(populated.first, 2.0)
    assert isinstance(populated.second, zdx.Map)

    realised = model.resolve()
    np.testing.assert_allclose(
        [realised.first, realised.second, realised.owner], [2.0, 6.0, 6.0]
    )


@pytest.mark.parametrize("reverse_order", [False, True])
def test_owner_dependencies_do_not_depend_on_tree_order(reverse_order):
    base = zdx.Linked(2.0, key="base")
    derived = zdx.Linked(zdx.Map(x=base.defer(), s=3.0), key="derived")
    tree = (derived.defer(), derived, base)
    if reverse_order:
        tree = tuple(reversed(tree))

    resolved = zdx.resolve(tree)

    expected = [2.0, 6.0, 6.0] if reverse_order else [6.0, 6.0, 2.0]
    np.testing.assert_allclose(resolved, expected)


def test_owned_pytrees_preserve_containers_metadata_and_array_leaves():
    payload = {"values": (jnp.array(2.0), jnp.array(3.0)), "labels": ["a", "b"]}
    owner = zdx.Linked(payload, key="pair")
    model = Measurements(owner.defer(), None, owner)

    populated = model.populate()

    assert populated.first is populated.owner
    assert isinstance(populated.first, dict)
    assert isinstance(populated.first["values"], tuple)
    assert populated.first["labels"] == ["a", "b"]
    np.testing.assert_allclose(populated.first["values"], [2.0, 3.0])
    assert owner.value["values"][0] is payload["values"][0]


def test_one_shared_definition_can_be_sampled_under_different_local_contexts():
    owner = zdx.Linked(Sampled(zdx.as_array(2.0)), key="response")
    model = Measurements(
        LocalSamples(owner.defer(), jnp.array([1.0, 2.0])),
        LocalSamples(owner.defer(), jnp.array([3.0, 5.0])),
        owner,
    )

    realised = model.resolve()

    np.testing.assert_allclose(realised.first, [2.0, 4.0])
    np.testing.assert_allclose(realised.second, [6.0, 10.0])
    assert isinstance(realised.owner, Sampled)


def test_all_resolved_uses_contribute_to_the_original_owner_gradient_under_jit():
    owner = zdx.Linked(2.0, key="gain")
    model = Measurements(
        zdx.Map(x=owner.defer(), s=3.0),
        zdx.Map(x=owner.defer(), s=4.0),
        owner,
    )

    def objective(definition):
        realised = definition.resolve()
        return realised.first + realised.second

    np.testing.assert_allclose(jax.jit(objective)(model), 14.0)
    gradient = jax.jit(jax.grad(objective))(model)
    np.testing.assert_allclose(gradient.owner.value, 7.0)
    assert jax.tree.leaves(gradient.first.x) == []
    assert jax.tree.leaves(gradient.second.x) == []


def test_outer_resolution_prepares_shared_operators_for_local_explicit_calls():
    owner = zdx.Linked(2.0, key="gain")
    model = Measurements(zdx.Map(s=owner.defer()), zdx.Map(s=owner.defer()), owner)

    def measure(definition):
        prepared = definition.resolve()
        return prepared.first(3.0) + prepared.second(4.0)

    np.testing.assert_allclose(jax.jit(measure)(model), 14.0)
    np.testing.assert_allclose(jax.grad(measure)(model).owner.value, 7.0)


def test_cross_field_operator_links_are_populated_before_explicit_application():
    operator = zdx.Map(s=zdx.Linked(2.0, key="gain"), b=zdx.Deferred("gain"))

    with pytest.raises(ValueError, match="gain"):
        operator.apply(3.0)

    np.testing.assert_allclose(operator.resolve().apply(3.0), 8.0)
    np.testing.assert_allclose(operator.populate().apply(3.0), 8.0)


def test_shared_values_combine_with_state_context_gradients_and_vmap():
    owner = zdx.Linked(2.0, key="gain")
    model = Measurements(
        Sampled(owner.defer()),
        zdx.Map(x=owner.defer(), b=zdx.StateRef("offset")),
        owner,
    )

    def objective(definition, state, coordinates):
        realised = definition.resolve(sample_coordinates=coordinates, state=state)
        return realised.first + realised.second

    state = zdx.State(offset=1.0)
    compiled = jax.jit(objective)
    np.testing.assert_allclose(compiled(model, state, jnp.array(3.0)), 9.0)
    np.testing.assert_allclose(
        compiled(model, state.set(offset=zdx.as_array(4.0)), jnp.array(3.0)), 12.0
    )
    gradient = jax.grad(objective)(model, state, jnp.array(3.0))
    np.testing.assert_allclose(gradient.owner.value, 4.0)
    values = jax.vmap(objective, in_axes=(None, None, 0))(
        model, state, jnp.array([0.0, 1.0, 2.0])
    )
    np.testing.assert_allclose(values, [3.0, 5.0, 7.0])


def test_link_identity_survives_immutable_parameter_updates():
    owner = zdx.Linked(2.0, key="gain")
    changed = owner.set(value=zdx.as_array(3.0))

    assert changed.key == owner.key == "gain"
    assert changed.defer().key == owner.defer().key
    np.testing.assert_allclose(changed.resolve(), 3.0)
    np.testing.assert_allclose(owner.resolve(), 2.0)


def test_prepopulated_tree_can_skip_population_during_resolution():
    owner = zdx.Linked(2.0, key="gain")
    model = Measurements(zdx.Map(x=owner.defer(), s=3.0), None, owner)

    realised = model.populate().resolve(populate=False)

    np.testing.assert_allclose(realised.first, 6.0)
    np.testing.assert_allclose(realised.owner, 2.0)


def test_unpopulated_detached_reference_cannot_find_an_owner_outside_its_tree():
    reference = zdx.Deferred("missing")

    with pytest.raises(ValueError, match="missing"):
        reference.resolve()
    with pytest.raises(ValueError, match="missing"):
        reference.resolve(populate=False)


@pytest.mark.parametrize("operation", [zdx.populate, zdx.validate_links, zdx.resolve])
@pytest.mark.parametrize("kind", ["missing", "duplicate", "cycle"])
def test_public_topology_checks_report_errors_and_allow_later_valid_trees(
    operation, kind
):
    if kind == "missing":
        tree = zdx.Deferred("missing")
    elif kind == "duplicate":
        tree = (zdx.Linked(1.0, key="same"), zdx.Linked(2.0, key="same"))
    else:
        tree = (
            zdx.Linked(zdx.Deferred("b"), key="a"),
            zdx.Linked(zdx.Deferred("a"), key="b"),
        )

    with pytest.raises(ValueError):
        operation(tree)

    owner = zdx.Linked(2.0, key="a")
    np.testing.assert_allclose(zdx.resolve((owner.defer(), owner)), [2.0, 2.0])


def test_repeating_the_same_owner_still_duplicates_storage():
    owner = zdx.Linked(2.0, key="gain")

    with pytest.raises(ValueError, match="gain"):
        zdx.validate_links((owner, owner))
