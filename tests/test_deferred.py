from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

import zodiax as zdx


class ScaleExpression(zdx.Expression):
    value: Any
    factor: Any

    def evaluate(self, **context):
        return zdx.resolve(self.value, **context) * self.factor


class AtTime(zdx.Expression):
    value: Any

    def evaluate(self, *, time, **context):
        return zdx.resolve(self.value, time=time, **context) * time


class LocalTime(zdx.Expression):
    value: Any
    time: Any

    def evaluate(self, **context):
        return zdx.resolve(self.value, time=self.time, **context)


class Reentrant(zdx.Expression):
    def evaluate(self, **context):
        return zdx.resolve(self, **context)


class Model(zdx.Module):
    reference: Any
    owner: Any


class RuntimeModel(zdx.Module):
    owner: Any
    reference: Any = zdx.field(runtime=True)


def test_resolve_nested_expressions_and_pytrees():
    value = ScaleExpression(
        value=ScaleExpression(value=jnp.asarray(2.0), factor=3.0),
        factor=4.0,
    )
    tree = {"value": value, "fixed": jnp.asarray(5.0)}

    output = zdx.resolve(tree)

    assert jnp.allclose(output["value"], 24.0)
    assert jnp.allclose(output["fixed"], 5.0)


def test_expression_cycle_is_rejected():
    with pytest.raises(ValueError, match="Expression evaluation contains a cycle"):
        zdx.resolve(Reentrant())


def test_link_reference_has_no_dynamic_leaves_and_owner_is_unique():
    owner = zdx.Linked(jnp.asarray(2.0), key="length")
    model = Model(owner.defer(), owner)

    assert jtu.tree_leaves(owner.defer()) == []
    assert len(jtu.tree_leaves(model)) == 1
    assert jtu.tree_leaves(model)[0] is owner.value


def test_links_have_stable_key_aware_prints():
    owner = zdx.Linked(jnp.asarray(2.0), key="length")
    reference = owner.defer()
    model = Model(reference, owner)

    assert type(reference).__name__ == "Deferred"
    assert repr(reference) == "Deferred(key='length')"
    assert repr(owner).startswith("Linked(key='length', value=")
    assert "Deferred(key='length')" in repr(model)
    assert "Linked(key='length', value=" in repr(model)


def test_populate_supports_forward_references_without_mutation():
    owner = zdx.Linked(jnp.asarray(2.0), key="length")
    model = Model(owner.defer(), owner)

    zdx.validate_links(model)
    output = model.populate()

    assert jnp.allclose(output.reference, 2.0)
    assert jnp.allclose(output.owner, 2.0)
    assert isinstance(model.owner, zdx.Linked)
    assert isinstance(model.reference, zdx.Deferred)


def test_populate_resolves_nested_owner_dependencies():
    base = zdx.Linked(jnp.asarray(2.0), key="base")
    derived = zdx.Linked(
        ScaleExpression(value=base.defer(), factor=3.0),
        key="derived",
    )
    tree = {
        "result": derived.defer(),
        "derived": derived,
        "base": base,
    }

    output = zdx.populate(tree)

    assert jnp.allclose(zdx.resolve(output["result"]), 6.0)
    assert jnp.allclose(zdx.resolve(output["derived"]), 6.0)
    assert jnp.allclose(output["base"], 2.0)


def test_unpopulated_deferred_raises_helpful_error():
    reference = zdx.Linked(jnp.asarray(1.0), key="orphan").defer()

    with pytest.raises(ValueError, match="has not been populated"):
        reference.resolve()


@pytest.mark.parametrize("operation", [zdx.populate, zdx.validate_links])
def test_dangling_reference_is_rejected(operation):
    reference = zdx.Linked(jnp.asarray(1.0), key="orphan").defer()

    with pytest.raises(ValueError, match="Dangling deferred link.*'orphan'"):
        operation(reference)


@pytest.mark.parametrize("operation", [zdx.populate, zdx.validate_links])
def test_duplicate_owners_are_rejected(operation):
    tree = (
        zdx.Linked(jnp.asarray(1.0), key="duplicate"),
        zdx.Linked(jnp.asarray(2.0), key="duplicate"),
    )

    with pytest.raises(ValueError, match="multiple Linked owners"):
        operation(tree)


@pytest.mark.parametrize("operation", [zdx.populate, zdx.validate_links])
def test_link_cycles_are_rejected(operation):
    a_token = zdx.Linked(None, key="a")
    b_token = zdx.Linked(None, key="b")
    a = zdx.Linked(b_token.defer(), key="a")
    b = zdx.Linked(a_token.defer(), key="b")

    with pytest.raises(ValueError, match=r"a -> b -> a"):
        operation((a, b))


def test_population_composes_with_jit_grad_and_vmap():
    owner = zdx.Linked(jnp.asarray(2.0), key="coefficient")
    model = Model(AtTime(value=owner.defer()), owner)

    def objective(raw_model, time):
        populated = raw_model.populate()
        return populated.reference.resolve(time=time) + populated.owner

    compiled = jax.jit(objective)
    assert jnp.allclose(compiled(model, jnp.asarray(3.0)), 8.0)

    gradient = jax.grad(objective)(model, jnp.asarray(3.0))
    assert jnp.allclose(gradient.owner.value, 4.0)
    assert jtu.tree_leaves(gradient.reference) == []

    times = jnp.asarray([0.0, 1.0, 2.0])
    values = jax.vmap(lambda time: objective(model, time))(times)
    assert jnp.allclose(values, jnp.asarray([2.0, 4.0, 6.0]))


def test_realise_combines_population_and_resolution_inside_transforms():
    owner = zdx.Linked(jnp.asarray(2.0), key="coefficient")
    model = Model(zdx.Mul(x=owner.defer(), s=3.0), owner)

    def objective(raw_model):
        realised = raw_model.realise()
        return realised.reference + realised.owner

    assert jnp.allclose(jax.jit(objective)(model), 8.0)
    gradient = jax.grad(objective)(model)
    assert jnp.allclose(gradient.owner.value, 4.0)
    assert jnp.allclose(zdx.realise(model).reference, 6.0)


def test_linked_definition_can_resolve_in_each_consumers_local_context():
    definition = zdx.Linked(AtTime(value=jnp.asarray(3.0)), key="time-series")
    model = (
        definition,
        LocalTime(value=definition.defer(), time=jnp.asarray(2.0)),
        LocalTime(value=definition.defer(), time=jnp.asarray(4.0)),
    )

    def objective(raw_model):
        populated = zdx.populate(raw_model)
        first = populated[1].resolve()
        second = populated[2].resolve()
        return first, second

    first, second = jax.jit(objective)(model)
    gradient = jax.grad(lambda tree: sum(objective(tree)))(model)

    assert jnp.allclose(first, 6.0)
    assert jnp.allclose(second, 12.0)
    assert jnp.allclose(gradient[0].value.value, 6.0)


def test_populate_runs_before_runtime_resolution_without_unlocking_field():
    owner = zdx.Linked(jnp.asarray(2.0), key="shared")
    model = RuntimeModel(owner=owner, reference=zdx.Mul(x=owner.defer(), s=3.0))

    populated = model.populate()
    context_resolved = populated.resolve()

    assert jnp.allclose(populated.owner, 2.0)
    assert isinstance(context_resolved.reference, zdx.Mul)
    realised = populated.resolve(runtime=True)
    assert jnp.allclose(realised.reference, 6.0)
    assert jnp.allclose(model.realise(runtime=True).reference, 6.0)


def test_explicit_link_key_survives_tree_updates():
    owner = zdx.Linked(jnp.asarray(2.0), key="semantic-length")
    updated = owner.set(value=jnp.asarray(3.0))

    assert updated.key == owner.key == "semantic-length"
    assert updated.defer().key == owner.defer().key


def test_link_keys_validate_input():
    with pytest.raises(TypeError, match="string or None"):
        zdx.Linked(1.0, key=1)
    with pytest.raises(ValueError, match="must not be empty"):
        zdx.Linked(1.0, key="")
