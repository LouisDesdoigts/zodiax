"""Public Module namespaces, aliases, and immutable parameter access.

These contracts describe model-visible behaviour without depending on the private
lookup algorithm, validation helpers, or exact diagnostic wording.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import zodiax as zdx


class Leaf(zdx.Module):
    value: Any


class Pair(zdx.Module):
    left: Any
    right: Any


class Holder(zdx.Module):
    child: Any


class Properties(zdx.Module):
    value: Any
    label: str = eqx.field(static=True, default="sample")

    @property
    def doubled(self):
        return 2 * self.value

    @property
    def unavailable(self):
        raise AttributeError("input not ready")


class Structure(zdx.Module):
    leaf: Any
    mapping: Any
    sequence: Any


class DirectValue(zdx.Module):
    value: Any
    child: Any


@dataclass
class PlainRecord:
    coefficient: float

    @property
    def doubled(self):
        return 2 * self.coefficient


class MissingAssignment(zdx.Module):
    value: Any

    def __init__(self):
        self.alias = None


class PrematureRead(zdx.Module):
    value: Any

    def __init__(self):
        self.value = self.value


def make_structure(alias=None):
    return Structure(
        leaf=Properties(jnp.array(1.0)),
        mapping={"sample": Leaf(jnp.array(2.0))},
        sequence=(Leaf(jnp.array(3.0)), [Leaf(jnp.array(4.0))]),
        alias=alias,
    )


def test_public_module_imports_refer_to_the_package_contract():
    from zodiax.module import Alias, Module, update, validate_aliases

    assert Module is zdx.Module
    assert Alias is zdx.Alias
    assert update is zdx.update
    assert validate_aliases is zdx.validate_aliases
    assert issubclass(Module, zdx.Base)


def test_legacy_base_wrapper_imports_refer_to_the_public_wrapper_classes():
    from zodiax.base import Base, EquinoxWrapper, WrapperHolder, build_wrapper

    assert Base is zdx.Base
    assert EquinoxWrapper is zdx.EquinoxWrapper
    assert WrapperHolder is zdx.WrapperHolder
    assert EquinoxWrapper.__module__ == WrapperHolder.__module__ == "zodiax.base"
    assert build_wrapper is zdx.build_wrapper


@pytest.mark.parametrize(
    "specification",
    [
        {"coefficient": "value"},
        ("coefficient", "value"),
        ["coefficient", "value"],
        [("coefficient", "value")],
        (("coefficient", "value"),),
    ],
)
def test_alias_input_forms_produce_the_same_static_metadata(specification):
    model = Leaf(jnp.array(2.0), alias=specification)

    assert model.alias == (("coefficient", "value"),)
    np.testing.assert_allclose(model.coefficient, 2.0)
    assert len(jax.tree.leaves(model)) == 1
    model.validate_aliases()


@pytest.mark.parametrize("specification", [None, {}, [], ()])
def test_empty_alias_inputs_use_the_same_empty_metadata(specification):
    assert Leaf(1.0, alias=specification).alias is None


def test_alias_order_does_not_change_pytree_structure_or_add_parameters():
    first = Leaf(jnp.array(2.0), alias={"beta": "value", "alpha": "value"})
    second = Leaf(jnp.array(2.0), alias=[("alpha", "value"), ("beta", "value")])

    assert first.alias == second.alias == (("alpha", "value"), ("beta", "value"))
    assert jax.tree.structure(first) == jax.tree.structure(second)
    assert len(jax.tree.leaves(first)) == 1
    changed = jax.tree.map(lambda value: value + 1, first)
    assert changed.alias == first.alias
    np.testing.assert_allclose(changed.alpha, 3.0)
    np.testing.assert_allclose(changed.beta, 3.0)


def test_structural_aliases_cross_module_mapping_and_sequence_children():
    model = make_structure(
        alias={
            "first": "leaf.value",
            "second": "mapping.sample.value",
            "third": "sequence.0.value",
            "fourth": "sequence.1.0.value",
        }
    )

    np.testing.assert_allclose(
        model.get(["first", "second", "third", "fourth"]), [1, 2, 3, 4]
    )
    changed = model.set(second=jnp.array(5.0), fourth=jnp.array(6.0))
    np.testing.assert_allclose(changed.mapping["sample"].value, 5.0)
    np.testing.assert_allclose(changed.sequence[1][0].value, 6.0)
    np.testing.assert_allclose(model.second, 2.0)
    np.testing.assert_allclose(model.fourth, 4.0)


def test_alias_validation_does_not_create_missing_mapping_entries():
    values = defaultdict(lambda: jnp.array(0.0), {"present": jnp.array(1.0)})

    with pytest.raises(ValueError):
        Holder(values, alias={"coefficient": "child.missing"})

    assert list(values) == ["present"]


def test_raised_alias_get_set_and_parameter_gradients_work_under_jit():
    model = Pair(
        Leaf(jnp.array(2.0), alias={"coefficient": "value"}),
        Leaf(jnp.array(3.0)),
    )

    np.testing.assert_allclose(model.get("coefficient", to_array=False), 2.0)
    gradient = eqx.filter_jit(jax.grad(lambda tree: tree.coefficient**2))(model)
    np.testing.assert_allclose(gradient.left.value, 4.0)
    np.testing.assert_allclose(gradient.right.value, 0.0)
    updated = eqx.filter_jit(lambda tree, value: tree.set(coefficient=value))(
        model, jnp.array(5.0)
    )
    np.testing.assert_allclose(updated.coefficient, 5.0)
    np.testing.assert_allclose(model.coefficient, 2.0)
    assert len(jax.tree.leaves(updated)) == 2


def test_wrapper_holder_path_operations_keep_aliases_despite_attribute_delegation():
    values, structure = zdx.build_wrapper({"coefficient": jnp.array([1.0, 2.0])})
    holder = zdx.WrapperHolder(values, structure, alias={"latent": "values"})

    np.testing.assert_allclose(holder.get("latent", to_array=False), [1.0, 2.0])
    changed = holder.set(latent=jnp.array([3.0, 4.0]))

    np.testing.assert_allclose(changed.get("latent", to_array=False), [3.0, 4.0])
    np.testing.assert_allclose(changed.build["coefficient"], [3.0, 4.0])
    np.testing.assert_allclose(holder.build["coefficient"], [1.0, 2.0])
    assert changed.alias == holder.alias


def test_direct_fields_and_local_aliases_precede_raised_descendants():
    direct = DirectValue(value=1.0, child=Leaf(2.0))
    np.testing.assert_allclose(direct.value, 1.0)

    aliased = Pair(Leaf(2.0), Leaf(3.0), alias={"value": "right.value"})
    np.testing.assert_allclose(aliased.value, 3.0)
    changed = aliased.set(value=4.0)
    np.testing.assert_allclose(changed.right.value, 4.0)
    np.testing.assert_allclose(changed.left.value, 2.0)


def test_raised_lookup_prefers_the_nearest_matching_owner():
    deep = Leaf(1.0, alias={"coefficient": "value"})
    near = Leaf(2.0, alias={"coefficient": "value"})
    model = Pair(Pair(deep, None), near)

    np.testing.assert_allclose(model.coefficient, 2.0)
    changed = model.set(coefficient=3.0)
    np.testing.assert_allclose(changed.right.value, 3.0)
    np.testing.assert_allclose(changed.left.left.value, 1.0)


@pytest.mark.parametrize("repeat_same_object", [False, True])
def test_equal_depth_slots_are_ambiguous_even_when_they_hold_the_same_object(
    repeat_same_object,
):
    left = Leaf(jnp.array(2.0), alias={"coefficient": "value"})
    right = (
        left
        if repeat_same_object
        else Leaf(jnp.array(3.0), alias={"coefficient": "value"})
    )
    model = Pair(left, right)

    for operation in (
        lambda: model.coefficient,
        lambda: model.get("coefficient", to_array=False),
        lambda: model.set(coefficient=jnp.array(5.0)),
    ):
        with pytest.raises(AttributeError) as error:
            operation()
        assert "left" in str(error.value)
        assert "right" in str(error.value)

    changed = model.set("left.coefficient", jnp.array(5.0))
    np.testing.assert_allclose(changed.left.value, 5.0)
    np.testing.assert_allclose(changed.right.value, right.value)
    np.testing.assert_allclose(model.left.value, 2.0)


def test_mapping_keys_and_sequence_children_are_available_through_qualified_paths():
    model = make_structure()

    assert model.sample is model.mapping["sample"]
    np.testing.assert_allclose(model.get("sample.value", to_array=False), 2.0)
    np.testing.assert_allclose(model.get("sequence.1.0.value", to_array=False), 4.0)
    changed = model.set("sequence.0.value", jnp.array(8.0))
    np.testing.assert_allclose(changed.sequence[0].value, 8.0)
    np.testing.assert_allclose(model.sequence[0].value, 3.0)


def test_properties_and_static_metadata_are_readable_without_becoming_alias_targets():
    model = Holder(Properties(jnp.array(2.0)))

    np.testing.assert_allclose(model.doubled, 4.0)
    np.testing.assert_allclose(model.get("child.doubled", to_array=False), 4.0)
    assert model.label == "sample"
    assert model.get("child.label", to_array=False) == "sample"

    for path in ("child.doubled", "child.label"):
        with pytest.raises(ValueError):
            Holder(Properties(jnp.array(2.0)), alias={"parameter": path})


def test_plain_dataclass_fields_and_properties_participate_in_raised_reads():
    model = Holder({"records": [PlainRecord(2.0)]})

    np.testing.assert_allclose(model.coefficient, 2.0)
    np.testing.assert_allclose(model.doubled, 4.0)
    np.testing.assert_allclose(
        model.get("child.records.0.coefficient", to_array=False), 2.0
    )


@pytest.mark.parametrize("operation", ["replace", "populate", "resolve"])
def test_topology_changes_preserve_alias_metadata_until_explicit_validation(operation):
    owner = zdx.Linked(zdx.Map(x=2.0, s=3.0), key="amplitude")
    model = Holder(owner, alias={"latent": "child.value.x"})
    model.validate_aliases()

    if operation == "replace":
        changed = model.set(child=jnp.array(6.0))
    else:
        changed = getattr(model, operation)()

    assert changed.alias == model.alias
    np.testing.assert_allclose(model.latent, 2.0)
    with pytest.raises(AttributeError):
        _ = changed.latent
    with pytest.raises(ValueError):
        changed.validate_aliases()
    with pytest.raises(ValueError):
        zdx.validate_aliases({"nested": [changed]})


@pytest.mark.parametrize(
    "specification",
    [
        1,
        ("name",),
        (("name", 1),),
        ("bad name", "value"),
        ("_private", "value"),
        ("class", "value"),
        ("name", ""),
        ("name", "child..value"),
        (("name", "value"), ("name", "value")),
    ],
)
def test_alias_input_rejects_invalid_names_paths_and_duplicate_names(specification):
    with pytest.raises((TypeError, ValueError)):
        Leaf(1.0, alias=specification)


@pytest.mark.parametrize("name", ["value", "doubled", "get"])
def test_alias_names_cannot_shadow_declared_fields_properties_or_methods(name):
    with pytest.raises(ValueError):
        Properties(1.0, alias={name: "value"})


@pytest.mark.parametrize(
    "path",
    [
        "missing",
        "leaf.value.more",
        "mapping.missing.value",
        "sequence.-1.value",
        "sequence.01.value",
        "sequence.2.value",
        "sequence.name.value",
    ],
)
def test_alias_targets_must_be_available_canonical_structural_paths(path):
    with pytest.raises(ValueError):
        make_structure(alias={"parameter": path})


def test_aliases_cannot_chain_through_other_aliases_or_raised_names():
    with pytest.raises(ValueError):
        Leaf(1.0, alias={"first": "value", "second": "first"})
    child = Leaf(1.0, alias={"coefficient": "value"})
    with pytest.raises(ValueError):
        Holder(child, alias={"parameter": "child.coefficient"})
    with pytest.raises(ValueError):
        Holder(child, alias={"parameter": "value"})


def test_incomplete_user_constructors_fail_without_recursive_lookup():
    with pytest.raises(TypeError):
        MissingAssignment()
    with pytest.raises(AttributeError):
        PrematureRead()


def test_unavailable_properties_and_missing_names_raise_attribute_errors():
    model = Properties(1.0)
    with pytest.raises(AttributeError):
        _ = model.unavailable
    with pytest.raises(AttributeError):
        _ = Holder(model).unavailable
    with pytest.raises(AttributeError):
        _ = Holder(model).not_present
