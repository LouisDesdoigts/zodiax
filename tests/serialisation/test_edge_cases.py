"""Public definition and reconstruction boundary contracts."""

from dataclasses import FrozenInstanceError
from io import BytesIO
import sys
from types import ModuleType

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import zodiax as zdx


class Metadata(zdx.Base):
    value: object


class Model(zdx.Module):
    value: object


class ForeignAlias(eqx.Module):
    alias: object = eqx.field(static=True)
    value: object


class Positive(eqx.Module):
    value: int

    def __init__(self, value):
        if value <= 0:
            raise ValueError("positive values only")
        self.value = value

    def doubled(self):
        return self.value * 2


def roundtrip(value, **kwargs):
    file = BytesIO()
    zdx.save(file, value)
    return zdx.load(file, **kwargs)


def array_node(**updates):
    node = {
        "kind": "jax_array",
        "shape": [2],
        "dtype": "float32",
        "weak_type": False,
    }
    node.update(updates)
    return node


def key_node(**updates):
    node = {
        "kind": "jax_prng_key",
        "shape": [],
        "data_shape": [2],
        "impl": str(jax.random.key_impl(jax.random.key(0))),
    }
    node.update(updates)
    return node


def field_node(name="value", value=None, static=False, **updates):
    node = {
        "name": name,
        "static": static,
        "value": value,
    }
    node.update(updates)
    return node


def module_node(fields=None, type_name="builtins:object", **updates):
    node = {
        "kind": "module",
        "type": type_name,
        "fields": [] if fields is None else fields,
    }
    node.update(updates)
    return node


@pytest.mark.parametrize(
    "definition",
    [
        {"kind": "list", "items": [], "extra": 1},
        {"kind": "type", "type": 1},
        {"kind": "type", "type": "invalid"},
        float("inf"),
        {},
        module_node(fields={}),
        module_node(fields=[1]),
        module_node(fields=[field_node(name="bad-name")]),
        module_node(fields=[field_node(static=1)]),
        module_node(fields=[field_node(metadata={})]),
        module_node(fields=[field_node(), field_node()]),
        {"kind": "list", "items": ()},
        {"kind": "mapping", "type": "builtins:list", "entries": []},
        {"kind": "mapping", "type": "builtins:dict", "entries": {}},
        {"kind": "mapping", "type": "builtins:dict", "entries": [1]},
        {"kind": "equinox_state", "version": 2, "entries": []},
        {"kind": "equinox_state", "version": 1, "entries": {}},
        {"kind": "equinox_state", "version": 1, "entries": [1]},
        {
            "kind": "equinox_state",
            "version": 1,
            "entries": [{"key": "", "value": None}],
        },
        {
            "kind": "equinox_state",
            "version": 1,
            "entries": [{"key": 1.0, "value": None}],
        },
        {
            "kind": "equinox_state",
            "version": 1,
            "entries": [
                {"key": 1, "value": None},
                {"key": 1, "value": None},
            ],
        },
        array_node(dtype="float8"),
        array_node(weak_type=True),
        key_node(impl=""),
        {"kind": "complex", "value": [1.0]},
        {"kind": "range", "start": 0, "stop": 1, "step": 0},
        {"kind": "unknown"},
    ],
)
def test_definition_schema_rejects_malformed_nodes(definition):
    with pytest.raises(ValueError):
        zdx.ObjectDefinition.from_dict(definition)


def test_definition_cycles_and_static_arrays_are_rejected():
    cyclic = {"kind": "list"}
    cyclic["items"] = [cyclic]
    with pytest.raises(ValueError):
        zdx.ObjectDefinition.from_dict(cyclic)
    for value in (array_node(), key_node()):
        with pytest.raises(ValueError):
            zdx.ObjectDefinition.from_dict(
                module_node(fields=[field_node(value=value, static=True)])
            )


@pytest.mark.parametrize(
    "definition",
    [
        array_node(dtype=[]),
        {"kind": "mapping", "type": [], "entries": []},
    ],
)
def test_unhashable_schema_values_raise_public_validation_error(definition):
    with pytest.raises(ValueError):
        zdx.ObjectDefinition.from_dict(definition)


def test_base_metadata_preserves_dotted_keys_without_module_path_restrictions():
    original = Metadata({"instrument.version": "v1", "value.units": "mm"})
    loaded = roundtrip(original)
    assert type(loaded) is Metadata
    assert loaded.value == original.value


def test_module_path_and_alias_checks_remain_explicit_archive_boundaries():
    original = Model({"amplitude": jnp.ones(2)}, alias={"gain": "value.amplitude"})
    invalid_paths = original.set("value", {"bad.key": jnp.ones(2)})
    stale_alias = original.set("value", {"other": jnp.ones(2)})
    for changed in (invalid_paths, stale_alias):
        with pytest.raises(ValueError):
            zdx.save(BytesIO(), changed)


def test_module_aliases_are_restored_from_archive_even_when_like_has_none():
    original = Model(jnp.ones(2), alias={"amplitude": "value"})
    loaded = roundtrip(original, like=Model(jnp.zeros(2)))
    assert loaded.alias == original.alias
    assert jnp.array_equal(loaded.amplitude, original.value)
    assert jnp.array_equal(loaded.set("amplitude", jnp.zeros(2)).value, jnp.zeros(2))


def test_foreign_static_alias_field_keeps_ordinary_template_topology():
    original = ForeignAlias(("one", "two"), jnp.ones(1))
    with pytest.raises(ValueError):
        roundtrip(original, like=ForeignAlias(("one",), jnp.zeros(1)))
    loaded = roundtrip(original, like=ForeignAlias(("old", "names"), jnp.zeros(1)))
    assert loaded.alias == original.alias


def test_reconstruction_restores_ordinary_frozen_class_without_constructor_checks():
    definition = zdx.ObjectDefinition.from_object(Positive(1)).root
    definition["fields"][0]["value"] = -2
    restored = zdx.ObjectDefinition.from_dict(definition).build_template()
    # Structural fidelity does not rerun a constructor's numerical domain checks.
    assert type(restored) is Positive
    assert restored.doubled() == -4
    with pytest.raises(FrozenInstanceError):
        restored.value = 3
    assert roundtrip(restored).value == -2


@pytest.mark.parametrize(
    "identifier",
    [
        "unimported_serialisation_provider:Model",
        "math:Missing",
        "math:pi",
    ],
)
def test_unavailable_classes_are_rejected_without_archive_directed_imports(identifier):
    with pytest.raises(ValueError):
        zdx.ObjectDefinition.from_dict(
            {"kind": "type", "type": identifier}
        ).build_template()
    assert "unimported_serialisation_provider" not in sys.modules


def test_type_resolution_does_not_invoke_module_attribute_hooks(monkeypatch):
    module = ModuleType("serialisation_lookup_fixture")

    def missing(name):
        raise AssertionError("module attribute hook ran")

    module.__getattr__ = missing
    monkeypatch.setitem(sys.modules, module.__name__, module)
    definition = {"kind": "type", "type": "serialisation_lookup_fixture:Missing"}
    with pytest.raises(ValueError):
        zdx.ObjectDefinition.from_dict(definition).build_template()


def test_uninitialised_fields_and_cyclic_objects_do_not_silently_lose_state():
    class Empty(eqx.Module):
        value: object

    with pytest.raises(TypeError):
        zdx.save(BytesIO(), object.__new__(Empty))
    cycle = []
    cycle.append(cycle)
    with pytest.raises(TypeError):
        zdx.save(BytesIO(), cycle)
