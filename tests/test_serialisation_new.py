"""Focused tests for generic Zodiax object serialisation."""

from io import BytesIO
import json
import zipfile

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
import zodiax as zdx

from zodiax.serialisation import ObjectDefinition, load, save


class ExampleModule(eqx.Module):
    """Small non-Zodiax-specific module used by archive tests."""

    array: object
    count: int
    mode: str = eqx.field(static=True)


class ConstructorModule(eqx.Module):
    """Module whose constructor calls are observable."""

    calls = 0
    array: object
    label: str

    def __init__(self, array, label):
        type(self).calls += 1
        self.array = array
        self.label = label


class CallableModule(eqx.Module):
    function: object = eqx.field(static=True)


class RuntimeModel(zdx.Module):
    response: object = zdx.field(runtime=True)


def _roundtrip(original, **kwargs):
    file = BytesIO()
    save(file, original)
    file.seek(0)
    return load(file, **kwargs)


def test_path_roundtrip_uses_zdx_suffix_and_manifest_branding(tmp_path):
    original = ExampleModule(jnp.arange(4.0).reshape(2, 2), 3, "science")
    path = tmp_path / "model"

    save(path, original)
    archive_path = path.with_suffix(".zdx")
    loaded = load(path)

    assert archive_path.is_file()
    assert eqx.tree_equal(loaded, original, typematch=True)
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["format"] == {"name": "zodiax", "version": 1}
    assert set(manifest["created_with"]["packages"]) == {
        "equinox",
        "interpax",
        "jax",
        "jaxlib",
        "numpy",
        "zodiax",
    }


def test_object_definition_builds_abstract_constructor_free_template():
    ConstructorModule.calls = 0
    original = ConstructorModule(jnp.arange(3.0), "archived")
    assert ConstructorModule.calls == 1

    definition = ObjectDefinition.from_object(original)
    template = definition.build_template()

    assert ConstructorModule.calls == 1
    assert type(template) is ConstructorModule
    assert isinstance(template.array, jax.ShapeDtypeStruct)
    assert template.array.shape == (3,)
    assert template.label == "archived"


def test_load_does_not_run_module_constructor():
    ConstructorModule.calls = 0
    original = ConstructorModule(jnp.arange(3.0), "archived")

    loaded = _roundtrip(original)

    assert ConstructorModule.calls == 1
    assert eqx.tree_equal(loaded, original, typematch=True)


def test_modern_jax_prng_key_roundtrips():
    original = {"key": jr.key(42), "array": jnp.arange(2, dtype=jnp.float32)}

    loaded = _roundtrip(original)

    assert jnp.array_equal(jr.key_data(loaded["key"]), jr.key_data(original["key"]))
    assert str(jr.key_impl(loaded["key"])) == str(jr.key_impl(original["key"]))
    assert jnp.array_equal(loaded["array"], original["array"])


def test_like_validates_structure_but_archived_values_win():
    original = ExampleModule(jnp.arange(3.0), 7, "archived")
    like = ExampleModule(jnp.zeros(3), -1, "template")

    loaded = _roundtrip(original, like=like)

    assert eqx.tree_equal(loaded, original, typematch=True)


def test_local_module_can_be_loaded_with_like():
    class LocalModule(eqx.Module):
        array: object

    original = LocalModule(jnp.arange(3.0))
    file = BytesIO()
    save(file, original)

    file.seek(0)
    with pytest.raises(ValueError, match="Local type.*Supply like"):
        load(file)

    file.seek(0)
    loaded = load(file, like=LocalModule(jnp.zeros(3)))
    assert eqx.tree_equal(loaded, original, typematch=True)


def test_callable_metadata_is_rejected():
    with pytest.raises(TypeError, match="unsupported callable"):
        ObjectDefinition.from_object(CallableModule(lambda value: value))


def test_tree_layout_and_values_share_the_generated_object_definition():
    original = zdx.TreeVector.from_tree(
        {
            "position": jnp.asarray([1.0, 2.0], dtype=jnp.float32),
            "flux": jnp.asarray(3.0, dtype=jnp.float32),
        }
    )

    loaded = _roundtrip(original)

    assert loaded.layout.compatible(original.layout)
    assert jnp.array_equal(loaded.vector, original.vector)


def test_mask_indices_are_one_array_payload_not_static_tree_metadata():
    original = zdx.Mask(mask=jnp.eye(64, dtype=bool))
    definition = ObjectDefinition.from_object(original).root
    indices = next(
        field["value"] for field in definition["fields"] if field["name"] == "indices"
    )

    loaded = _roundtrip(original)

    assert indices == {
        "kind": "jax_array",
        "shape": [64],
        "dtype": "int32",
        "weak_type": False,
    }
    assert jnp.array_equal(loaded.indices, original.indices)
    assert jnp.allclose(loaded.apply(jnp.ones(64)), jnp.eye(64))


def test_runtime_field_semantics_survive_archive_roundtrip():
    original = RuntimeModel(response=zdx.Gaussian(cov=jnp.eye(2)))
    definition = ObjectDefinition.from_object(original)
    response = next(
        field for field in definition.root["fields"] if field["name"] == "response"
    )
    loaded = _roundtrip(original)
    coords = zdx.Grid(n=(3, 3), d=1.0).resolve()

    protected = loaded.resolve(coords=coords)
    realised = loaded.resolve(runtime=True, coords=coords)

    assert response["metadata"] == {"runtime": True}
    assert isinstance(protected.response, zdx.Gaussian)
    assert realised.response.shape == (3, 3)
    assert jnp.allclose(realised.response[1, 1], 1.0)


def test_runtime_field_metadata_is_part_of_the_installed_class_schema():
    original = RuntimeModel(response=zdx.Gaussian(cov=jnp.eye(2)))
    definition = ObjectDefinition.from_object(original).to_dict()
    response = next(
        field for field in definition["fields"] if field["name"] == "response"
    )
    response["metadata"]["runtime"] = False

    changed = ObjectDefinition.from_dict(definition)

    with pytest.raises(ValueError, match="installed class schema.*changed"):
        changed.build_template()


def test_unrealised_linked_model_is_the_serialised_lifecycle_form():
    owner = zdx.Linked(jnp.asarray(2.0), key="shared")
    original = {
        "owner": owner,
        "use": zdx.Mul(x=owner.defer(), s=3.0),
    }

    loaded = _roundtrip(original)
    realised = zdx.resolve(zdx.populate(loaded))

    assert isinstance(loaded["owner"], zdx.Linked)
    assert isinstance(loaded["use"].x, zdx.Deferred)
    assert jnp.allclose(realised["owner"], 2.0)
    assert jnp.allclose(realised["use"], 6.0)
