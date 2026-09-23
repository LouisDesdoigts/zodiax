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


class CoordinateModel(zdx.Module):
    response: object


class CoordinateExpression(zdx.Expression):
    scale: object

    def evaluate(self, *, coords, **context):
        return self.scale * coords


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
    with pytest.raises(ValueError):
        load(file)

    file.seek(0)
    loaded = load(file, like=LocalModule(jnp.zeros(3)))
    assert eqx.tree_equal(loaded, original, typematch=True)


def test_callable_metadata_is_rejected():
    with pytest.raises(TypeError):
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


def test_mask_array_and_selected_count_roundtrip():
    original = zdx.Mask(mask=jnp.eye(64, dtype=bool))
    definition = ObjectDefinition.from_object(original).root
    mask = next(
        field["value"] for field in definition["fields"] if field["name"] == "mask"
    )

    loaded = _roundtrip(original)

    assert mask == {
        "kind": "jax_array",
        "shape": [64, 64],
        "dtype": "bool",
        "weak_type": False,
    }
    assert jnp.array_equal(loaded.mask, original.mask)
    assert loaded.size == original.size == 64
    assert jnp.allclose(loaded.apply(jnp.ones(64)), jnp.eye(64))


def test_context_signature_semantics_survive_archive_roundtrip():
    original = CoordinateModel(
        response=CoordinateExpression(jnp.asarray(2.0, dtype=jnp.float32))
    )
    definition = ObjectDefinition.from_object(original)
    response = next(
        field for field in definition.root["fields"] if field["name"] == "response"
    )
    loaded = _roundtrip(original)
    coords = jnp.asarray([1.0, 3.0])

    protected = loaded.resolve()
    realised = loaded.resolve(coords=coords)

    assert set(response) == {"name", "static", "value"}
    assert isinstance(protected.response, CoordinateExpression)
    assert jnp.allclose(realised.response, 2 * coords)


def test_installed_field_schema_excludes_unknown_metadata():
    original = CoordinateModel(
        response=CoordinateExpression(jnp.asarray(2.0, dtype=jnp.float32))
    )
    definition = ObjectDefinition.from_object(original).to_dict()
    response = next(
        field for field in definition["fields"] if field["name"] == "response"
    )
    response["metadata"] = {"local": True}

    with pytest.raises(ValueError):
        ObjectDefinition.from_dict(definition)


def test_unrealised_linked_model_is_the_serialised_lifecycle_form():
    owner = zdx.Linked(jnp.asarray(2.0), key="shared")
    original = {
        "owner": owner,
        "use": zdx.Map(x=owner.defer(), s=3.0),
    }

    loaded = _roundtrip(original)
    realised = zdx.resolve(loaded)

    assert isinstance(loaded["owner"], zdx.Linked)
    assert isinstance(loaded["use"].x, zdx.Deferred)
    assert jnp.allclose(realised["owner"], 2.0)
    assert jnp.allclose(realised["use"], 6.0)


def test_loaded_links_keep_shared_gradient_accumulation():
    owner = zdx.Linked(2.0, key="gain")
    loaded = _roundtrip({"owner": owner, "use": zdx.Map(x=owner.defer(), s=3.0)})

    def loss(model):
        values = zdx.resolve(model)
        return values["owner"] + values["use"]

    gradient = eqx.filter_grad(loss)(loaded)
    assert jnp.allclose(gradient["owner"].value, 4.0)
    assert isinstance(loaded["use"].x, zdx.Deferred)


@pytest.mark.parametrize("kind", ["dangling", "duplicate", "cycle"])
def test_archive_requires_complete_link_topology(kind):
    owner = zdx.Linked(1.0, key="shared")
    if kind == "dangling":
        model = owner.defer()
    elif kind == "duplicate":
        model = [owner, owner]
    else:
        model = [
            zdx.Linked(zdx.Deferred("b"), key="a"),
            zdx.Linked(zdx.Deferred("a"), key="b"),
        ]
    with pytest.raises(ValueError):
        save(BytesIO(), model)


def test_state_checkpoint_retains_deferred_context_and_current_arrays():
    state = zdx.State(index=2, time=0.5, dt=0.1, amplitude=3.0)
    original = {
        "state": state,
        "model": zdx.Map(x=state.ref("amplitude"), s=2.0),
        "key": jr.key(5),
    }
    loaded = _roundtrip(original)
    zdx.validate_state(loaded["model"], loaded["state"])
    assert isinstance(loaded["model"].resolve(), zdx.Map)
    assert jnp.allclose(loaded["model"].resolve(state=loaded["state"]), 6.0)
    assert loaded["state"].value("index").dtype == state.value("index").dtype
    advanced = jax.jit(lambda state: state.step())(loaded["state"])
    assert advanced.value("index") == 3
    assert jnp.allclose(advanced.value("time"), 0.6)


def test_unit_archive_preserves_definition_without_capturing_process_settings():
    previous = zdx.get_units()
    try:
        zdx.set_units({"cartesian": "m"})
        original = {
            "current": zdx.Unit(x=2.0, unit="mm"),
            "fixed": zdx.Unit(x=2.0, unit="mm", to="m"),
        }
        file = BytesIO()
        save(file, original)
        zdx.set_units({"cartesian": "um"})
        loaded = load(file)
        assert loaded["current"].unit_out is None
        assert loaded["fixed"].unit_out == "m"
        assert jnp.allclose(loaded["current"].resolve(), 2000.0)
        assert jnp.allclose(loaded["fixed"].resolve(), 0.002)
    finally:
        zdx.set_units(previous)
