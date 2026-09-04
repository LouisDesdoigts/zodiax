"""Regression tests for generated definitions and generic save/load workflows."""

from collections import OrderedDict, defaultdict, namedtuple
from io import BytesIO, StringIO
import json
import math
import zipfile

import equinox as eqx
import jax
import jax.numpy as np
import jax.random as jr
import numpy as onp
import pytest

import zodiax as zdx


class ForeignModule(eqx.Module):
    """Non-Zodiax-specific module used to exercise generic definitions."""

    array: object
    count: int
    mode: str = eqx.field(static=True)


class AlternativeModule(eqx.Module):
    """Nominally different module used for template mismatch tests."""

    array: object
    count: int
    mode: str = eqx.field(static=True)


class StaticExpression(zdx.Expression):
    """Small expression with structured static metadata."""

    value: object
    metadata: object = eqx.field(static=True)

    def evaluate(self, **context):
        del context
        return self.value


class ConstructorModule(eqx.Module):
    """Module whose constructor calls can be observed."""

    calls = 0
    array: object
    label: str

    def __init__(self, array, label):
        type(self).calls += 1
        self.array = array
        self.label = label


class CustomArrayLike:
    """Unsupported object implementing the JAX array protocol."""

    def __jax_array__(self):
        return np.ones(1)


def _roundtrip(original, **load_kwargs):
    """Round-trip an object through the public functional API."""
    file = BytesIO()
    zdx.save(file, original)
    assert not file.closed
    file.seek(0)
    loaded = zdx.load(file, **load_kwargs)
    assert not file.closed
    return loaded


def test_object_definition_has_readable_generated_schema():
    obj = ForeignModule(np.arange(3.0, dtype=np.float32), 3, "science")
    definition = zdx.ObjectDefinition.from_object(obj)

    expected = {
        "kind": "module",
        "type": f"{ForeignModule.__module__}:ForeignModule",
        "fields": [
            {
                "name": "array",
                "static": False,
                "metadata": {"runtime": False},
                "value": {
                    "kind": "jax_array",
                    "shape": [3],
                    "dtype": "float32",
                    "weak_type": False,
                },
            },
            {
                "name": "count",
                "static": False,
                "metadata": {"runtime": False},
                "value": 3,
            },
            {
                "name": "mode",
                "static": True,
                "metadata": {"runtime": False},
                "value": "science",
            },
        ],
    }
    assert definition.root == expected
    assert definition.to_dict() == expected
    assert json.loads(definition.to_json()) == expected
    assert str(definition) == definition.to_json()


def test_object_definition_owns_defensive_copies():
    source = zdx.ObjectDefinition.from_object(
        ForeignModule(np.ones(2), 2, "science")
    ).to_dict()
    definition = zdx.ObjectDefinition.from_dict(source)

    source["fields"][0]["value"]["shape"][0] = 99
    returned = definition.root
    returned["fields"][1]["value"] = 99

    assert definition.root["fields"][0]["value"]["shape"] == [2]
    assert definition.root["fields"][1]["value"] == 2


def test_object_definition_validation_compares_metadata_not_array_data():
    definition = zdx.ObjectDefinition.from_object(
        ForeignModule(np.arange(3.0), 3, "science")
    )

    definition.validate(ForeignModule(np.zeros(3), 3, "science"))

    for changed in (
        ForeignModule(np.zeros(4), 3, "science"),
        ForeignModule(np.zeros(3, dtype=np.complex64), 3, "science"),
        ForeignModule(np.zeros(3), 4, "science"),
        ForeignModule(np.zeros(3), 3, "engineering"),
    ):
        with pytest.raises(ValueError, match="Object definition mismatch"):
            definition.validate(changed)


def test_object_definition_builds_template_without_constructor():
    ConstructorModule.calls = 0
    original = ConstructorModule(np.arange(3.0), "stored")
    definition = zdx.ObjectDefinition.from_object(original)
    calls = ConstructorModule.calls

    template = definition.build_template()

    assert ConstructorModule.calls == calls
    assert isinstance(template.array, jax.ShapeDtypeStruct)
    assert template.array.shape == (3,)
    assert template.label == "stored"
    definition.validate(template)


@pytest.mark.parametrize(
    ("keys", "message"),
    [
        ([1, True], "Duplicate mapping keys"),
        ([2, 1], "not canonical"),
        ([1, "a"], "do not have a stable JAX ordering"),
    ],
    ids=["equal", "noncanonical", "unsortable"],
)
def test_object_definition_rejects_ambiguous_plain_dict_keys(keys, message):
    definition = {
        "kind": "mapping",
        "type": "builtins:dict",
        "entries": [{"key": key, "value": None} for key in keys],
    }

    with pytest.raises(ValueError, match=message):
        zdx.ObjectDefinition.from_dict(definition)


@pytest.mark.parametrize(
    ("definition", "message"),
    [
        (2**63, "signed 64 bits"),
        (
            {
                "kind": "jax_prng_key",
                "shape": [],
                "data_shape": [1_000_001],
                "impl": "fry",
            },
            "random-key data.*too large",
        ),
    ],
    ids=["integer", "prng-size"],
)
def test_object_definition_rejects_bounded_schema_values(definition, message):
    with pytest.raises(ValueError, match=message):
        zdx.ObjectDefinition.from_dict(definition)


def test_object_definition_rejects_excessive_nesting():
    definition = None
    for _ in range(130):
        definition = {"kind": "list", "items": [definition]}

    with pytest.raises(ValueError, match="maximum nesting depth"):
        zdx.ObjectDefinition.from_dict(definition)


def test_object_definition_rejects_changed_installed_class_schema():
    definition = zdx.ObjectDefinition.from_object(
        ForeignModule(np.ones(2), 2, "science")
    ).to_dict()
    definition["fields"].pop()

    changed = zdx.ObjectDefinition.from_dict(definition)
    with pytest.raises(ValueError, match="installed class schema.*changed"):
        changed.build_template()


def test_nested_generic_path_archive_has_complete_manifest(tmp_path):
    original = {
        "model": ForeignModule(np.arange(4.0).reshape(2, 2), 4, "science"),
        "offset": np.asarray([1.0, -1.0], dtype=np.float32),
    }
    path = tmp_path / "model"

    zdx.save(path, original)
    loaded = zdx.load(path)

    archive_path = path.with_suffix(".zdx")
    assert archive_path.is_file()
    assert eqx.tree_equal(loaded, original, typematch=True)

    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == {"manifest.json", "leaves.eqx"}
        manifest = json.loads(archive.read("manifest.json"))
    assert set(manifest) == {"format", "definition", "payload", "created_with"}
    assert manifest["format"] == {"name": "zodiax", "version": 1}
    assert manifest["definition"] == zdx.ObjectDefinition.from_object(original).root
    assert manifest["payload"]["member"] == "leaves.eqx"
    assert manifest["payload"]["codec"] == "equinox-jax-leaves"
    assert manifest["payload"]["version"] == 1
    assert manifest["payload"]["size"] > 0
    assert manifest["created_with"]["packages"]["zodiax"] == zdx.__version__


def test_functional_save_replaces_caller_owned_stream():
    first = ForeignModule(np.zeros((2, 2)), 1, "first")
    second = ForeignModule(np.ones((2, 2)), 2, "second")
    file = BytesIO(b"stale archive contents")

    zdx.save(file, first)
    zdx.save(file, second)
    file.seek(0)
    loaded = zdx.load(file)

    assert not file.closed
    assert eqx.tree_equal(loaded, second, typematch=True)


def test_explicit_path_suffix_is_preserved(tmp_path):
    original = ForeignModule(np.ones((2, 2)), 2, "science")
    path = tmp_path / "model.archive"

    zdx.save(path, original)
    loaded = zdx.load(path)

    assert path.is_file()
    assert not path.with_suffix(".zdx").exists()
    assert eqx.tree_equal(loaded, original, typematch=True)


def test_structured_literals_and_containers_roundtrip_without_like():
    original = OrderedDict(
        [
            ("none", None),
            ("bool", True),
            ("limits", (-(2**63), 2**63 - 1)),
            ("negative_zero", -0.0),
            ("float", 1.25),
            ("complex", 1 + 2j),
            ("unicode", "lambda: λ"),
            ("range", range(1, 8, 2)),
            ("slice", slice(1, 8, 2)),
            ("type", float),
            ("list", [1, "two", None]),
            ("plain_dict", {"z": np.arange(2), "a": np.ones((2, 2))}),
        ]
    )

    loaded = _roundtrip(original)

    assert list(loaded) == list(original)
    assert list(loaded["plain_dict"]) == ["a", "z"]
    assert math.copysign(1.0, loaded["negative_zero"]) == -1.0
    assert eqx.tree_equal(loaded, original, typematch=True)


def test_structured_static_metadata_roundtrip_without_like():
    metadata = [
        {"complex": 1 + 2j, "range": range(1, 5, 2)},
        slice(1, 5, 2),
        float,
    ]
    original = StaticExpression(np.arange(3.0), metadata)

    loaded = _roundtrip(original)

    assert eqx.tree_equal(loaded, original, typematch=True)


def test_shape_dtype_struct_is_accepted_as_like():
    original = ForeignModule(np.arange(3.0), 3, "science")
    like = eqx.filter_eval_shape(lambda: original)

    loaded = _roundtrip(original, like=like)

    assert isinstance(like.array, jax.ShapeDtypeStruct)
    assert isinstance(loaded.array, jax.Array)
    assert eqx.tree_equal(loaded, original, typematch=True)


@pytest.mark.parametrize("mismatch", ["shape", "dtype"])
def test_like_rejects_mismatched_array_metadata(mismatch):
    original = ForeignModule(np.ones((2, 2)), 2, "science")
    if mismatch == "shape":
        like = ForeignModule(np.ones((3, 3)), 2, "science")
    else:
        like = ForeignModule(np.ones((2, 2), dtype=np.complex64), 2, "science")
    file = BytesIO()
    zdx.save(file, original)
    file.seek(0)

    with pytest.raises(ValueError, match=f"Object structure mismatch.*{mismatch}"):
        zdx.load(file, like=like)


@pytest.mark.parametrize("mismatch", ["root-type", "mapping-key"])
def test_like_rejects_mismatched_topology(mismatch):
    leaf = ForeignModule(np.ones((2, 2)), 2, "science")
    if mismatch == "root-type":
        original = leaf
        like = AlternativeModule(np.zeros((2, 2)), 2, "science")
    else:
        original = {"model": leaf}
        like = {"other": ForeignModule(np.zeros((2, 2)), 2, "science")}
    file = BytesIO()
    zdx.save(file, original)
    file.seek(0)

    with pytest.raises(ValueError, match="Object structure mismatch"):
        zdx.load(file, like=like)


def test_module_scoped_foreign_module_roundtrip_without_configuration():
    original = ForeignModule(np.arange(3.0), 3, "science")

    loaded = _roundtrip(original)

    assert type(loaded) is ForeignModule
    assert eqx.tree_equal(loaded, original, typematch=True)


def test_local_class_can_be_resolved_by_like_or_custom_types():
    class LocalModule(eqx.Module):
        array: object
        label: str

    original = LocalModule(np.arange(3.0), "archived")
    like = LocalModule(np.zeros(3), "template")
    definition = zdx.ObjectDefinition.from_object(original)
    identifier = definition.root["type"]
    file = BytesIO()
    zdx.save(file, original)

    file.seek(0)
    with pytest.raises(ValueError, match="Local type.*Supply like"):
        zdx.load(file)

    file.seek(0)
    loaded_like = zdx.load(file, like=like)
    file.seek(0)
    loaded_registered = zdx.load(file, custom_types={identifier: LocalModule})

    assert eqx.tree_equal(loaded_like, original, typematch=True)
    assert eqx.tree_equal(loaded_registered, original, typematch=True)


def test_custom_type_arguments_are_validated():
    original = ForeignModule(np.arange(2.0), 2, "science")
    identifier = zdx.ObjectDefinition.from_object(original).root["type"]
    file = BytesIO()
    zdx.save(file, original)

    file.seek(0)
    with pytest.raises(TypeError, match="custom_types must be a mapping"):
        zdx.load(file, custom_types=[])
    file.seek(0)
    with pytest.raises(TypeError, match="is not a class"):
        zdx.load(file, custom_types={identifier: object()})
    file.seek(0)
    with pytest.raises(ValueError, match="nominal identifier"):
        zdx.load(file, custom_types={identifier: StaticExpression})
    file.seek(0)
    with pytest.raises(TypeError, match="cannot be supplied together"):
        zdx.load(file, like=original, custom_types={})


@pytest.mark.parametrize(
    "dtype_name",
    [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
        "bfloat16",
    ],
)
def test_supported_jax_dtype_roundtrip(dtype_name):
    enable_x64 = getattr(jax, "enable_x64", None)
    if enable_x64 is None:
        enable_x64 = jax.experimental.enable_x64

    with enable_x64():
        original = np.asarray([0, 1], dtype=getattr(np, dtype_name))
        loaded = _roundtrip(original)

    assert isinstance(loaded, jax.Array)
    assert loaded.dtype == original.dtype
    assert np.array_equal(loaded, original)


def test_typed_prng_keys_roundtrip_without_like():
    key = jr.key(17)
    original = {"batched": jr.split(key, 3), "single": key}

    loaded = _roundtrip(original)

    for name in original:
        assert str(jr.key_impl(loaded[name])) == str(jr.key_impl(original[name]))
        assert loaded[name].shape == original[name].shape
        assert np.array_equal(jr.key_data(loaded[name]), jr.key_data(original[name]))


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (onp.array([1.0], dtype=onp.float32), "only JAX arrays"),
        (onp.float32(1.0), "only JAX arrays"),
        (object(), "unsupported object metadata"),
        (CustomArrayLike(), "custom array-like"),
        (np.array(1.0), "weakly typed"),
        (np.zeros(1, dtype=np.float8_e5m2), "unsupported dtype"),
        (jax.ShapeDtypeStruct((1,), np.float32), "abstract array"),
        (float("nan"), "non-finite Python float"),
        (complex(1, float("inf")), "non-finite Python complex"),
        (-(2**63) - 1, "signed 64-bit"),
        (2**63, "signed 64-bit"),
    ],
    ids=[
        "numpy-array",
        "numpy-scalar",
        "opaque",
        "custom-array-like",
        "weak-jax",
        "float8",
        "abstract-jax",
        "nonfinite-float",
        "nonfinite-complex",
        "integer-underflow",
        "integer-overflow",
    ],
)
def test_save_rejects_unsupported_leaf(value, message):
    with pytest.raises(TypeError, match=message):
        zdx.save(BytesIO(), value)


def test_save_rejects_unregistered_static_callable():
    transformed = StaticExpression(np.arange(3.0), lambda value: value)

    with pytest.raises(TypeError, match=r"metadata.*unsupported callable"):
        zdx.save(BytesIO(), transformed)


def test_save_rejects_static_array():
    with pytest.warns(UserWarning, match="being set as static"):
        original = StaticExpression(np.ones(1), np.ones(1))

    with pytest.raises(TypeError, match="unsupported static array"):
        zdx.save(BytesIO(), original)


@pytest.mark.parametrize("state_kind", ["dictionary", "slot"])
def test_save_rejects_non_field_instance_state(state_kind):
    if state_kind == "dictionary":
        original = ForeignModule(np.ones(1), 1, "science")
        object.__setattr__(original, "cache", "populated")
    else:

        class SlottedModule(eqx.Module):
            __slots__ = ("cache",)

            value: object

            def __init__(self, value):
                self.value = value
                object.__setattr__(self, "cache", "populated")

        original = SlottedModule(np.ones(1))

    with pytest.raises(TypeError, match="non-field state: cache"):
        zdx.save(BytesIO(), original)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (defaultdict(int, value=1), "unsupported mapping type defaultdict"),
        (namedtuple("Pair", "left right")(1, 2), "unsupported tuple type Pair"),
        ({1, 2}, "unsupported set metadata"),
        (OrderedDict([(frozenset({1}), 2)]), "unsupported mapping key type"),
        ({1: "one", "two": 2}, "without a stable JAX ordering"),
    ],
    ids=["mapping-subclass", "namedtuple", "set", "mapping-key", "mixed-dict"],
)
def test_save_rejects_unsupported_container(value, message):
    with pytest.raises(TypeError, match=message):
        zdx.save(BytesIO(), value)


@pytest.mark.parametrize("file", [StringIO(), object()], ids=["text", "not-file"])
def test_save_rejects_non_binary_destination(file):
    with pytest.raises(TypeError, match="binary file|path or writable"):
        zdx.save(file, np.ones(1))
