from __future__ import annotations

from copy import copy
from io import BytesIO

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import zodiax as zdx
from zodiax.serialisation import _archive, _callables, _leaves, _schema
from zodiax.serialisation._reconstruction import _build_template, _decode_literal
from zodiax.serialisation._types import _identifier_parts, _resolve_type
from zodiax.serialisation._validation import _validate_rebuilt
from zodiax.serialisation.definition import (
    _collect_types,
    _describe_root,
    _extra_state_names,
    _first_difference,
    _literal_definition,
    _mapping_key_definition,
    _structural_view,
)


class Plain(eqx.Module):
    value: object


class ValidatedBase(zdx.Module):
    value: object

    def __zodiax_validate__(self):
        if self.value < 0:
            raise ValueError("negative")


class ValidatedChild(ValidatedBase):
    def __zodiax_validate__(self):
        if self.value == 0:
            raise ValueError("zero")


class NonCallableValidation(eqx.Module):
    value: object
    __zodiax_validate__ = 1


class Slotted(eqx.Module):
    __slots__ = ("cache",)
    value: object


class Namespace:
    class Nested:
        pass


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


def field_node(name="value", value=None, static=False, runtime=False, **updates):
    node = {
        "name": name,
        "static": static,
        "metadata": {"runtime": runtime},
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
    ("definition", "message"),
    [
        ({"kind": "list", "items": [], "extra": 1}, "Invalid fields"),
        ({"kind": "type", "type": 1}, "Invalid type identifier"),
        ({"kind": "type", "type": "invalid"}, "Invalid type identifier"),
        ({"kind": "callable", "identifier": "bad-name"}, "callable identifier"),
        (float("inf"), "Non-finite Python float"),
        ({}, "no valid kind"),
        (module_node(fields={}), "bounded list"),
        (module_node(fields=[1]), "must be a mapping"),
        (module_node(fields=[field_node(name="bad-name")]), "field name"),
        (module_node(fields=[field_node(static=1)]), "static marker"),
        (
            module_node(fields=[field_node(metadata={})]),
            "Zodiax metadata",
        ),
        (
            module_node(fields=[field_node(metadata={"runtime": 1})]),
            "runtime marker",
        ),
        (
            module_node(fields=[field_node(static=True, runtime=True)]),
            "static and runtime",
        ),
        (
            module_node(fields=[field_node(), field_node()]),
            "Duplicate module fields",
        ),
        ({"kind": "list", "items": ()}, "bounded list"),
        (
            {"kind": "mapping", "type": "builtins:list", "entries": []},
            "Unsupported mapping type",
        ),
        (
            {"kind": "mapping", "type": "builtins:dict", "entries": {}},
            "bounded list",
        ),
        (
            {"kind": "mapping", "type": "builtins:dict", "entries": [1]},
            "entry.*must be a mapping",
        ),
        (
            {"kind": "equinox_state", "version": 2, "entries": []},
            "State version",
        ),
        (
            {"kind": "equinox_state", "version": 1, "entries": {}},
            "bounded list",
        ),
        (
            {"kind": "equinox_state", "version": 1, "entries": [1]},
            "entry.*must be a mapping",
        ),
        (
            {
                "kind": "equinox_state",
                "version": 1,
                "entries": [{"key": "", "value": None}],
            },
            "Invalid State key",
        ),
        (
            {
                "kind": "equinox_state",
                "version": 1,
                "entries": [{"key": 1.0, "value": None}],
            },
            "Invalid State key",
        ),
        (
            {
                "kind": "equinox_state",
                "version": 1,
                "entries": [
                    {"key": 1, "value": None},
                    {"key": 1, "value": None},
                ],
            },
            "Duplicate State keys",
        ),
        (array_node(dtype="float8"), "Invalid JAX array dtype"),
        (array_node(weak_type=True), "Weak JAX arrays"),
        (key_node(impl=""), "random-key implementation"),
        ({"kind": "complex", "value": [1.0]}, "complex literal"),
        ({"kind": "range", "start": 0, "stop": 1, "step": 0}, "range literal"),
        ({"kind": "unknown"}, "Unsupported object definition kind"),
    ],
)
def test_definition_schema_rejects_malformed_nodes(definition, message):
    with pytest.raises(ValueError, match=message):
        _schema._validate_definition(definition)


def test_definition_schema_limits_nodes_cycles_static_arrays_and_mapping_keys(
    monkeypatch,
):
    with monkeypatch.context() as patch:
        patch.setattr(_schema, "_MAX_NODES", 1)
        with pytest.raises(ValueError, match="too many nodes"):
            _schema._validate_definition({"kind": "list", "items": [None]})

    cyclic = {"kind": "list"}
    cyclic["items"] = [cyclic]
    with pytest.raises(ValueError, match="contains a cycle"):
        _schema._validate_definition(cyclic)

    static_array = module_node(fields=[field_node(value=array_node(), static=True)])
    with pytest.raises(ValueError, match="Static array"):
        _schema._validate_definition(static_array)
    static_key = module_node(fields=[field_node(value=key_node(), static=True)])
    with pytest.raises(ValueError, match="Static random-key"):
        _schema._validate_definition(static_key)

    tuple_key = {"kind": "tuple", "items": ["a", 1]}
    mapping = {
        "kind": "mapping",
        "type": "builtins:dict",
        "entries": [{"key": tuple_key, "value": None}],
    }
    _schema._validate_definition(mapping)

    with pytest.raises(ValueError, match="Unsupported mapping key"):
        _schema._validate_mapping_key(
            {"kind": "complex", "value": [1.0, 0.0]},
            "root.key",
            0,
            [0],
            set(),
        )
    with pytest.raises(ValueError, match="Unsupported mapping key definition"):
        _schema._decode_mapping_key(
            {"kind": "complex", "value": [1.0, 0.0]}, "root.key"
        )


def test_callable_registration_decorator_conflicts_and_resolution():
    @_callables.register_callable("zodiax.tests.edge.decorated")
    def decorated(value):
        return value

    assert decorated(2) == 2
    assert _callables._callable_identifier(decorated) == "zodiax.tests.edge.decorated"
    assert _callables._callable_identifier(object()) is None
    assert (
        _callables._resolve_callable("zodiax.tests.edge.decorated", "root") is decorated
    )

    with pytest.raises(TypeError, match="function must be callable"):
        zdx.register_callable("zodiax.tests.edge.invalid", 1)

    def other(value):
        return value

    with pytest.raises(ValueError, match="already registered"):
        zdx.register_callable("zodiax.tests.edge.decorated", other)
    with pytest.raises(ValueError, match="already registered as"):
        zdx.register_callable("zodiax.tests.edge.other_name", decorated)
    with pytest.raises(ValueError, match="is not registered"):
        _callables._resolve_callable("zodiax.tests.edge.missing", "root")


def test_builtin_callable_registration_tolerates_uninspectable_mlp(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(
            _callables.inspect,
            "signature",
            lambda value: (_ for _ in ()).throw(TypeError("uninspectable")),
        )
        _callables._register_builtin_callables()


@pytest.mark.parametrize(
    ("identifier", "message"),
    [
        (1, "Invalid type identifier"),
        ("missing-separator", "Invalid type identifier"),
        ("module:<locals>.Type", "Local type"),
    ],
)
def test_type_identifier_parsing_validation(identifier, message):
    with pytest.raises(ValueError, match=message):
        _identifier_parts(identifier, "root")

    with pytest.raises(ValueError, match="Invalid type identifier"):
        _identifier_parts("bad-module:Type", "root")


def test_type_resolution_rejects_unavailable_unregistered_and_non_types():
    with pytest.raises(ValueError, match="not available"):
        _resolve_type("missing_package:Type", None, "root")
    with pytest.raises(ValueError, match="not registered"):
        _resolve_type("math:Missing", None, "root")
    with pytest.raises(ValueError, match="not registered"):
        _resolve_type("math:pi.child", None, "root")
    with pytest.raises(ValueError, match="is not a type"):
        _resolve_type("math:pi", None, "root")
    identifier = f"{Namespace.Nested.__module__}:Namespace.Nested"
    assert _resolve_type(identifier, None, "root") is Namespace.Nested


def test_reconstruction_reports_invalid_literal_and_container_definitions():
    assert _decode_literal([1], None, "root") is _leaves._NOT_PAYLOAD

    invalid = [
        ({"kind": "callable"}, "Invalid 'callable'"),
        ({"kind": "complex", "value": [1, 2, 3]}, "Invalid 'complex'"),
        ({"kind": "range", "start": 0, "stop": 1, "step": "x"}, "Invalid 'range'"),
        ({"kind": "type", "type": "missing:Type"}, "Invalid 'type'"),
        ([1], "must be a mapping"),
        (module_node(type_name="builtins:int"), "is not a Module"),
        (
            module_node(fields={}, type_name=f"{Plain.__module__}:Plain"),
            "fields.*must be a list",
        ),
        ({"kind": "list", "items": {}}, "Sequence items"),
        (
            {"kind": "mapping", "type": "builtins:list", "entries": []},
            "Unsupported mapping type",
        ),
        (
            {"kind": "mapping", "type": "builtins:dict", "entries": {}},
            "Mapping entries",
        ),
        (
            {"kind": "mapping", "type": "builtins:dict", "entries": [1]},
            "Invalid mapping entry",
        ),
        ({"kind": "equinox_state", "entries": {}}, "State entries"),
        ({"kind": "equinox_state", "entries": [1]}, "Invalid State entry"),
        ({"kind": "unknown"}, "Unsupported object definition kind"),
    ]
    for definition, message in invalid:
        with pytest.raises(ValueError, match=message):
            _build_template(definition, None)


def test_leaf_definition_and_placeholder_edge_cases(monkeypatch):
    key_dtype = jax.random.key(0).dtype
    with monkeypatch.context() as patch:
        patch.setattr(jax.dtypes, "prng_key", None)
        assert not _leaves._is_prng_dtype(jnp.float32)
    assert not _leaves._is_prng_dtype(object())

    class BadShape:
        def __iter__(self):
            raise TypeError

    with pytest.raises(TypeError, match="non-concrete shape"):
        _leaves._normalise_shape(BadShape(), "root")

    abstract_key = jax.ShapeDtypeStruct((), key_dtype)
    with pytest.raises(TypeError, match="abstract JAX random key"):
        _leaves._payload_definition(abstract_key, template=True, path="root")

    invalid_placeholders = [
        ({"kind": "jax_array"}, "Invalid JAX array definition"),
        (array_node(weak_type=1), "Invalid JAX weak type"),
        ({"kind": "jax_prng_key"}, "Invalid JAX random-key definition"),
        (key_node(shape=[1]), "Invalid JAX random-key shape"),
    ]
    for definition, message in invalid_placeholders:
        with pytest.raises(ValueError, match=message):
            _leaves._payload_placeholder(definition, "root")


def test_npy_header_versions_fortran_and_invalid_versions():
    for version in ((1, 0), (2, 0)):
        file = BytesIO()
        np.lib.format.write_array(
            file,
            np.asfortranarray(np.arange(4, dtype=np.float32).reshape(2, 2)),
            version=version,
        )
        file.seek(0)
        reader = _archive._PayloadReader(file, len(file.getvalue()))
        output = _leaves._read_array(
            reader,
            jax.ShapeDtypeStruct((2, 2), jnp.float32),
        )
        assert jnp.array_equal(output, jnp.arange(4).reshape(2, 2))

    file = BytesIO(b"\x93NUMPY\x03\x00")
    with pytest.raises(ValueError, match="Unsupported NPY payload version"):
        _leaves._read_npy_header(file)


def test_array_reader_rejects_header_and_payload_mismatches(monkeypatch):
    file = BytesIO()
    np.lib.format.write_array(file, np.ones(2, dtype=np.float32))
    data = file.getvalue()

    with pytest.raises(ValueError, match="shape.*does not match"):
        _leaves._read_array(
            _archive._PayloadReader(BytesIO(data), len(data)),
            jax.ShapeDtypeStruct((3,), jnp.float32),
        )
    with pytest.raises(ValueError, match="dtype.*does not match"):
        _leaves._read_array(
            _archive._PayloadReader(BytesIO(data), len(data)),
            jax.ShapeDtypeStruct((2,), jnp.int32),
        )

    class Reader:
        remaining = 8

        def read(self, size):
            del size
            return b""

    with monkeypatch.context() as patch:
        patch.setattr(
            _leaves,
            "_read_npy_header",
            lambda file: ((2,), "not-bool", np.dtype(np.float32)),
        )
        with pytest.raises(ValueError, match="fortran_order"):
            _leaves._read_array(Reader(), jax.ShapeDtypeStruct((2,), jnp.float32))

    with monkeypatch.context() as patch:
        patch.setattr(
            _leaves,
            "_read_npy_header",
            lambda file: ((2,), False, np.dtype(np.float32)),
        )
        with pytest.raises(ValueError, match="truncated"):
            _leaves._read_array(Reader(), jax.ShapeDtypeStruct((2,), jnp.float32))

    class ShortReader(Reader):
        remaining = 0

    with monkeypatch.context() as patch:
        patch.setattr(
            _leaves,
            "_read_npy_header",
            lambda file: ((2,), False, np.dtype(np.float32)),
        )
        with pytest.raises(ValueError, match="more array data"):
            _leaves._read_array(ShortReader(), jax.ShapeDtypeStruct((2,), jnp.float32))


def test_archive_stream_helpers_and_manifest_validation(monkeypatch):
    class PartialWriter:
        def write(self, data):
            return len(data) - 1

    with pytest.raises(OSError, match="complete"):
        _archive._CountingWriter(PartialWriter()).write(b"data")

    reader = _archive._PayloadReader(BytesIO(b"abc"), 3)
    assert reader.read() == b"abc"

    class OverReader:
        def read(self, size):
            return b"x" * (size + 1)

    with pytest.raises(OSError, match="exceeded its declared bound"):
        _archive._PayloadReader(OverReader(), 2).read(1)

    monkeypatch.setattr(
        _archive.importlib.metadata,
        "version",
        lambda name: (_ for _ in ()).throw(
            _archive.importlib.metadata.PackageNotFoundError()
        ),
    )
    _archive._package_version.cache_clear()
    assert _archive._package_version("missing") is None

    with pytest.raises(ValueError, match="Invalid JSON constant"):
        _archive._reject_json_constant("NaN")
    with pytest.raises(ValueError, match="must be a JSON object"):
        _archive._validate_manifest([])


def test_archive_additional_manifest_and_stream_failures(monkeypatch, tmp_path):
    original = jnp.ones(1)
    file = BytesIO()
    zdx.save(file, original)

    file.seek(0)
    import json
    import zipfile

    with zipfile.ZipFile(file) as archive:
        manifest = json.loads(archive.read("manifest.json"))

    invalid = [
        (
            {**manifest, "created_with": {"python": "x", "packages": {}, "extra": 1}},
            "provenance fields",
        ),
        (
            {
                **manifest,
                "created_with": {
                    "python": "",
                    "packages": manifest["created_with"]["packages"],
                },
            },
            "Python version",
        ),
    ]
    for value, message in invalid:
        with pytest.raises(ValueError, match=message):
            _archive._validate_manifest(value)

    bad_name = copy(manifest)
    bad_name["created_with"] = copy(manifest["created_with"])
    bad_name["created_with"]["packages"] = copy(manifest["created_with"]["packages"])
    bad_name["created_with"]["packages"][1] = "1"
    with pytest.raises(ValueError, match="package name"):
        _archive._validate_manifest(bad_name)

    class NonSeekable:
        def __init__(self):
            self.data = bytearray()

        def write(self, data):
            self.data.extend(data)
            return len(data)

    zdx.save(NonSeekable(), original)

    monkeypatch.setattr(_archive, "_MAX_MANIFEST_SIZE", 1)
    with pytest.raises(ValueError, match="manifest is too large"):
        zdx.save(BytesIO(), original)

    monkeypatch.setattr(_archive, "_MAX_MANIFEST_SIZE", 16 * 1024**2)

    def fail_write(*args):
        raise RuntimeError("write failed")

    monkeypatch.setattr(_archive, "_write_archive", fail_write)
    with pytest.raises(RuntimeError, match="write failed"):
        _archive._save_path(tmp_path / "failed.zdx", original, {})


def test_archive_cleanup_tolerates_already_removed_temporary_file(
    monkeypatch, tmp_path
):
    descriptor, temporary = _archive.tempfile.mkstemp(dir=tmp_path)

    monkeypatch.setattr(
        _archive.tempfile,
        "mkstemp",
        lambda **kwargs: (descriptor, temporary),
    )

    def remove_then_fail(file, pytree, definition):
        del file, pytree, definition
        _archive.os.unlink(temporary)
        raise RuntimeError("write failed")

    monkeypatch.setattr(_archive, "_write_archive", remove_then_fail)
    with pytest.raises(RuntimeError, match="write failed"):
        _archive._save_path(tmp_path / "failed.zdx", jnp.ones(1), {})


def test_archive_rejects_large_manifest_and_encrypted_member(monkeypatch):
    file = BytesIO()
    zdx.save(file, jnp.ones(1))
    monkeypatch.setattr(_archive, "_MAX_MANIFEST_SIZE", 1)
    file.seek(0)
    with pytest.raises(ValueError, match="manifest is too large"):
        zdx.load(file)

    class Info:
        def __init__(self, filename, flag_bits=0):
            self.filename = filename
            self.flag_bits = flag_bits
            self.compress_type = _archive.zipfile.ZIP_STORED

    class Archive:
        def infolist(self):
            return [Info("manifest.json", 1), Info("leaves.eqx")]

    with pytest.raises(ValueError, match="Encrypted"):
        _archive._validate_members(Archive())


def test_definition_helpers_cover_structural_differences_and_type_collection():
    assert _literal_definition(range(0, 2), "root")["kind"] == "range"
    assert _mapping_key_definition(1.5, "root") == 1.5
    assert _mapping_key_definition(("a", 1), "root")["kind"] == "tuple"

    class Bare:
        __slots__ = ()

    assert _extra_state_names(Bare(), ()) == set()
    assert "cache" not in _extra_state_names(Slotted(jnp.asarray(1.0)), ())
    with pytest.raises(TypeError, match="signed 64-bit integer"):
        _literal_definition(range(0, 2**63), "root")

    assert _first_difference(1, "1") == ("root", 1, "1")
    assert _first_difference({"a": 1}, {"b": 1})[0] == "root"
    assert _first_difference([1], [1, 2])[0] == "root"
    assert _first_difference(-0.0, 0.0) == ("root", -0.0, 0.0)

    definition = {
        "kind": "slice",
        "start": None,
        "stop": 2,
        "step": None,
    }
    assert _structural_view(definition)["kind"] == "slice"
    assert _structural_view({"kind": "jax_array"}) == {"kind": "jax_array"}
    assert _structural_view({"kind": "list", "items": [None]})["kind"] == "list"
    assert (
        _structural_view(
            {
                "kind": "mapping",
                "type": "builtins:dict",
                "entries": [{"key": "a", "value": None}],
            }
        )["kind"]
        == "mapping"
    )
    assert (
        _structural_view(
            {
                "kind": "equinox_state",
                "version": 1,
                "entries": [{"key": 0, "value": None}],
            }
        )["kind"]
        == "equinox_state"
    )
    assert _structural_view({"kind": "callable", "identifier": "x"}) == {
        "kind": "callable"
    }

    assert _collect_types(int)["builtins:int"] is int
    assert _collect_types(1) == {}
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="cyclic object graph"):
        _collect_types(cyclic)

    registry = {f"{Plain.__module__}:Plain": ValidatedBase}
    with pytest.raises(ValueError, match="ambiguous type"):
        _collect_types(Plain(1), registry=registry)
    assert _collect_types({"module": Plain(1)})


def test_definition_reports_uninitialised_modules_cycles_and_invalid_states(
    monkeypatch,
):
    uninitialised = object.__new__(Plain)
    with pytest.raises(TypeError, match="is not initialised"):
        _describe_root(uninitialised, template=False)

    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(TypeError, match="cyclic or too deeply nested"):
        _describe_root(cyclic, template=False)

    state = eqx.nn.State.tree_unflatten((0,), (jnp.ones(1),))
    with monkeypatch.context() as patch:
        patch.setattr(
            eqx.nn.State,
            "tree_flatten",
            lambda self: (_ for _ in ()).throw(ValueError("bad state")),
        )
        with pytest.raises(TypeError, match="not a valid Equinox State"):
            _describe_root(state, template=False)
        with pytest.raises(ValueError, match="invalid Equinox State"):
            _collect_types(state)
        with pytest.raises(ValueError, match="Invalid Equinox State"):
            _validate_rebuilt(state)

    with monkeypatch.context() as patch:
        patch.setattr(eqx.nn.State, "tree_flatten", lambda self: ((jnp.ones(1),), ()))
        with pytest.raises(TypeError, match="inconsistent Equinox State"):
            _describe_root(state, template=False)

    class StateSubclass(eqx.nn.State):
        pass

    subclass = StateSubclass.tree_unflatten((0,), (jnp.ones(1),))
    with pytest.raises(ValueError, match="unsupported Equinox State subclass"):
        _collect_types(subclass)


def test_rebuilt_validation_runs_inheritance_and_wraps_invariants():
    _validate_rebuilt(ValidatedChild(1))

    with pytest.raises(ValueError, match="Object validation failed"):
        _validate_rebuilt(ValidatedChild(0))
    with pytest.raises(TypeError, match="must be callable"):
        _validate_rebuilt(NonCallableValidation(1))

    invalid_paths = copy(ValidatedChild(1))
    object.__setattr__(invalid_paths, "value", {"bad.key": 1})
    with pytest.raises(ValueError, match="Path validation failed"):
        _validate_rebuilt(invalid_paths)

    invalid_alias = copy(ValidatedChild(1))
    object.__setattr__(invalid_alias, "alias", (("missing", "bad.path"),))
    with pytest.raises(ValueError, match="Alias validation failed"):
        _validate_rebuilt(invalid_alias)
