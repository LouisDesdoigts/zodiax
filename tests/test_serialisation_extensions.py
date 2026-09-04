"""Focused tests for callable, state, validation, and bounded-payload codecs."""

from io import BytesIO
import json
import zipfile

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import zodiax as zdx


class Counter(eqx.Module):
    index: eqx.nn.StateIndex

    def __init__(self):
        self.index = eqx.nn.StateIndex(jnp.asarray(0, dtype=jnp.int32))


class Positive(eqx.Module):
    value: int

    def __zodiax_validate__(self):
        if self.value <= 0:
            raise ValueError("value must be positive")


class MutatingValidator(eqx.Module):
    value: int

    def __zodiax_validate__(self):
        object.__setattr__(self, "value", self.value + 1)


class SymbolicCallable(eqx.Module):
    function: object


class StaticMetadata(eqx.Module):
    value: object = eqx.field(static=True)


class StateSubclass(eqx.nn.State):
    pass


def _square(value):
    return value**2


def _roundtrip(value):
    file = BytesIO()
    zdx.save(file, value)
    file.seek(0)
    return zdx.load(file)


def _unsafe_module(cls, **values):
    value = object.__new__(cls)
    for name, item in values.items():
        object.__setattr__(value, name, item)
    return value


def _rewrite(file, update_manifest, payload=None):
    file.seek(0)
    with zipfile.ZipFile(file) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        original_payload = archive.read("leaves.eqx")
    payload = original_payload if payload is None else payload
    update_manifest(manifest, payload)

    file.seek(0)
    file.truncate(0)
    with zipfile.ZipFile(file, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("leaves.eqx", payload)
        archive.writestr("manifest.json", json.dumps(manifest))
    file.seek(0)


def _huge_npy_header(shape):
    file = BytesIO()
    file.write(np.lib.format.magic(1, 0))
    np.lib.format.write_array_header_1_0(
        file,
        {"descr": np.dtype(np.float32).str, "fortran_order": False, "shape": shape},
    )
    return file.getvalue()


def test_registered_symbolic_callable_roundtrips_and_lambda_remains_rejected():
    zdx.register_callable("zodiax.tests.square", _square)
    loaded = _roundtrip(SymbolicCallable(_square))

    assert loaded.function is _square
    assert loaded.function(jnp.asarray(3.0)) == 9

    with pytest.raises(TypeError, match="unsupported callable"):
        zdx.save(BytesIO(), SymbolicCallable(lambda value: value))


def test_default_equinox_mlp_roundtrips_with_builtin_callable_symbols():
    model = eqx.nn.MLP(2, 2, 4, 1, key=jr.key(1))
    loaded = _roundtrip(model)
    value = jnp.asarray([0.25, -0.5], dtype=jnp.float32)

    assert eqx.tree_equal(loaded, model, typematch=True)
    assert jnp.allclose(loaded(value), model(value))


def test_nested_numerical_definition_roundtrips():
    definition = zdx.Exp(
        x=zdx.Map(
            x=jnp.asarray([0.25, -0.5]),
            M=jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
            b=jnp.asarray([0.1, -0.2]),
        )
    )

    loaded = _roundtrip(definition)

    assert eqx.tree_equal(loaded, definition, typematch=True)
    assert jnp.allclose(
        jax.jit(lambda value: value.resolve())(loaded), definition.resolve()
    )


def test_archive_callable_resolution_is_registry_only():
    zdx.register_callable("zodiax.tests.square", _square)
    file = BytesIO()
    zdx.save(file, SymbolicCallable(_square))

    def replace_identifier(manifest, payload):
        del payload
        callable_node = manifest["definition"]["fields"][0]["value"]
        callable_node["identifier"] = "unregistered.package.function"

    _rewrite(file, replace_identifier)

    with pytest.raises(ValueError, match="is not registered"):
        zdx.load(file)


def test_equinox_state_and_its_model_roundtrip_together():
    model, state = eqx.nn.make_with_state(Counter)()
    state = state.set(model.index, jnp.asarray(7, dtype=jnp.int32))

    loaded_model, loaded_state = _roundtrip((model, state))

    assert loaded_model.index.marker == model.index.marker
    assert loaded_state.get(loaded_model.index) == 7


def test_state_with_unstable_object_keys_is_rejected():
    model = Counter()
    state = eqx.nn.State(model)

    with pytest.raises(TypeError, match="not a stable state key"):
        zdx.save(BytesIO(), state)


def test_state_subclasses_are_rejected_instead_of_losing_their_type():
    state = StateSubclass.tree_unflatten((0,), (jnp.asarray(1.0),))

    with pytest.raises(TypeError, match="unsupported Equinox State subclass"):
        zdx.save(BytesIO(), state)


def test_object_validation_hook_rejects_constructor_bypassed_invariant():
    file = BytesIO()
    zdx.save(file, Positive(1))

    def make_invalid(manifest, payload):
        del payload
        manifest["definition"]["fields"][0]["value"] = -1

    _rewrite(file, make_invalid)

    with pytest.raises(ValueError, match="Object validation failed at root"):
        zdx.load(file)


def test_object_validation_hook_must_not_mutate_metadata():
    file = BytesIO()
    zdx.save(file, MutatingValidator(1))
    file.seek(0)

    with pytest.raises(ValueError, match="Object definition mismatch"):
        zdx.load(file)


@pytest.mark.parametrize(
    "value",
    [
        _unsafe_module(
            zdx.Linked,
            alias=None,
            value=jnp.asarray(1.0, dtype=jnp.float32),
            key="",
        ),
        _unsafe_module(zdx.Deferred, alias=None, key=""),
        _unsafe_module(
            zdx.Mask,
            alias=None,
            x=None,
            indices=jnp.asarray([1, 0], dtype=jnp.int32),
            shape=(2,),
        ),
        _unsafe_module(
            zdx.Map,
            alias=None,
            x=1,
            s=None,
            M=None,
            b=None,
        ),
        _unsafe_module(zdx.TreeLayout, paths=(), shapes=()),
    ],
    ids=("linked", "deferred", "mask", "map", "layout"),
)
def test_zodiax_classes_validate_constructor_bypassed_values_before_save(value):
    with pytest.raises(ValueError, match="Object validation failed at root"):
        zdx.save(BytesIO(), value)


@pytest.mark.parametrize(
    "value",
    [
        zdx.Deferred("hidden"),
        [zdx.Deferred("nested")],
        zdx.Linked(None, key="owner"),
    ],
    ids=("deferred", "nested-deferred", "linked"),
)
def test_link_nodes_are_rejected_inside_static_metadata(value):
    with pytest.raises(TypeError, match="node in static metadata"):
        zdx.save(BytesIO(), StaticMetadata(value))


def test_npy_header_is_checked_before_array_allocation():
    file = BytesIO()
    zdx.save(file, jnp.ones(1, dtype=jnp.float32))
    payload = _huge_npy_header((2**40,))

    def declare_huge_shape(manifest, new_payload):
        manifest["definition"]["shape"] = [2**40]
        manifest["payload"]["size"] = len(new_payload)

    _rewrite(file, declare_huge_shape, payload)

    with pytest.raises(ValueError, match="payload could not be decoded"):
        zdx.load(file)


def test_header_first_decoder_roundtrips_bfloat16_and_typed_prng_keys():
    original = {
        "array": jnp.asarray([1, 2], dtype=jnp.bfloat16),
        "key": jr.key(5),
    }

    loaded = _roundtrip(original)

    assert loaded["array"].dtype == jnp.bfloat16
    assert jnp.array_equal(loaded["array"], original["array"])
    assert jnp.array_equal(jr.key_data(loaded["key"]), jr.key_data(original["key"]))
