"""Callable model classes, unsupported functions, state, and payload boundaries."""

from functools import partial
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


class FunctionField(eqx.Module):
    function: object


class StaticFunctionField(eqx.Module):
    function: object = eqx.field(static=True)


class CallableModel(zdx.Module):
    weights: jax.Array
    bias: jax.Array

    def __call__(self, value):
        return jnp.sum(self.weights * value**2) + self.bias


class ForeignCallableModel(eqx.Module):
    weights: jax.Array
    bias: jax.Array

    def __call__(self, value):
        return jnp.sum(self.weights * value**2) + self.bias


class MethodOwner:
    def square(self, value):
        return value**2


class StaticMetadata(eqx.Module):
    value: object = eqx.field(static=True)


class StateSubclass(eqx.nn.State):
    pass


def _square(value):
    return value**2


def _offset_function(offset):
    def shifted(value):
        return value + offset

    return shifted


def _roundtrip(value):
    file = BytesIO()
    zdx.save(file, value)
    file.seek(0)
    return zdx.load(file)


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
    np.lib.format.write_array_header_1_0(
        file,
        {"descr": np.dtype(np.float32).str, "fortran_order": False, "shape": shape},
    )
    return file.getvalue()


@pytest.mark.parametrize(
    "function",
    [
        _square,
        lambda value: value**2,
        _offset_function(2.0),
        MethodOwner().square,
        partial(_offset_function(2.0), value=3.0),
        jr.normal,
    ],
    ids=["function", "lambda", "closure", "bound-method", "partial", "jax-sampler"],
)
@pytest.mark.parametrize("storage", ["standalone", "dynamic-field", "static-field"])
def test_functions_are_unsupported_as_values_or_fields(function, storage):
    value = function
    if storage == "dynamic-field":
        value = FunctionField(function)
    elif storage == "static-field":
        value = StaticFunctionField(function)

    with pytest.raises(TypeError) as error:
        zdx.save(BytesIO(), value)
    # These identify the rejected concept and the supported alternative, without
    # fixing the complete diagnostic sentence.
    message = str(error.value).lower()
    assert "unsupported callable" in message
    assert "zodiax" in message and "class method" in message


@pytest.mark.parametrize("model_type", [CallableModel, ForeignCallableModel])
def test_callable_model_path_roundtrip_preserves_fields_jit_and_gradients(
    tmp_path, model_type
):
    original = model_type(
        weights=jnp.asarray([2.0, 3.0], dtype=float),
        bias=jnp.asarray(0.5, dtype=float),
    )
    path = tmp_path / "model"
    zdx.save(path, original)
    loaded = zdx.load(path)

    assert path.with_suffix(".zdx").is_file()
    assert type(loaded) is model_type
    assert eqx.tree_equal(loaded, original, typematch=True)
    value = jnp.asarray([1.0, -2.0], dtype=float)
    assert jnp.allclose(jax.jit(lambda model, x: model(x))(loaded, value), 14.5)

    parameter_gradient = eqx.filter_grad(lambda model: model(value))(loaded)
    np.testing.assert_allclose(parameter_gradient.weights, [1.0, 4.0])
    np.testing.assert_allclose(parameter_gradient.bias, 1.0)
    np.testing.assert_allclose(jax.grad(loaded)(value), [4.0, -12.0])


def test_default_equinox_mlp_rejects_its_function_valued_fields():
    model = eqx.nn.MLP(2, 2, 4, 1, key=jr.key(1))
    with pytest.raises(TypeError, match="unsupported callable"):
        zdx.save(BytesIO(), model)


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


def test_old_callable_nodes_are_rejected_by_the_definition_schema():
    callable_node = {"kind": "callable", "identifier": "jax.random.normal"}
    with pytest.raises(ValueError, match="callable"):
        zdx.ObjectDefinition.from_dict(callable_node)

    file = BytesIO()
    zdx.save(file, jnp.ones(1))

    def replace_definition(manifest, payload):
        manifest["definition"] = callable_node

    _rewrite(file, replace_definition)
    with pytest.raises(ValueError, match="callable"):
        zdx.load(file, strict=False)


def test_equinox_state_and_its_model_roundtrip_together():
    model, state = eqx.nn.make_with_state(Counter)()
    state = state.set(model.index, jnp.asarray(7, dtype=jnp.int32))

    loaded_model, loaded_state = _roundtrip((model, state))

    assert loaded_model.index.marker == model.index.marker
    assert loaded_state.get(loaded_model.index) == 7


def test_state_with_unstable_object_keys_is_rejected():
    model = Counter()
    state = eqx.nn.State(model)

    with pytest.raises(TypeError):
        zdx.save(BytesIO(), state)


def test_state_subclasses_are_rejected_instead_of_losing_their_type():
    state = StateSubclass.tree_unflatten((0,), (jnp.asarray(1.0),))

    with pytest.raises(TypeError):
        zdx.save(BytesIO(), state)


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
    with pytest.raises(TypeError):
        zdx.save(BytesIO(), StaticMetadata(value))


def test_npy_header_is_checked_before_array_allocation():
    file = BytesIO()
    zdx.save(file, jnp.ones(1, dtype=jnp.float32))
    payload = _huge_npy_header((2**40,))

    def declare_huge_shape(manifest, new_payload):
        manifest["definition"]["shape"] = [2**40]
        manifest["payload"]["size"] = len(new_payload)

    _rewrite(file, declare_huge_shape, payload)

    with pytest.raises(ValueError):
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
