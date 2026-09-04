from __future__ import annotations

from copy import copy
from typing import Any

import equinox as eqx
import pytest

import zodiax as zdx
from zodiax.base import (
    _NO_ALIAS,
    _children,
    _is_runtime_field,
    _local_alias,
    _normalise_alias,
    _raised_candidates,
    _resolve_raised_attribute,
    _resolve_structural_path,
    _sequence_index,
    _validate_mapping_keys,
)


class Leaf(zdx.Module):
    value: Any


class Branch(zdx.Module):
    leaf: Any
    mapping: Any
    sequence: Any
    fixed: Any = eqx.field(static=True)


class Pair(zdx.Module):
    left: Any
    right: Any


class Uninitialised(zdx.Module):
    value: Any

    def __init__(self):
        self.alias = None


class BrokenProperty(zdx.Module):
    leaf: Any

    @property
    def broken(self):
        raise AttributeError("unavailable")


class RuntimeLeaf(zdx.Module):
    ordinary: Any
    dynamic: Any = zdx.field(runtime=True)


def make_branch(alias=None):
    return Branch(
        leaf=Leaf(value=1.0),
        mapping={"item": Leaf(value=2.0)},
        sequence=(Leaf(value=3.0),),
        fixed="metadata",
        alias=alias,
    )


def test_field_metadata_and_runtime_detection_validation():
    with pytest.raises(TypeError, match="runtime must be a bool"):
        zdx.field(runtime=1)
    with pytest.raises(ValueError, match="must not define"):
        zdx.field(metadata={"zodiax_runtime": False})

    model = RuntimeLeaf(ordinary=1.0, dynamic=2.0)
    assert _is_runtime_field(model, "dynamic")
    assert not _is_runtime_field(model, "ordinary")
    assert not _is_runtime_field(model, "missing")


def test_alias_normalisation_accepts_mappings_and_rejects_invalid_specs():
    assert _normalise_alias({"second": "leaf.value", "first": "mapping.item"}) == (
        ("first", "mapping.item"),
        ("second", "leaf.value"),
    )
    assert _normalise_alias([]) is None

    invalid = [
        (1, "alias must be None"),
        (("only-one",), r"\(name, path\) pair"),
        ((("name", 1),), "names and paths must be strings"),
        (("not-valid", "leaf.value"), "public Python identifier"),
        (("class", "leaf.value"), "public Python identifier"),
        (("name", ""), "non-empty dotted path"),
        ((("name", "leaf.value"), ("name", "mapping.item")), "must be unique"),
    ]
    for value, message in invalid:
        with pytest.raises((TypeError, ValueError), match=message):
            _normalise_alias(value)


def test_local_alias_handles_uninitialised_owner():
    owner = object.__new__(Uninitialised)
    assert _local_alias(owner, "value") is _NO_ALIAS
    assert _local_alias(owner, "value") is _NO_ALIAS
    assert _local_alias(object(), "value") is _NO_ALIAS


@pytest.mark.parametrize(
    ("value", "key", "message"),
    [
        ([1], "value", "integer sequence index"),
        ([1], "-1", "canonical non-negative"),
        ([1], "01", "canonical non-negative"),
        ([1], "1", "outside a sequence"),
    ],
)
def test_sequence_index_validation(value, key, message):
    with pytest.raises(KeyError, match=message):
        _sequence_index(value, key, "items")


def test_structural_path_resolves_each_container_and_rejects_invalid_paths():
    model = make_branch()
    assert _resolve_structural_path(model, "leaf.value") == 1.0
    assert _resolve_structural_path(model, "mapping.item.value") == 2.0
    assert _resolve_structural_path(model, "sequence.0.value") == 3.0

    errors = [
        ("mapping.missing", "mapping has no key"),
        ("leaf.value.more", "traverses non-structural"),
        ("missing", "no public field"),
        ("fixed", "crosses static field"),
    ]
    for path, message in errors:
        with pytest.raises(KeyError, match=message):
            _resolve_structural_path(model, path)

    uninitialised = object.__new__(Uninitialised)
    object.__setattr__(uninitialised, "alias", None)
    with pytest.raises(KeyError, match="has not been initialised"):
        _resolve_structural_path(uninitialised, "value")


def test_module_attribute_validation_covers_collisions_uninitialised_and_private():
    with pytest.raises(ValueError, match="collides with a declared attribute"):
        make_branch(alias=("leaf", "mapping.item"))

    uninitialised = object.__new__(Uninitialised)
    object.__setattr__(uninitialised, "alias", None)
    with pytest.raises(AttributeError, match="has not been initialised"):
        _ = uninitialised.value
    with pytest.raises(AttributeError, match="has no attribute '_private'"):
        _ = make_branch()._private

    collision = copy(make_branch())
    object.__setattr__(collision, "alias", (("leaf", "mapping.item"),))
    with pytest.raises(AttributeError, match="collides with a declared attribute"):
        collision.get("leaf", to_array=False)

    with pytest.raises(AttributeError, match="unavailable"):
        _ = BrokenProperty(leaf=Leaf(value=1.0)).broken

    with pytest.raises(AttributeError, match="field 'value'.*not been initialised"):
        _resolve_raised_attribute(uninitialised, "value")


def test_raised_lookup_ignores_broken_properties_and_cycles():
    broken = BrokenProperty(leaf=Leaf(value=1.0))
    with pytest.raises(AttributeError, match="has no attribute 'broken'"):
        _ = Pair(left=broken, right=0.0).broken

    cyclic = []
    cyclic.append(cyclic)
    assert list(_children(cyclic, "left", frozenset({id(cyclic)}))) == []

    uninitialised = object.__new__(Uninitialised)
    object.__setattr__(uninitialised, "alias", None)
    assert [path for _, path, _ in _children(uninitialised, "root", frozenset())] == [
        "root.alias"
    ]
    _validate_mapping_keys(uninitialised)
    assert _raised_candidates(uninitialised, "missing")[0] == []


def test_alias_validation_rejects_noncanonical_and_stale_storage():
    model = make_branch(alias=("entry", "mapping.item.value"))
    model.validate_aliases()
    zdx.validate_aliases((model, {"copy": model}, [model], 1.0))

    invalid = copy(model)
    object.__setattr__(invalid, "alias", (("bad name", "leaf.value"),))
    with pytest.raises(ValueError, match="not a valid alias specification"):
        invalid.validate_aliases()

    noncanonical = copy(model)
    object.__setattr__(
        noncanonical,
        "alias",
        (("second", "leaf.value"), ("first", "mapping.item.value")),
    )
    with pytest.raises(ValueError, match="canonical order"):
        noncanonical.validate_aliases()

    stale = copy(model)
    object.__setattr__(stale, "alias", (("entry", "mapping.missing.value"),))
    with pytest.raises(ValueError, match="does not target an available"):
        stale.validate_aliases()

    with pytest.raises(AttributeError, match="points to unavailable"):
        _ = stale.entry


def test_module_update_first_all_non_strict_and_validation():
    first = Leaf(value=1.0)
    second = Leaf(value=2.0)

    updated_first = zdx.update({"value": 5.0}, first, second)
    assert updated_first[0].value == 5.0
    assert updated_first[1].value == 2.0

    updated_all = zdx.update({"value": 6.0}, first, second, mode="all")
    assert [model.value for model in updated_all] == [6.0, 6.0]

    unchanged = zdx.update({"missing": 1.0}, first, strict=False)
    assert unchanged[0] is first

    invalid = [
        (([("value", 1.0)], first), {}, "parameters must be a mapping"),
        (({1: 1.0}, first), {}, "parameter paths must be strings"),
        (({"value": 1.0}, first), {"strict": 1}, "strict must be a bool"),
        (({"value": 1.0}, first), {"mode": "none"}, "mode must be"),
        (({"value": 1.0}, object()), {}, "objects must be Zodiax"),
        (({"missing": 1.0}, first), {}, "Unused parameter paths"),
    ]
    for args, kwargs, message in invalid:
        with pytest.raises((TypeError, ValueError, KeyError), match=message):
            zdx.update(*args, **kwargs)
