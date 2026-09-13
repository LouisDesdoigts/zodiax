"""Public multi-model update contracts, including partially matching paths."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import zodiax as zdx


class Leaf(zdx.Module):
    value: Any


class Pair(zdx.Module):
    left: Any
    right: Any


def test_update_assigns_each_path_to_its_first_matching_model_without_mutation():
    first = Leaf(value=1.0)
    second = Leaf(value=2.0)

    updated = zdx.update({"value": 5.0}, first, second)

    assert isinstance(updated, tuple)
    np.testing.assert_allclose(updated[0].value, 5.0)
    assert updated[1] is second
    np.testing.assert_allclose(first.value, 1.0)


def test_update_all_changes_every_matching_model():
    first, second = Leaf(1.0), Leaf(2.0)

    updated = zdx.update({"value": 6.0}, first, second, mode="all")

    np.testing.assert_allclose([model.value for model in updated], [6.0, 6.0])
    np.testing.assert_allclose([first.value, second.value], [1.0, 2.0])


def test_update_routes_independent_aliases_to_their_owners():
    first = Leaf(jnp.array(1.0), alias={"gain": "value"})
    second = Leaf(jnp.array(2.0), alias={"offset": "value"})

    updated = zdx.update(
        {"gain": jnp.array(3.0), "offset": jnp.array(4.0)}, first, second
    )

    np.testing.assert_allclose(updated[0].gain, 3.0)
    np.testing.assert_allclose(updated[1].offset, 4.0)
    assert updated[0].alias == first.alias
    assert updated[1].alias == second.alias


@pytest.mark.parametrize("mode", ["first", "all"])
def test_update_skips_missing_sequence_entries(mode):
    first = Leaf(value=[1.0])
    second = Leaf(value=[2.0, 3.0])
    third = Leaf(value=[4.0, 5.0])

    updated = zdx.update({"value.1": 9.0}, first, second, third, mode=mode)

    assert updated[0] is first
    assert updated[1].value == [2.0, 9.0]
    assert updated[2].value == [4.0, 9.0 if mode == "all" else 5.0]
    assert second.value == [2.0, 3.0]
    assert third.value == [4.0, 5.0]


@pytest.mark.parametrize("path", ["missing", "value.1"])
def test_update_strictness_only_controls_paths_unmatched_by_every_model(path):
    model = Leaf(value=[1.0])

    with pytest.raises(KeyError) as error:
        zdx.update({path: 9.0}, model)
    assert path in str(error.value)
    assert zdx.update({path: 9.0}, model, strict=False)[0] is model


def test_nonstrict_update_still_applies_available_paths():
    model = Leaf(1.0)

    updated = zdx.update({"value": 2.0, "missing": 3.0}, model, strict=False)

    np.testing.assert_allclose(updated[0].value, 2.0)
    np.testing.assert_allclose(model.value, 1.0)


def test_update_can_use_a_qualified_path_to_disambiguate_siblings():
    model = Pair(Leaf(1.0), Leaf(2.0))

    updated = zdx.update({"left.value": 3.0}, model)

    np.testing.assert_allclose(updated[0].left.value, 3.0)
    np.testing.assert_allclose(updated[0].right.value, 2.0)


def test_empty_updates_preserve_models():
    first, second = Leaf(1.0), Leaf(2.0)

    updated = zdx.update({}, first, second)

    assert updated[0] is first
    assert updated[1] is second
