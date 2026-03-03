from __future__ import annotations
import pytest
import zodiax

from jax import config, numpy as np

config.update("jax_debug_nans", True)


PATHS = [
    "param",
    "b.param",
    ["param", "b.param"],
]


@pytest.mark.parametrize("path", PATHS)
def test_boolean_filter(create_base, path):
    pytree = create_base()
    out = zodiax.tree.boolean_filter(pytree, path)
    values = out.get(path)
    if isinstance(values, list):
        assert all(values)
    else:
        assert values


@pytest.mark.parametrize("path", PATHS)
def test_boolean_filter_inverse(create_base, path):
    pytree = create_base()
    out = zodiax.tree.boolean_filter(pytree, path, inverse=True)
    values = out.get(path)
    if isinstance(values, list):
        assert all(not value for value in values)
    else:
        assert not values


@pytest.mark.parametrize("params", ["param", None])
def test_set_array(create_base, params):
    pytree = create_base()
    out = zodiax.tree.set_array(pytree, params)
    assert hasattr(out.get("param"), "shape")


def test_to_array_array_passthrough():
    arr = np.array([1.0])
    out = zodiax.tree._to_array(arr)
    assert out is arr
