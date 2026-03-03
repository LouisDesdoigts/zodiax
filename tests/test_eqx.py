from __future__ import annotations
import pytest
import zodiax

from jax import config

config.update("jax_debug_nans", True)

PATHS = [
    "param",
    "b.param",
    ["param", "b.param"],
]


def fn(pytree):
    return pytree.param * pytree.b.param


@pytest.mark.parametrize("path", PATHS)
def test_filter_grad(create_base, path):
    pytree = create_base()
    grads = zodiax.eqx.filter_grad(path)(fn)(pytree)
    assert grads is not None


@pytest.mark.parametrize("path", PATHS)
def test_filter_value_and_grad(create_base, path):
    pytree = create_base()
    value, grads = zodiax.eqx.filter_value_and_grad(path)(fn)(pytree)
    assert value is not None
    assert grads is not None


@pytest.mark.parametrize("path", PATHS)
def test_partition(create_base, path):
    pytree = create_base()
    traced, static = zodiax.eqx.partition(pytree, path)
    assert traced is not None
    assert static is not None
