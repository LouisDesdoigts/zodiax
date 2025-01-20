from __future__ import annotations
import zodiax

from jax import config

config.update("jax_debug_nans", True)


# Paths
paths = [
    "param",
    "b.param",
    ["param", "b.param"],
]


def test_boolean_filter(create_base):
    """
    test the boolean_filter function from zodiax/tree.py
    """
    pytree = create_base()
    for path in paths:
        zodiax.tree.boolean_filter(pytree, path)


def test_set_array(create_base):
    """
    test the set_array function from zodiax/tree.py
    """
    pytree = create_base()

    for params in [paths[0], None]:
        zodiax.tree.set_array(pytree, params)
