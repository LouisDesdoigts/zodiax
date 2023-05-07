from __future__ import annotations
from jax import config
config.update("jax_debug_nans", True)
import zodiax

paths = [
    'param',
    'b.param',
    ['param', 'b.param'],
]

def fn(pytree):
    return pytree.param * pytree.b.param


def test_filter_grad(create_base):
    pytree = create_base()
    for path in paths:
        zodiax.eqx.filter_grad(path)(fn)(pytree)


def test_filter_value_and_grad(create_base):
    pytree = create_base()
    for path in paths:
        zodiax.eqx.filter_value_and_grad(path)(fn)(pytree)


def test_partition(create_base):
    pytree = create_base()
    for path in paths:
        zodiax.eqx.partition(pytree, path)
