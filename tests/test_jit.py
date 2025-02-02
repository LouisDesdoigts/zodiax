from __future__ import annotations
import zodiax

from jax import config

config.update("jax_debug_nans", True)


paths = [
    "param",
    "b.param",
    ["param", "b.param"],
]


def test_improve_jit_hash(create_base):
    pytree = create_base()
    for path in paths:
        zodiax.experimental.jit.improve_jit_hash(pytree, path)
