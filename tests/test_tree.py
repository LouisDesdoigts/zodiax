from __future__ import annotations
from jax import config
config.update("jax_debug_nans", True)
import zodiax


# Define paths
path1 = 'param'
path2 = 'b.param'


def test_get_args(Base_instance):
    pytree = Base_instance

    filter_spec = zodiax.tree.get_args(pytree, [path1])

    assert filter_spec.get(path1) == True
    assert filter_spec.get(path2) == False