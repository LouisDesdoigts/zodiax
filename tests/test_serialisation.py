from __future__ import annotations
from jax import config
import zodiax
from equinox import tree_equal
import os
config.update("jax_debug_nans", True)


def test_serialise_deserialise(create_base):
    pytree = create_base()
    try:
        zodiax.experimental.serialisation.serialise('test_serialisation.zdx', pytree)
        # pytree_2 = zodiax.experimental.serialisation.deserialise('test_serialisation.zdx')
        
        os.remove('test_serialisation.zdx')
        # assert tree_equal(pytree, pytree_2)
    except Exception as e:
        try:
            os.remove('test_serialisation.zdx')
        except FileNotFoundError:
            pass
        raise e