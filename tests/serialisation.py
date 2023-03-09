from __future__ import annotations
from jax import config
import zodiax
from equinox import tree_equal
import os
from utilities import BaseUtility
config.update("jax_debug_nans", True)


def test_serialise_deserialise():
    pytree = BaseUtility().construct(2., 2.)
    zodiax.experimental.serialisation.serialise('test_serialisation.zdx', pytree)
    # pytree_2 = zodiax.experimental.serialisation.deserialise('test_serialisation.zdx')
    os.remove('test_serialisation.zdx')
    # assert tree_equal(pytree, pytree_2)