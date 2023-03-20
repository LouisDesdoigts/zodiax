from __future__ import annotations
from jax import config
config.update("jax_debug_nans", True)
import zodiax


def sample_function(pytree):
    return pytree.param * pytree.b.param


# Define paths
path1 = 'param'
path2 = 'b.param'


def test_filter_grad(Base_instance):
    pytree = Base_instance

    args = [path1]
    filter_spec = zodiax.tree.get_args(pytree, args)

    _ = zodiax.equinox.filter_grad(filter_spec)(sample_function)(pytree)
    _ = zodiax.equinox.filter_grad(args)(sample_function)(pytree)
    

def test_filter_value_and_grad(Base_instance):
    pytree = Base_instance

    args = [path1]
    filter_spec = zodiax.tree.get_args(pytree, args)

    _ = zodiax.equinox.filter_value_and_grad(filter_spec)(sample_function)(pytree)
    _ = zodiax.equinox.filter_value_and_grad(args)(sample_function)(pytree)