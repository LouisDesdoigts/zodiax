from __future__ import annotations
from utilities import BaseUtility
from jax import config
import zodiax
config.update("jax_debug_nans", True)


def sample_function(pytree):
    return pytree.param * pytree.b.param


def test_filter_grad():
    pytree = BaseUtility().construct(2., 2.)

    args = ['param']
    filter_spec = pytree.get_args(args)

    _ = zodiax.filter.filter_grad(filter_spec)(sample_function)(pytree)
    _ = zodiax.filter.filter_grad(args)(sample_function)(pytree)
    

def test_filter_value_and_grad():
    pytree = BaseUtility().construct(2., 2.)

    args = ['param']
    filter_spec = pytree.get_args(args)

    _ = zodiax.filter.filter_value_and_grad(filter_spec)(sample_function)(pytree)
    _ = zodiax.filter.filter_value_and_grad(args)(sample_function)(pytree)