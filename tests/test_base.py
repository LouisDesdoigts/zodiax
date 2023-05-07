from __future__ import annotations
from jax import config
config.update("jax_debug_nans", True)
import jax.numpy as np
import zodiax
import pytest


def test_unwrap():
    """
    Test the _unwrap method
    """

    # Test unwrapping
    wrapped_a = ['a', ['b', ['c', ['d']]]]
    wrapped_b = [[[['a'], 'b'], 'c'], 'd']
    wrapped_c = ['a', 'b', 'c', 'd']

    assert zodiax.base._unwrap(wrapped_a) == ['a', 'b', 'c', 'd']
    assert zodiax.base._unwrap(wrapped_b) == ['a', 'b', 'c', 'd']
    assert zodiax.base._unwrap(wrapped_c) == ['a', 'b', 'c', 'd']

    # Test with values
    wrapped_a = ['a', ['b', 'c', 'd']]
    wrapped_b = [['a', 'b', 'c'], 'd']
    wrapped_c = ['a', 'b', 'c', 'd']

    assert zodiax.base._unwrap(wrapped_a, [1, 2])[1]       == [1, 2, 2, 2]
    assert zodiax.base._unwrap(wrapped_b, [1, 2])[1]       == [1, 1, 1, 2]
    assert zodiax.base._unwrap(wrapped_c, [1, 2, 3, 4])[1] == [1, 2, 3, 4]


def test_format():
    """
    test the _format method
    """

    # Test formatting
    path_a = ['a.b', ['b.c', ['c.d', ['d.e']]]]
    path_b = [[[['a.b'], 'b.c'], 'c.d'], 'd.e']
    path_c = ['a.b', 'b.c', 'c.d', 'd.e']

    assert zodiax.base._format(path_a) == [['a', 'b'], ['b', 'c'], ['c', 'd'], ['d', 'e']]
    assert zodiax.base._format(path_b) == [['a', 'b'], ['b', 'c'], ['c', 'd'], ['d', 'e']]
    assert zodiax.base._format(path_c) == [['a', 'b'], ['b', 'c'], ['c', 'd'], ['d', 'e']]

    # Test with values
    path_a = ['a.b', ['b.c', 'c.d', 'd.e']]
    path_b = [['a.b', 'b.c', 'c.d'], 'd.e']
    path_c = ['a.b', 'b.c', 'c.d', 'd.e']

    assert zodiax.base._format(path_a, [1, 2])[1]       == [1, 2, 2, 2]
    assert zodiax.base._format(path_b, [1, 2])[1]       == [1, 1, 1, 2]
    assert zodiax.base._format(path_c, [1, 2, 3, 4])[1] == [1, 2, 3, 4]


class TestBase():
    """
    Tests the Base class.
    """

    def test_get(self, create_base):
        """
        tests the get method
        """
        # Test single parameter
        create_base().get('param')

        # Test multiple parameters
        create_base().get(['param', 'b.param'])


    def test_set(self, create_base):
        """
        tests the set method
        """
        # Test single parameter
        create_base().set('param', 10.)

        # Test multiple parameters
        create_base().set(['param', 'b.param'], [10., 10.])


    def test_add(self, create_base):
        """
        tests the add method
        """
        # Test single parameter
        create_base().add('param', 10.)

        # Test multiple parameters
        create_base().add(['param', 'b.param'], [10., 10.])


    def test_multiply(self, create_base):
        """
        tests the multiply method
        """
        # Test single parameter
        create_base().multiply('param', 10.)

        # Test multiple parameters
        create_base().multiply(['param', 'b.param'], [10., 10.])


    def test_divide(self, create_base):
        """
        tests the divide method
        """
        # Test single parameter
        create_base().divide('param', 10.)

        # Test multiple parameters
        create_base().divide(['param', 'b.param'], [10., 10.])


    def test_power(self, create_base):
        """
        tests the power method
        """
        # Test single parameter
        create_base().power('param', 10.)

        # Test multiple parameters
        create_base().power(['param', 'b.param'], [10., 10.])


    def test_min(self, create_base):
        """
        tests the min method
        """
        # Test single parameter
        create_base().min('param', 10.)

        # Test multiple parameters
        create_base().min(['param', 'b.param'], [10., 10.])


    def test_max(self, create_base):
        """
        tests the max method
        """
        # Test single parameter
        create_base().max('param', 10.)

        # Test multiple parameters
        create_base().max(['param', 'b.param'], [10., 10.])