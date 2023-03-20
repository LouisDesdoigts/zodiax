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


# Define paths
path1 = 'param'
path2 = 'b.param'

@pytest.mark.usefixtures("Base_instance")
class TestBase():
    """
    Tests the Base class.
    """

    def test_get(self, Base_instance):
        """
        tests the get method
        """
        # Define parameters and construct base
        base = Base_instance

        assert base.get(path1) == 1.
        assert base.get(path2) == 2.


    def test_set(self, Base_instance):
        """
        tests the set method
        """
        # Define parameters and construct base
        param1 = 5.
        param2 = 10.
        base = Base_instance

        # Test set
        new_base = base.set([path1, path2], [param1, param2])
        assert new_base.param   == param1
        assert new_base.b.param == param2


    def test_add(self, Base_instance):
        """
        tests the add method
        """
        # Define parameters and construct base
        param1 = 5.
        param2 = 10.
        base = Base_instance

        # Test add
        new_base = base.add([path1, path2], [param1, param2])
        assert new_base.param   == base.get(path1) + param1
        assert new_base.b.param == base.get(path2) + param2


    def test_multiply(self, Base_instance):
        """
        tests the multiply method
        """
        # Define parameters and construct base
        param1 = 5.
        param2 = 10.
        base = Base_instance

        # Test multiply
        new_base = base.multiply([path1, path2], [param1, param2])
        assert new_base.param   == base.get(path1) * param1
        assert new_base.b.param == base.get(path2) * param2


    def test_divide(self, Base_instance):
        """
        tests the divide method
        """
        # Define parameters and construct base
        param1 = 5.
        param2 = 10.
        base = Base_instance

        # Test divide
        new_base = base.divide([path1, path2], [param1, param2])
        assert new_base.param   == base.get(path1) / param1
        assert new_base.b.param == base.get(path2) / param2


    def test_power(self, Base_instance):
        """
        tests the power method
        """
        # Define parameters and construct base
        param1 = 5.
        param2 = 10.
        base = Base_instance

        # Test power
        new_base = base.power([path1, path2], [param1, param2])
        assert new_base.param   == base.get(path1)**param1
        assert new_base.b.param == base.get(path2)**param2


    def test_min(self, Base_instance):
        """
        tests the min method
        """
        # Define parameters and construct base
        param1 = 5.
        param2 = 10.
        base = Base_instance

        # Test min
        new_base = base.min([path1, path2], [param1, param2])
        assert new_base.param   == np.minimum(base.get(path1), param1)
        assert new_base.b.param == np.minimum(base.get(path2), param2)


    def test_max(self, Base_instance):
        """
        tests the max method
        """
        # Define parameters and construct base
        param1 = 5.
        param2 = 10.
        base = Base_instance

        # Test max
        new_base = base.max([path1, path2], [param1, param2])
        assert new_base.param   == np.maximum(base.get(path1), param1)
        assert new_base.b.param == np.maximum(base.get(path2), param2)


    def test_apply(self, Base_instance):
        """
        tests the  method
        """
        # Define parameters and construct base
        fn1 = lambda x: x + 5.
        fn2 = lambda x: x + 10.
        base = Base_instance

        # Test 
        new_base = base.apply([path1, path2], [fn1, fn2])
        assert new_base.param   == base.get(path1) + 5.
        assert new_base.b.param == base.get(path2) + 10.


    def test_apply_args(self, Base_instance):
        """
        tests the apply_args method
        """
        # Define parameters and construct base
        fn = lambda x, a: x + a
        param1 = 5.
        param2 = 10.
        base = Base_instance

        # Test apply_args
        new_base = base.apply_args([path1, path2], fn, [(param1,), (param2,)])
        assert new_base.param   == base.get(path1) + param1
        assert new_base.b.param == base.get(path2) + param2


    def test_set_and_call(self, Base_instance):
        """
        tests the set_and_call method
        """
        # Define parameters and construct base
        param1 = 2.
        param2 = 4.
        base = Base_instance

        # Define groups
        values = [param1, param2]

        # Test paths
        new_base = base.set_and_call([path1, path2], values, "model")
        assert new_base == param1**2 + param2**2

    def test_apply_and_calll(self, Base_instance):
        """
        tests the apply_and_calll method
        """
        # Define parameters and construct base
        base = Base_instance

        # Define paths & fns
        fn1 = lambda x: x * 2.
        fn2 = lambda x: x * 4.
        fns = [fn1, fn2]

        # Test paths
        new_base = base.apply_and_call([path1, path2], fns, "model")
        assert new_base == (base.get(path1)*2.)**2 + (base.get(path2)*4.)**2