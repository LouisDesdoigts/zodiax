import pytest
import equinox as eqx
import zodiax as zdx
import jax.numpy as np
import jax.random as jr


class Foo(zdx.WrapperHolder):

    def __init__(self, nn):
        values, structure = zdx.build_wrapper(nn)
        self.values = values
        self.structure = structure

    def __call__(self, x):
        return self.build(x)


def test_WrapperHolder():
    """
    Test the WrapperHolder class
    """

    eqx_model = eqx.nn.MLP(
        in_size=16, out_size=16, width_size=32, depth=1, key=jr.PRNGKey(0)
    )

    x = np.ones(16)
    foo = Foo(eqx_model)

    # Test the __init__ method
    foo = Foo(eqx_model)
    assert isinstance(foo, zdx.WrapperHolder)

    # Test the __call__ method
    x = np.ones(16)
    assert np.allclose(foo(x), eqx_model(x))

    # Test value updating
    assert np.allclose(foo.multiply("values", 0.0)(x), np.zeros_like(x))
