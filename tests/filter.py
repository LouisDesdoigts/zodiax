from __future__ import annotations
from utilities import Utility, UtilityUser
import jax.numpy as np
from optax import GradientTransformation, MultiTransformState, adam
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
    


class BaseUtility(Utility):
    """
    Utility for the Base class.
    """
    param1 : float
    param2 : float


    class A(zodiax.base.ExtendedBase):
        """
        Test subclass to test the Base methods
        """
        param : float
        b     : B


        def __init__(self, param, b):
            """
            Constructor for the Base testing class
            """
            self.param = param
            self.b = b


        def model(self):
            """
            Sample modelling function
            """
            return self.param**2 + self.b.param**2


    class B(zodiax.base.ExtendedBase):
        """
        Test subclass to test the Base methods
        """
        param : float


        def __init__(self, param):
            """
            Constructor for the Base testing class
            """
            self.param = param


    def __init__(self : Utility):
        """
        Constructor for the Optics Utility.
        """
        self.param1 = 1.
        self.param2 = 1.


    def construct(self : Utility,
                  param1 : float = None,
                  param2 : float = None):
        """
        Safe constructor for the dLuxModule, associated with this utility.
        """
        param1 = self.param1 if param1 is None else param1
        param2 = self.param2 if param2 is None else param2
        return self.A(param1, self.B(param2))