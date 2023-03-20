from __future__ import annotations
# from abc import ABC, abstractmethod
import zodiax
import pytest


@pytest.fixture(scope='class')
def Base_instance(a=1., b=2.):
    """
    Construct a Base instance for testing
    """
    yield A(a, B(b))


class A(zodiax.base.Base):
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


class B(zodiax.base.Base):
    """
    Test subclass to test the Base methods
    """
    param : float

    def __init__(self, param):
        """
        Constructor for the Base testing class
        """
        self.param = param