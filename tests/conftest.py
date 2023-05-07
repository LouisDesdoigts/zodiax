from __future__ import annotations
# from abc import ABC, abstractmethod
import zodiax
import pytest


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


# @pytest.fixture(scope='class')
# def create_Base(a=1., b=2.):
@pytest.fixture
def create_base():
    """
    Construct a Base instance for testing
    """
    def _create_base(
        param : float = 1.,
        b     : float = 2.,
    ) -> zodiax.base.Base:
        """
        Construct a Base instance for testing
        """
        return A(param, B(b))
    return _create_base