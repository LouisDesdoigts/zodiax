from __future__ import annotations
from abc import ABC, abstractmethod
import zodiax

class UtilityUser(ABC):
    """
    The base utility class. These utility classes are designed to
    define safe constructors and constants for testing. These
    classes are for testing purposes only.
    """
    utility : Utility


class Utility(ABC):
    """

    """
    def __init__(self : Utility) -> Utility:
        """

        """
        return


    @abstractmethod
    def construct(self : Utility) -> object:
        """

        """
        return


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


class ExtendedBaseUtility(BaseUtility):
    """
    Utility for the Base class.
    """
    pass