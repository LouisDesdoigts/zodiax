name = "zodiax"
__version__ = "0.3.3"

# Import as modules
from . import base
from . import tree
from . import optimisation
from . import experimental
from . import equinox

# Import core functions from modules
from .base import *
from .tree import *
from .optimisation import *
from .equinox import *

# Add to __all__
__all__ = base.__all__ + tree.__all__ + optimisation.__all__ + \
    experimental.__all__ + equinox.__all__
