name = "zodiax"
__version__ = "0.3.3"

# Import as modules
from . import base
from . import tree
from . import optimisation
from . import experimental
from . import eqx
from . import bayes
from . import jit

# Import core functions from modules
from .base import *
from .tree import *
from .optimisation import *
from .eqx import *
from .bayes import *
from .jit import *

# Add to __all__
__all__ = base.__all__ + tree.__all__ + optimisation.__all__ + \
    experimental.__all__ + eqx.__all__ + jit.__all__ + bayes.__all__
