# Import as modules
from . import (
    base,
    tree,
    optimisation,
    experimental,
    eqx,
    bayes,
)

name = "zodiax"
__version__ = "0.4.2"

# Add to __all__
__all__ = (
    base.__all__
    + tree.__all__
    + optimisation.__all__
    + experimental.__all__
    + eqx.__all__
    + bayes.__all__
)
