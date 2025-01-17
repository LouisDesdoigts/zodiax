# Import as modules
from . import (
    base,
    eqx,
    fisher,
    optimisation,
    tree,
    experimental,
)

name = "zodiax"
__version__ = "0.4.2"

# Dynamically import symbols into the top-level namespace
for module in [base, fisher, eqx, optimisation, tree]:
    globals().update({name: getattr(module, name) for name in module.__all__})

# Add to __all__
__all__ = (
    base.__all__
    + fisher.__all__
    + eqx.__all__
    + optimisation.__all__
    + tree.__all__
    + experimental.__all__
)
