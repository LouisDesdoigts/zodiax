# Import modules in dependency order
from . import base, stats
from . import derivatives, numerics, serialisation
from . import optimisation

name = "zodiax"
__version__ = "0.5.0"

# Dynamically import symbols into the top-level namespace
for module in [
    base,
    stats,
    numerics,
    derivatives,
    serialisation,
    optimisation,
]:
    globals().update({name: getattr(module, name) for name in module.__all__})

# Add to __all__
__all__ = (
    base.__all__
    + stats.__all__
    + numerics.__all__
    + derivatives.__all__
    + serialisation.__all__
    + optimisation.__all__
)
