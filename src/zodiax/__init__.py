# Import modules in dependency order
from . import base, stats
from . import numerics, linalg, serialisation
from . import diffops, optimisation

name = "zodiax"
__version__ = "0.5.0"

# Dynamically import symbols into the top-level namespace
for module in [
    base,
    stats,
    numerics,
    linalg,
    serialisation,
    diffops,
    optimisation,
]:
    globals().update({name: getattr(module, name) for name in module.__all__})

# Add to __all__
__all__ = (
    base.__all__
    + stats.__all__
    + numerics.__all__
    + linalg.__all__
    + serialisation.__all__
    + diffops.__all__
    + optimisation.__all__
)
