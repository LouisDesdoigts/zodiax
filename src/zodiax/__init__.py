# Import as modules
from . import base, diffops, optimisation, stats

name = "zodiax"
__version__ = "0.5.0"

# Dynamically import symbols into the top-level namespace
for module in [base, optimisation, stats, diffops]:
    globals().update({name: getattr(module, name) for name in module.__all__})

# Add to __all__
__all__ = base.__all__ + optimisation.__all__ + stats.__all__ + diffops.__all__
