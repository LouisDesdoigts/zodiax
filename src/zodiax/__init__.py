# Import as modules
from . import base, optimisation, stats, batching, eqx, fisher, tree

name = "zodiax"
__version__ = "0.5.0"

# Dynamically import symbols into the top-level namespace
# for module in [base, fisher, eqx, optimisation, tree, wrappers]:
for module in [base, optimisation, stats, batching, eqx, fisher, tree]:
    globals().update({name: getattr(module, name) for name in module.__all__})

# Add to __all__
__all__ = (
    base.__all__
    + optimisation.__all__
    + stats.__all__
    + batching.__all__
    + eqx.__all__
    + fisher.__all__
    + tree.__all__
)
