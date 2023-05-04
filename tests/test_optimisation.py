from __future__ import annotations
from jax import config
config.update("jax_debug_nans", True)
from optax import adam, GradientTransformation, MultiTransformState
import zodiax


# Paths
paths = [
    'param',
    'b.param',
    ['param', 'b.param'],
]

# Optimisers
optimisers = [
    adam(0),
    adam(1),
    [adam(0), adam(1)],
]


def test_get_optimiser(create_base):
    """
    tests the get_optimiser method
    """
    # Define parameters and construct base
    pytree = create_base()

    # Test paths
    for path, optimiser in zip(paths, optimisers):
        zodiax.optimisation.get_optimiser(pytree, path, optimiser)