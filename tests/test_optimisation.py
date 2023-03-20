from __future__ import annotations
from jax import config
config.update("jax_debug_nans", True)
from optax import adam, GradientTransformation, MultiTransformState
import zodiax


# Define paths
path1 = 'param'
path2 = 'b.param'


def test_get_optimiser(Base_instance):
    """
    tests the get_optimiser method
    """
    # Define parameters and construct base
    base = Base_instance

    # Define paths & groups
    optimisers = [adam(0), adam(1)] # These are actually arbitrary

    # Test paths
    optim, opt_state = zodiax.optimisation.get_optimiser(base, [path1, path2], optimisers)
    assert isinstance(optim, GradientTransformation)
    assert isinstance(opt_state, MultiTransformState)