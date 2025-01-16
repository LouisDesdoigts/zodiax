from __future__ import annotations
import zodiax
from jax import config

config.update("jax_debug_nans", True)


def test_scheduler():
    """
    tests the scheduler method
    """
    zodiax.optimisation.scheduler(0.1, 0, (1, 0.5), (2, 0.25))


def test_sgd():
    """
    tests the sgd method
    """
    zodiax.optimisation.sgd(0.1, 0, (1, 0.5), (2, 0.25))


def test_adam():
    """
    tests the adam method
    """
    zodiax.optimisation.adam(0.1, 0, (1, 0.5), (2, 0.25))


def test_debug_nan_check(create_base):
    """
    tests the debug_nan_check method
    """
    # Define parameters and construct base
    pytree = create_base()

    # Test the debug_nan_check method
    zodiax.optimisation.debug_nan_check(pytree)


def test_zero_nan_check(create_base):
    """
    tests the zero_nan_check method
    """
    # Define parameters and construct base
    pytree = create_base()

    # Test the zero_nan_check method
    zodiax.optimisation.zero_nan_check(pytree)


def test_get_optimiser(create_base):
    """
    tests the get_optimiser method
    """
    # Define parameters and construct base
    pytree = create_base()

    # Optimisers
    optimisers = {
        "param": zodiax.optimisation.sgd(1),
        "b.param": zodiax.optimisation.adam(10),
    }

    # Test paths
    zodiax.optimisation.get_optimiser(pytree, optimisers)


# NOTE: This test is for the old version
# def test_get_optimiser(create_base):
#     """
#     tests the get_optimiser method
#     """
#     # Define parameters and construct base
#     pytree = create_base()

#     # Test paths
#     for path, optimiser in zip(paths, optimisers):
#         zodiax.optimisation.get_optimiser(pytree, path, optimiser)
