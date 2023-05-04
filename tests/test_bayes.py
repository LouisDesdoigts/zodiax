from __future__ import annotations
from jax import config
config.update("jax_debug_nans", True)
import jax.numpy as np
import zodiax
import pytest


# Paths
paths = [
    'param',
    'b.param',
    ['param', 'b.param'],
]


def test_poiss_loglike(create_base):
    pytree = create_base()
    data = pytree.model()
    zodiax.bayes.poiss_loglike(pytree, data)


def test_chi2_loglike(create_base):
    pytree = create_base()
    data = pytree.model()
    zodiax.bayes.chi2_loglike(pytree, data, noise=2)


def test_fisher_matrix(create_base):
    pytree = create_base()
    data = pytree.model()
    loglike_fn = zodiax.bayes.poiss_loglike
    shape_dict = {'param': (1,)}

    for param in paths:
        zodiax.bayes.fisher_matrix(pytree, param, loglike_fn, data, 
            shape_dict=shape_dict)


def test_covariance_matrix(create_base):
    pytree = create_base()
    data = pytree.model()
    loglike_fn = zodiax.bayes.poiss_loglike
    shape_dict = {'param': (1,)}

    for param in paths:
        zodiax.bayes.covariance_matrix(pytree, param, loglike_fn, data, 
            shape_dict=shape_dict)


def test_self_fisher_matrix(create_base):
    pytree = create_base()
    loglike_fn = zodiax.bayes.poiss_loglike
    shape_dict = {'param': (1,)}

    for param in paths:
        zodiax.bayes.self_fisher_matrix(pytree, param, loglike_fn, 
            shape_dict=shape_dict)


def test_self_covariance_matrix(create_base):
    pytree = create_base()
    loglike_fn = zodiax.bayes.poiss_loglike
    shape_dict = {'param': (1,)}

    for param in paths:
        zodiax.bayes.self_covariance_matrix(pytree, param, loglike_fn, 
            shape_dict=shape_dict)