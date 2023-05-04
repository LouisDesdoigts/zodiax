# Bayes

The bayesian module is designed to ease the calculation of things like convariance and fisher matrices in differentiable ways. It implements two likelihood functions `poiss_loglike` and `chi2_loglike`. They both take in a pytree and data and return a scalar log likelihood. The `poiss_loglike` function assumes the data is poisson distributed and the `chi2_loglike` function assumes the data is normally distributed. To use these functions the input pytree _must_ have a `.model()` function.

There are also four functions used to calcualte fisher and covariances matrices: `fisher_matrix`, `covariance_matrix`, `self_fisher_matrix`, `self_covariance_matrix`. The `fisher_matrix` and `covariance_matrix` functions take in a pytree, parameters, a log likelihood function and data. They return the fisher and covariance matrices respectively. The `self_fisher_matrix` and `self_covariance_matrix` functions take in a pytree, parameters and a log likelihood function. They return the fisher and covariance matrices respectively, but the data is generated from the model itself.

!!! info "Full API"
    ::: zodiax.bayes