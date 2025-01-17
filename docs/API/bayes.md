# Fisher

The Fisher module is designed to ease the calculation of things like covariance and Fisher matrices in differentiable ways. These calculations are built off the `hessian` function. The `fisher_matrix` and `covariance_matrix` functions take in a pytree, parameters, a log likelihood function and data.

!!! info "Full API"
    ::: zodiax.bayes