# Diff Ops

The `zodiax.diffops` module provides memory-conscious wrappers around JAX's Jacobian
and Hessian utilities. `jacobian` and `hessian` return serialisable `Jacobian` and
`Hessian` objects whose `.matrix` contains the realised derivative and whose single
`.layout` maps parameter coordinates back to tree leaves.

This typed return replaces tuple unpacking into `(matrix, unflatten)`. Use
`result.matrix` for the derivative array and `result.layout.unflatten(vector)` for a
path-keyed parameter view. Jacobians accept one floating array-like output and retain
its axes, giving shape `f(x).shape + (result.layout.size,)`; `columns(nested=True)`
maps the final parameter axis to nested leaves. Hessians require a floating scalar
output and expose square parameter blocks through `blocks(nested=True)`.

::: zodiax.diffops
