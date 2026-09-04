# Derivatives

The `zodiax.derivatives` package combines PyTree-aware derivative operations with
serialisable containers. `TreeLayout` associates parameter leaves with one ordered
coordinate axis. Dots separate nested paths, so literal mapping keys containing
dots are rejected.

`jacobian`, `hessian`, and `gauss_newton` provide memory-conscious wrappers around
JAX and return typed `Jacobian`, `Hessian`, and `GaussNewton` objects. Their arrays
retain the realised numerical values, while the shared layout maps derivative axes
back to model paths and shapes. No JAX transformation closures are stored.

The complete second argument is the parameter PyTree: every leaf is differentiated
and appears in the resulting layout. All parameter leaves must therefore be
floating. Integer, Boolean, complex, key, or non-array leaves raise rather than
silently changing parameter membership. Fixed configuration should be captured by
the differentiated function or supplied through its surrounding model context.

## Example

```python
import jax.numpy as jnp
import zodiax as zdx

parameters = {
    "layer": {
        "weights": jnp.array([1.0, 2.0], dtype=jnp.float32),
        "bias": jnp.array(0.5, dtype=jnp.float32),
    },
    "power": jnp.array(2.0, dtype=jnp.float32),
}


def function(values):
    layer = values["layer"]
    return layer["weights"] ** values["power"] + layer["bias"]


def scalar_loss(values):
    return jnp.sum(function(values) ** 2)


observations = jnp.array([1.2, 4.1], dtype=jnp.float32)
residuals = lambda values: function(values) - observations
covariance = jnp.diag(jnp.array([0.2, 0.3], dtype=jnp.float32) ** 2)

jacobian = zdx.jacobian(function, parameters)
hessian = zdx.hessian(scalar_loss, parameters)
gauss_newton = zdx.gauss_newton(residuals, parameters, cov=covariance)

print(jacobian)
# Jacobian(
#   matrix=f32[2,4],
#   layout=TreeLayout(
#     paths=('layer.bias', 'layer.weights', 'power'), shapes=((), (2,), ())
#   )
# )

print(hessian)
# Hessian(
#   matrix=f32[4,4],
#   layout=TreeLayout(
#     paths=('layer.bias', 'layer.weights', 'power'), shapes=((), (2,), ())
#   )
# )

print(gauss_newton)
# GaussNewton(
#   matrix=f32[4,4],
#   layout=TreeLayout(
#     paths=('layer.bias', 'layer.weights', 'power'), shapes=((), (2,), ())
#   )
# )

# Map the parameter axis, square blocks, or diagonal back to named leaves.
jacobian.columns(nested=True)
hessian.blocks(nested=True)
hessian.diagonal().nested_dict()
gauss_newton.blocks(nested=True)
```

The four derivative columns correspond to scalar `layer.bias`, the two elements of
`layer.weights`, and scalar `power`.

`nbatches` divides derivative columns into fixed blocks to reduce peak memory;
`checkpoint=True` trades additional computation for memory. Jacobians accept one
floating array-like output and preserve its axes. Exact Hessians require one floating
scalar output. Gauss--Newton matrices require an unreduced floating residual output.

## Exact Hessian methods

`hessian` supports two mixed-mode automatic-differentiation compositions:

```python
forward_over_reverse = zdx.hessian(
    scalar_loss,
    parameters,
    method="fwd-rev",
    nbatches=4,
)
reverse_over_forward = zdx.hessian(
    scalar_loss,
    parameters,
    method="rev-fwd",
    nbatches=4,
)
```

Both calculate the same exact Hessian for a smooth scalar function. `"fwd-rev"` is
the default: it linearises a reverse-mode gradient once and normally has less
overhead for scalar objectives. Its retained gradient linearisation is reused across
the parameter-column blocks.

`"rev-fwd"` reverse-differentiates a scalar directional forward derivative. The
outer reverse transformation acts on the larger primal-and-tangent computation, so
this method is usually slower and may repeat more work across blocks. It remains
useful when a model has unusually favourable forward derivative rules or memory
behaviour, and for compatibility tests and independent numerical cross-checks. It
requires forward-mode derivative support; a custom-VJP-only operation may therefore
be incompatible. The methods differ in execution rather than mathematical accuracy.

## Gauss--Newton Hessians

`gauss_newton` calculates

\[
G(x) = J_r(x)^\mathsf{T} C^{-1} J_r(x)
\]

from the Jacobian of a residual function. It applies JVPs, residual weighting, and
the transposed linearisation in parameter-column blocks, so the full residual
Jacobian is never materialised.

!!! warning "Pass residuals, not a scalar loss"

    `residual_fn(x)` must return the complete, unreduced residual array. Do not pass
    a loss that sums, averages, squares, or otherwise reduces those residuals. A
    scalar return value is accepted because a problem may genuinely have one scalar
    residual; Zodiax cannot distinguish that case from an incorrectly supplied
    scalar objective.

```python
# Correct: the individual residuals remain visible to the linearisation.
residual_fn = lambda values: function(values) - observations
G = zdx.gauss_newton(residual_fn, parameters, cov=covariance)

# Incorrect: this is a reduced scalar objective, not a residual function.
loss_fn = lambda values: jnp.sum(residual_fn(values) ** 2)
G_wrong = zdx.gauss_newton(loss_fn, parameters)
```

For the constant-covariance weighted least-squares objective

\[
L(x) = \frac{1}{2} r(x)^\mathsf{T} C^{-1} r(x),
\]

the exact Hessian is

\[
\nabla^2 L(x)
= J_r(x)^\mathsf{T} C^{-1}J_r(x)
+ \sum_i \left(C^{-1}r(x)\right)_i\nabla^2r_i(x).
\]

Gauss--Newton omits the second term. It is exact for affine residual functions and
agrees with the exact Hessian at a zero-residual solution. It is often a useful
positive-semidefinite local approximation when residuals are small or the model is
locally linear. It can be misleading far from a solution, for strongly nonlinear
residuals, when negative curvature matters, or when the covariance depends on the
parameters. Use `hessian` on the complete scalar objective when all curvature terms
are required.

A Gauss--Newton matrix coincides with Fisher information only under additional
statistical assumptions. Zodiax therefore returns a `GaussNewton`, not a `Fisher`.

### Covariance or inverse covariance

Pass `cov` when the residual covariance itself is available:

```python
G = zdx.gauss_newton(residual_fn, parameters, cov=covariance)
```

Zodiax applies its inverse action with a linear solve; it does not explicitly form
`C^-1`. This is generally more stable and avoids constructing a full inverse when
only its action on residual responses is required. `cov` should be symmetric and
positive definite.

Pass `inv_cov` when a precision matrix has already been calculated and can be reused:

```python
precision = jnp.linalg.inv(covariance)
G = zdx.gauss_newton(residual_fn, parameters, inv_cov=precision)
```

This replaces each covariance solve with direct multiplication and is often faster
when the same precision is reused across many parameter points. Explicit inversion
can be less stable for poorly conditioned covariance matrices, so it should normally
be performed once outside `gauss_newton` using an appropriate factorisation. `cov`
and `inv_cov` are mutually exclusive; omitting both selects identity weighting. Both
matrices have shape `(residuals.size, residuals.size)` under row-major residual
flattening and are treated as constant with respect to the parameters.

## Legacy Hessian conversion

`hessian_to_pytree` is deprecated as of version 0.5.0. It remains available during
the compatibility period for callers that need the original PyTree-of-PyTrees
representation. New code should use `Hessian.blocks(nested=True)`, which reads the
layout already stored on the realised result. A raw matrix can first be wrapped as
`Hessian(matrix, TreeLayout.from_tree(parameters))`.

::: zodiax.derivatives
