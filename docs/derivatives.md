# Choosing a curvature calculation

Zodiax provides two kinds of second-order curvature calculation:

- `hessian` differentiates a scalar objective twice and returns its exact Hessian.
- `gauss_newton` linearises an unreduced residual function and returns the
  Gauss--Newton approximation to a weighted least-squares Hessian.

They answer different questions. The fact that both can use JVPs and VJPs does not
make their outputs interchangeable.

## Exact Hessians of scalar objectives

Use `hessian` when the function of interest is already a scalar objective and every
source of curvature should be retained:

```python
import jax.numpy as jnp
import zodiax as zdx

parameters = {
    "scale": jnp.array(1.2),
    "offset": jnp.array(-0.1),
}


def loss(values):
    prediction = values["scale"] ** 2 + values["offset"]
    return (prediction - 2.0) ** 2 + 0.1 * values["scale"] ** 4


H = zdx.hessian(loss, parameters, method="fwd-rev", nbatches=2)
```

The result includes negative curvature, nonlinear residual curvature, priors,
regularisation, parameter-dependent weighting, and any other differentiable terms
present in `loss`.

### Forward-over-reverse and reverse-over-forward

Both available methods are exact for a smooth function:

```python
H_fwd_rev = zdx.hessian(loss, parameters, method="fwd-rev")
H_rev_fwd = zdx.hessian(loss, parameters, method="rev-fwd")
```

`"fwd-rev"` takes forward directional derivatives of the reverse-mode gradient. It
is the default and normally the most efficient composition for a scalar objective.
The reverse-mode gradient is linearised once and that linearisation is reused while
the Hessian columns are assembled.

`"rev-fwd"` reverse-differentiates scalar directional forward derivatives. Reverse
mode therefore acts on the larger primal-and-tangent computation, which usually adds
runtime and compilation overhead. It can nevertheless be worth trying when a model
has specialised forward derivative rules, an unusual memory profile, or a failure in
one transform composition. Calculating both is also a useful derivative
cross-check. `"rev-fwd"` requires forward-mode support and is not compatible with an
operation that supplies only a custom VJP.

There is no general accuracy advantage to either method. Benchmark both methods on
the actual model if the default is too slow or memory-intensive.

## Gauss--Newton requires residuals

Suppose a problem has residuals (r(x)), constant covariance (C), and the
weighted least-squares objective

\[
L(x) = \frac{1}{2}r(x)^\mathsf{T}C^{-1}r(x).
\]

The Gauss--Newton matrix is

\[
G(x) = J_r(x)^\mathsf{T}C^{-1}J_r(x).
\]

Pass the complete residual array before it is squared or reduced:

```python
observations = jnp.array([1.0, 2.1, 2.9])
coordinates = jnp.array([-1.0, 0.0, 1.0])
covariance = jnp.array(
    [
        [0.04, 0.01, 0.00],
        [0.01, 0.09, 0.02],
        [0.00, 0.02, 0.04],
    ]
)


def residuals(values):
    prediction = values["scale"] * coordinates + values["offset"]
    return prediction - observations


G = zdx.gauss_newton(
    residuals,
    parameters,
    cov=covariance,
    nbatches=2,
)
```

Do not pass the reduced objective:

```python
# This is appropriate for zdx.hessian, but not zdx.gauss_newton.
def loss(values):
    residual = residuals(values)
    return 0.5 * residual @ jnp.linalg.solve(covariance, residual)


H = zdx.hessian(loss, parameters)
```

A scalar return from `gauss_newton` is not rejected because a dataset may genuinely
have one scalar residual. Zodiax cannot determine whether a scalar represents one
residual or an already reduced objective, so this distinction remains part of the
function contract.

### Why the approximation can differ

For constant covariance, the exact Hessian of the weighted least-squares loss is

\[
\nabla^2L(x)
= G(x)
+ \sum_i\left(C^{-1}r(x)\right)_i\nabla^2r_i(x).
\]

Gauss--Newton omits the residual-weighted second term. Consequently it is:

- exact when the residual function is affine;
- equal to the exact Hessian at a zero-residual solution;
- often useful near a small-residual solution or where the residual model is locally
  linear;
- positive semidefinite when the weighting is positive semidefinite.

Prefer the exact Hessian when residuals are large and nonlinear, negative curvature
matters, the covariance depends on the parameters, or the scalar objective contains
terms that are not represented by the residual function. Gauss--Newton is also not a
general construction for an arbitrary scalar objective.

Although a Gauss--Newton matrix can coincide numerically with Fisher information in
particular statistical models, that identification requires additional likelihood
and expectation assumptions. `gauss_newton` therefore returns `GaussNewton`, not
`Fisher`.

## Covariance weighting

Use `cov` when the covariance itself is available:

```python
G = zdx.gauss_newton(residuals, parameters, cov=covariance)
```

Zodiax solves against the covariance with all directions in the current parameter
block as right-hand sides. This avoids explicitly constructing an inverse and is
usually the more stable choice for a symmetric positive-definite covariance.

Use `inv_cov` when a precision matrix has already been calculated and will be reused:

```python
precision = jnp.linalg.inv(covariance)
G = zdx.gauss_newton(residuals, parameters, inv_cov=precision)
```

Applying a cached precision replaces each solve with matrix multiplication and can
be faster across repeated evaluations. Explicit inversion can lose accuracy for an
ill-conditioned covariance, so prefer an existing precision matrix over repeatedly
forming one inside a fit.

`cov` and `inv_cov` are mutually exclusive. Omitting both explicitly selects
identity weighting:

```python
G_identity = zdx.gauss_newton(residuals, parameters)
```

Both matrix forms use shape `(residuals.size, residuals.size)` after row-major
flattening and are treated as constants with respect to the parameters.

## Choosing the number of batches

All derivative operations use `nbatches` to divide parameter directions into fixed
blocks:

```python
H = zdx.hessian(loss, parameters, nbatches=2)
G = zdx.gauss_newton(residuals, parameters, cov=covariance, nbatches=2)
```

Increasing `nbatches` reduces the number of simultaneous tangent directions and the
associated temporary memory. It also introduces more sequential work. For
Gauss--Newton, each covariance solve receives all residual responses in the current
block as multiple right-hand sides, so larger blocks can make better use of dense
linear algebra.

Start with `nbatches=1`, increase it when memory is excessive, and benchmark the
resulting runtime. `checkpoint=True` provides another computation-for-memory trade by
rematerialising intermediates during differentiation.

## Accessing Hessian blocks

The returned `Hessian` already contains the parameter layout. Use its block view
instead of reconstructing a second PyTree from the flat matrix:

```python
hessian_blocks = H.blocks(nested=True)
```

`hessian_to_pytree` is deprecated as of version 0.5.0. It remains temporarily
available for legacy code that requires JAX's exact PyTree-of-PyTrees structure, but
new code should use `Hessian.blocks(nested=True)`. If starting from a raw matrix,
attach its layout explicitly first:

```python
layout = zdx.TreeLayout.from_tree(parameters)
H = zdx.Hessian(hessian_matrix, layout)
hessian_blocks = H.blocks(nested=True)
```

## Decision guide

Use `hessian` when:

- the natural input is a scalar loss;
- exact observed curvature is required;
- negative curvature or non-residual terms matter;
- or covariance depends on the parameters.

Use `gauss_newton` when:

- the complete residual vector is available;
- the objective is locally a constant-covariance weighted least-squares problem;
- a positive-semidefinite curvature approximation is desirable;
- or materialising the residual Jacobian is too expensive.
