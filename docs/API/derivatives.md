# Derivatives

`zodiax.derivatives` returns numerical derivative objects with parameter layouts.
The [walkthrough](../derivatives.md) contains executable examples and the complete
class hierarchy; the [factor reference](../decompositions.md) defines local
coordinate and gradient equations.

## Calculations

| API | Function and parameter contract | Result |
|---|---|---|
| `jacobian(f, x, nbatches=1, jit=True, checkpoint=False)` | Floating dynamic parameter leaves; one floating array output | Jacobian with `.matrix.shape == f(x).shape + (n,)` |
| `hessian(f, x, nbatches=1, jit=True, checkpoint=False, *, method="fwd-rev")` | Floating parameter leaves; one floating scalar objective | Hessian with `.matrix.shape == (n, n)` |
| `gauss_newton(residual_fn, x, *, std=None, cov=None, inv_cov=None, nbatches=1, jit=True, checkpoint=False)` | Floating parameters; unreduced floating residuals; fixed local noise weighting | GaussNewton with `.matrix.shape == (n, n)` |
| `J.fisher(*, std=None, covariance=None)` | Gaussian model Jacobian with parameter-independent noise | Fisher on J's layout |

`n` counts every scalar in the trainable parameter leaves. Fixed configuration
belongs outside those dynamic leaves. Function outputs are checked during derivative
tracing; calculations and decompositions require at least one parameter coordinate.

Noise arguments are mutually exclusive within each API. `std` broadcasts to the
output/residual shape and must be positive. Covariance is a symmetric
positive-definite `(m, m)` matrix, where `m` is the flattened output size; supplied
precision is symmetric positive semidefinite. Real numerical noise and stored-result
inputs, including integers, convert to the configured default floating dtype.
Trainable integer parameters remain unsupported.

## Stored results and factors

| Classes | Purpose and main accessors |
|---|---|
| Derivative | Shared matrix/layout base of Jacobian, Hessian, and GaussNewton; Fisher is separate |
| Decomposition | Field-free family of all four factor/coordinate-map classes |
| TreeLayout | Static coordinate order/shapes; `from_tree`, `from_paths`, `flatten`, `unflatten`, `split_axis` |
| TreeVector | Flat parameter values; `flat_dict`, `nested_dict` |
| TreeMatrix | Square parameter matrix; `rows`, `columns`, `blocks`, `diagonal`, norms, diagnostics, `check` |
| Jacobian | Arbitrary output axes plus a parameter axis; `columns`, norms, `fisher` |
| Hessian, GaussNewton, Fisher | Symmetric matrices; `eigh`, `cholesky`, `projection`, `parameterise` |
| EigenDecomposition | Ascending eigenvalues and column eigenvectors; `reconstruct` |
| CholeskyDecomposition | Lower factor; `reconstruct`, `solve`, `log_determinant` |
| ParameterProjection | Latent-to-parameter factor; `apply`, `solve`, `pullback`, natural-gradient methods, `bind` |
| LocalParameterisation | Bound origin tree; call, `step`, `encode`, gradient methods, variance and standard deviation |

Matrix `is_*` diagnostics return scalar JAX booleans. `check(method)` makes an
explicit eager validation decision; numerical methods do not call it implicitly.
See the [typed developer contracts](../derivatives.md#5-developer-reference) for
shape, dtype, rank, and transformation details and their associated test files.

## Migration from the main API

Main's `jacobian` and `hessian` returned `(array, unflatten_function)`. The new
functions return Jacobian and Hessian objects. Read `.matrix` for the array and use
`.columns()` or `.blocks()` for named parameter views. A layout's `unflatten`
returns dictionaries rather than recreating arbitrary original Module/list types;
no reconstruction closure is stored in a derivative result.

Top-level names such as `zdx.hessian` remain available. Module imports now come
from `zodiax.derivatives`, replacing `zodiax.diffops`.

`hessian_to_pytree(H, x)` remains available with its existing deprecated status.
It accepts a Hessian or a raw `(n, n)` matrix and returns the original
PyTree-of-PyTrees structure, with paired leaf shapes concatenated. Use it when
that exact container structure is required; new dictionary-based views normally
use `H.blocks(nested=True)`.

## API reference

::: zodiax.derivatives.operations

::: zodiax.derivatives.containers

::: zodiax.derivatives.decompositions
