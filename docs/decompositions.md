# Matrix factors and local parameter coordinates

The [Linear walkthrough](derivatives.md#2-use-the-curvature-for-factors-and-parameter-steps)
calculates `H`, `projection`, and `geometry` from one parameter dictionary. This
reference summarises the operations used in that example.

## 1. Factor the curvature

Factors reuse a symmetric matrix for reconstruction, linear solves, and parameter steps.

| Object | Meaning | Main operations |
|---|---|---|
| EigenDecomposition | Eigenvalues and column eigenvectors of a symmetric matrix | `H.eigh()`, `reconstruct()` |
| CholeskyDecomposition | Lower factor `L`, with `H.matrix == L @ L.T` | `H.cholesky()`, `solve(value)`, `log_determinant` |
| ParameterProjection | Matrix `P` mapping a latent vector into a parameter step | `H.projection()`, `apply(latent)`, `solve(step)` |
| LocalParameterisation | A projection bound to the original parameter dictionary | `projection.bind(parameters)`, call, `step`, `encode` |

All four belong to the Decomposition family. Eigen decomposition supports
indefinite Hessians. Cholesky requires positive definiteness; an eigen projection
requires positive semidefiniteness. The optional eager `H.check(method)` validates
the corresponding numerical domain. Factor methods never call it implicitly.

For a full-rank, undamped positive matrix `M`, the projection satisfies
`P.T @ M @ P == I` and `P @ P.T == inv(M)`, within floating-point accuracy.
It is a coordinate map, rather than an idempotent orthogonal projector.

## 2. Choose rank and damping

Projection options determine which directions are represented and how strongly they are scaled.

By default, `equilibrate=True` rescales positive diagonal entries towards one before
factorisation. Nonpositive diagonal entries use unit scale. `equilibrate=False`
factors in the original parameter units.

| Option | Eigen method `"eigh"` | Cholesky method `"cholesky"` |
|---|---|---|
| `rtol` | Retain positive eigenvalues above `rtol * largest_eigenvalue` | Ignored; all modes retained |
| `damping` | Add `damping * largest_eigenvalue` to retained eigenvalues | Add `damping` directly to the scaled diagonal |
| Required matrix | Positive semidefinite | Positive definite after damping |

Damping acts after scaling and changes the represented inverse metric. The two
methods use different damping conventions; eigen damping does not restore discarded
modes. `rtol` and `damping` are finite nonnegative Python scalars, and all options
are fixed during JAX tracing.

Discarded columns remain zero so shapes stay fixed. A nonempty zero metric has rank
zero: steps, recovered latent coordinates, and represented covariance are zero.
Factorisation entry points reject empty parameter spaces.

The induced covariance is `P @ P.T`. With truncation and equilibration it is
generally not the Euclidean Moore–Penrose inverse of the original matrix.
Zero contribution from a discarded direction does not imply zero statistical
uncertainty in an unconstrained parameter.

## 3. Apply steps and transform gradients

Parameter steps and gradients have different coordinate transformations.

| Method | Calculation | Result |
|---|---|---|
| `projection.apply(u)` | `P @ u` | Parameter TreeVector |
| `projection.solve(v)` | Minimum-norm least-squares solution of `P @ u = v` | Latent Array |
| `geometry(u)` | Original parameters plus `P @ u` | Parameter dictionary |
| `geometry.step(u)` | `P @ u` | Parameter-step dictionary |
| `geometry.encode(parameters)` | Solve for the change from the origin | Latent Array |
| `geometry.pullback(gradient)` | `P.T @ gradient` | Latent gradient Array |
| `geometry.natural_gradient(gradient)` | `P @ P.T @ gradient` | Parameter-gradient dictionary |
| `geometry.natural_norm(gradient)` | Square root of `gradient.T @ P @ P.T @ gradient` | Scalar Array |

The solve trusts the originally selected rank and keeps discarded latent entries
zero. A manually constructed projection must therefore have independent retained
columns and zero discarded columns.

The walkthrough uses dictionaries; LocalParameterisation also preserves supported
PyTree structures and static metadata. Numerical outputs use JAX's configured
default float. `variance` and `standard_deviation` return the diagonal of the
represented covariance and its square root in that structure. Reconstruction
functions are temporary and are not stored on the objects.

## 4. Differentiate within the supported domain

Applying a fixed factor is separate from differentiating its construction.

Applying a projection, or differentiating a fixed-factor solve with respect to its
right-hand side, follows ordinary JAX differentiation. Rank changes are discrete.
Eigenvector derivatives can be undefined at repeated eigenvalues, including those
introduced by equilibration. Differentiating `solve` with respect to its factor
also inherits SVD limits at repeated singular values.

Cholesky construction is smooth on its positive-definite domain. A mathematically
smooth covariance does not guarantee smooth derivatives through a chosen spectral
factorisation. The [typed contracts](derivatives.md#54-decompositions-and-coordinate-maps)
and `tests/derivatives/test_decompositions.py` define the supported shape, dtype,
rank, and transformation behaviours.
