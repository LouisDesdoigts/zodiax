"""Dense derivatives calculated in reusable parameter-column blocks.

Every leaf of the supplied parameter tree is differentiated and must be floating.
Realised arrays and noise inputs use the configured default floating dtype; the
parameter tree itself is never coerced or filtered.
"""

from collections.abc import Callable
from operator import index
from typing import Any, Literal
from warnings import warn

import equinox as eqx
import jax
import jax.numpy as np
from jax import Array
from jax.flatten_util import ravel_pytree

from ..numerics.arrays import as_array
from .containers import (
    GaussNewton,
    Hessian,
    Jacobian,
    TreeLayout,
    _standard_deviation,
)

PyTree = Any

__all__ = [
    "jacobian",
    "hessian",
    "gauss_newton",
    "hessian_to_pytree",
]


def _flatten_parameters(
    x: PyTree,
) -> tuple[TreeLayout, Array, Callable[[Array], PyTree]]:
    """Check parameter membership, retaining each floating leaf's original dtype."""
    for leaf in jax.tree.leaves(x):
        array = as_array(leaf)
        if not eqx.is_array(array) or not np.issubdtype(array.dtype, np.floating):
            raise TypeError("Every parameter leaf must be floating and array-like.")

    layout = TreeLayout.from_tree(x)
    vector, unravel = ravel_pytree(x)
    return layout, vector, unravel


def _floating_output(value: Any, name: str) -> Array:
    """Convert a function result without coercing a non-floating calculation."""
    try:
        output = as_array(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must return one floating array-like value.") from error
    if not eqx.is_array(output) or not np.issubdtype(output.dtype, np.floating):
        raise TypeError(f"{name} must return one floating array-like value.")
    return output


def _get_batch_sizes(n: int, nbatches: int) -> tuple[Array, int]:
    """Build fixed-size blocks of column indices, padding the final block."""
    if isinstance(nbatches, bool):
        raise TypeError("nbatches must be a positive integer.")
    try:
        nbatches = index(nbatches)
    except TypeError as error:
        raise TypeError("nbatches must be a positive integer.") from error
    if nbatches < 1:
        raise ValueError("nbatches must be a positive integer.")
    if n == 0:
        raise ValueError("The parameter tree must contain at least one coordinate.")

    # Convert "nbatches" into a fixed block size
    batch_size = (n + nbatches - 1) // nbatches
    total = nbatches * batch_size
    pad = total - n

    # Each block contains at least one real parameter column.
    if nbatches > n:
        raise ValueError(
            f"nbatches={nbatches} is too large for n={n} parameters. "
            f"Choose nbatches <= n, or set nbatches=1 for no batching."
        )

    # Pad the indices to a full grid and reshape into batches
    idx = np.arange(n, dtype=int)
    if pad:
        idx = np.pad(idx, (0, pad), constant_values=0)
    idx = idx.reshape(nbatches, batch_size)

    return idx, total


def _materialise_columns(
    product: Callable[[Array], Array],
    x_flat: Array,
    indices: Array,
    total: int,
    *,
    batched: bool = False,
) -> Array:
    """Apply a linear product to basis vectors in fixed column blocks."""
    n = x_flat.size

    def step(carry, idxs):
        basis = jax.nn.one_hot(idxs, n, dtype=x_flat.dtype)
        outputs = product(basis) if batched else jax.vmap(product)(basis)
        columns = np.moveaxis(outputs, 0, -1)
        return carry, columns

    _, blocks = jax.lax.scan(step, None, indices)
    blocks = np.moveaxis(blocks, 0, -2)
    return blocks.reshape(*blocks.shape[:-2], total)[..., :n]


def _square_matrix(value: Any, name: str, size: int) -> Array:
    """Convert one covariance or precision array and check its residual axes."""
    matrix = as_array(value, dtype=float)
    if matrix.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size}).")
    return matrix


def jacobian(
    f: Callable,
    x: PyTree,
    nbatches: int = 1,
    jit: bool = True,
    checkpoint: bool = False,
) -> Jacobian:
    """
    A batched version of `jax.jacobian` that computes the Jacobian in column blocks
    to reduce peak memory. Increase `nbatches` to reduce block size. Set
    `checkpoint=True` to trade extra computation for further memory savings. `f(x)`
    must return one floating array-like value. Its output contract is checked
    while tracing the differentiated calculation.

    Parameters
    ----------
    f : callable
        The function to differentiate. It must accept a pytree with the same
        structure as `x` and return one floating array-like value.
    x : PyTree
        The point at which to evaluate the Jacobian.
    nbatches : int = 1
        Number of column blocks. Higher values use less memory.
    jit : bool = True
        Whether to JIT-compile the inner function.
    checkpoint : bool = False
        Whether to apply `jax.checkpoint` to `f` to reduce memory at the cost
        of extra computation.

    Returns
    -------
    jacobian : Jacobian
        Realised Jacobian with shape `f(x).shape + (n,)`, where `n` is the flattened
        parameter size. Its single layout describes the final parameter axis.
    """

    # Flatten parameter coordinates without changing the original leaf dtypes.
    layout, x_flat, unravel = _flatten_parameters(x)
    indices, total = _get_batch_sizes(x_flat.size, nbatches)

    def function(vector):
        return _floating_output(f(unravel(vector)), "f(x)")

    function = jax.checkpoint(function) if checkpoint else function
    function = jax.jit(function) if jit else function

    # A single block uses JAX directly; otherwise reuse one linearisation.
    if indices.shape[0] == 1:
        matrix = jax.jacobian(function)(x_flat)
    else:
        _, product = jax.linearize(function, x_flat)
        product = jax.jit(product) if jit else product
        matrix = _materialise_columns(product, x_flat, indices, total)
    return Jacobian(matrix, layout)


def hessian(
    f: Callable,
    x: PyTree,
    nbatches: int = 1,
    jit: bool = True,
    checkpoint: bool = False,
    *,
    method: Literal["fwd-rev", "rev-fwd"] = "fwd-rev",
) -> Hessian:
    """
    Calculate an exact Hessian in column blocks using forward-over-reverse or
    reverse-over-forward automatic differentiation. Increase `nbatches` to reduce
    block size. Set `checkpoint=True` to trade extra computation for further memory
    savings. `f(x)` must return a scalar; its output contract is checked while tracing
    the differentiated calculation.

    Parameters
    ----------
    f : callable
        The scalar-valued function to differentiate twice. Must accept a pytree
        of the same structure as `x`.
    x : PyTree
        The point at which to evaluate the Hessian.
    nbatches : int = 1
        Number of column blocks. Higher values use less memory.
    jit : bool = True
        Whether to JIT-compile the inner function.
    checkpoint : bool = False
        Whether to apply `jax.checkpoint` to `f` to reduce memory at the cost
        of extra computation.
    method : {"fwd-rev", "rev-fwd"} = "fwd-rev"
        Automatic-differentiation composition. ``"fwd-rev"`` linearises the
        reverse-mode gradient and is normally faster for scalar objectives.
        ``"rev-fwd"`` reverse-differentiates directional forward derivatives. It
        usually has more overhead, but can be useful for model-specific derivative
        rules, memory profiles, compatibility testing, and numerical cross-checks.
        Both methods calculate the same exact Hessian for a smooth function.

    Returns
    -------
    hessian : Hessian
        Realised Hessian with shape `(n, n)` and the shared parameter layout of both
        axes, where `n` is the flattened parameter size.
    """

    layout, x_flat, unravel = _flatten_parameters(x)
    indices, total = _get_batch_sizes(x_flat.size, nbatches)
    if method not in ("fwd-rev", "rev-fwd"):
        raise ValueError("method must be 'fwd-rev' or 'rev-fwd'.")

    def function(vector):
        output = _floating_output(f(unravel(vector)), "f(x)")
        if output.shape != ():
            raise ValueError("f(x) must return one floating scalar.")
        return output

    function = jax.checkpoint(function) if checkpoint else function
    function = jax.jit(function) if jit else function

    if method == "fwd-rev" and indices.shape[0] == 1:
        return Hessian(jax.hessian(function)(x_flat), layout)

    # Build the requested Hessian-vector product, keeping closures temporary.
    if method == "fwd-rev":
        _, product = jax.linearize(jax.grad(function), x_flat)
    else:

        def product(direction):
            def directional(vector):
                return jax.jvp(function, (vector,), (direction,))[1]

            return jax.grad(directional)(x_flat)

    product = jax.jit(product) if jit else product
    matrix = _materialise_columns(product, x_flat, indices, total)
    return Hessian(matrix, layout)


def gauss_newton(
    residual_fn: Callable,
    x: PyTree,
    *,
    std: Any | None = None,
    cov: Any | None = None,
    inv_cov: Any | None = None,
    nbatches: int = 1,
    jit: bool = True,
    checkpoint: bool = False,
) -> GaussNewton:
    """Calculate a Gauss--Newton Hessian from unreduced residuals.

    This function materialises ``J.T @ C^-1 @ J`` using chunked JVP--VJP
    products, without materialising the residual Jacobian ``J``. At most one of
    ``std``, ``cov``, and ``inv_cov`` may be supplied. Independent noise supplied
    through ``std`` uses elementwise inverse-variance weighting without constructing
    a dense diagonal matrix. When ``cov`` is supplied, its inverse action is
    evaluated with a linear solve; a supplied ``inv_cov`` is applied directly.
    Omitting all three selects identity weighting.

    ``residual_fn`` must return the complete, unreduced floating residual array. A
    scalar output is accepted for genuine one-residual problems, but a scalar loss
    formed by summing, averaging, squaring, or otherwise reducing residuals is not a
    valid input. For nonlinear residuals, the Gauss--Newton matrix is not generally
    the exact Hessian of the corresponding scalar loss: it omits residual-weighted
    second derivatives. It is most accurate for locally linear models or near a
    small-residual solution. It is only identifiable with Fisher information under
    additional statistical assumptions.

    Parameters
    ----------
    residual_fn : callable
        Function accepting a parameter PyTree like ``x`` and returning one floating
        residual array. Array axes are flattened in row-major order for weighting.
    x : PyTree
        Point at which to evaluate the residual Jacobian.
    std : array-like, optional
        Positive standard deviation of independent residual noise. A scalar or an
        array broadcastable to ``residual_fn(x).shape`` is accepted. This is the
        fast path for uncorrelated Gaussian noise. Mutually exclusive with ``cov``
        and ``inv_cov``.
    cov : array-like, optional
        Constant residual covariance with shape ``(m, m)``, where ``m`` is the
        flattened residual size. It should be symmetric and positive definite. A
        solve is performed for every product block. Mutually exclusive with
        ``inv_cov``.
    inv_cov : array-like, optional
        Precomputed constant inverse covariance with shape ``(m, m)``. It should be
        symmetric and positive semidefinite. Mutually exclusive with ``cov``.
    nbatches : int = 1
        Number of parameter-column blocks. Higher values use less temporary memory.
    jit : bool = True
        Whether to JIT-compile the residual and product functions.
    checkpoint : bool = False
        Whether to apply `jax.checkpoint` to ``residual_fn`` to trade computation
        for memory.

    Returns
    -------
    gauss_newton : GaussNewton
        Realised Gauss--Newton matrix with shape ``(n, n)`` and the shared parameter
        layout of both axes, where ``n`` is the flattened parameter size.

    Notes
    -----
    For ``loss(x) = 0.5 * r(x).T @ C^-1 @ r(x)`` with constant covariance,
    the exact Hessian additionally contains
    ``sum_i (C^-1 @ r(x))[i] * hessian(r_i)(x)``. Use :func:`hessian` on the full
    scalar loss when that correction or other objective terms are required.
    """

    if sum(value is not None for value in (std, cov, inv_cov)) > 1:
        raise ValueError("Pass only one of std, cov, or inv_cov.")

    layout, x_flat, unravel = _flatten_parameters(x)
    indices, total = _get_batch_sizes(x_flat.size, nbatches)

    def residuals(vector):
        output = _floating_output(residual_fn(unravel(vector)), "residual_fn(x)")
        if std is None:
            return output.reshape(-1)

        # Whiten before differentiating, so both sides of J.T @ J are scaled.
        # Forming std**2 first can overflow or underflow for finite noise scales.
        noise = _standard_deviation(std, output.shape)
        return output.reshape(-1) / noise

    residuals = jax.checkpoint(residuals) if checkpoint else residuals
    residuals = jax.jit(residuals) if jit else residuals

    # Freeze one residual linearisation and transpose that exact linear map.
    residual, jvp = jax.linearize(residuals, x_flat)
    vjp = jax.linear_transpose(jvp, x_flat)
    if cov is not None:
        cov = _square_matrix(cov, "cov", residual.size)
    if inv_cov is not None:
        inv_cov = _square_matrix(inv_cov, "inv_cov", residual.size)

    def product(basis):
        responses = jax.vmap(jvp)(basis)
        if std is not None:
            # Keep the two whitening stages separate: compiler reassociation of
            # their reciprocal factors can otherwise recreate 1 / std**2.
            responses = jax.lax.optimization_barrier(responses)
        if cov is not None:
            responses = np.linalg.solve(cov, responses.T).T
        elif inv_cov is not None:
            responses = responses @ inv_cov.T
        return jax.vmap(lambda response: vjp(response)[0])(responses)

    product = jax.jit(product) if jit else product
    matrix = _materialise_columns(product, x_flat, indices, total, batched=True)
    return GaussNewton(matrix, layout)


def hessian_to_pytree(H: Array | Hessian, x: PyTree) -> PyTree:
    """
    Convert a flat Hessian into a legacy pytree-of-pytrees representation.

    .. deprecated:: 0.5.0
        Use :meth:`Hessian.blocks` to access parameter blocks from the realised
        derivative container. Pass ``nested=True`` for nested dictionaries.

    Parameters
    ----------
    H : Array or Hessian
        A realised Hessian or flat `(n, n)` Hessian matrix where
        `n` is the total size of the parameter leaves in `x`.
    x : PyTree
        The pytree whose structure defines the block partition of `H`.

    Returns
    -------
    H_tree : PyTree
        A pytree-of-pytrees with the same structure as `x` twice over, where
        each leaf block `H_tree[i][j]` has shape `leaf_i.shape + leaf_j.shape`.
    """

    warn(
        "hessian_to_pytree is deprecated as of v0.5.0; use "
        "Hessian.blocks(nested=True) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    layout = TreeLayout.from_tree(x)
    if isinstance(H, Hessian):
        if not H.layout.compatible(layout):
            raise ValueError("Hessian layout is not compatible with x.")
        matrix = H.matrix
    else:
        matrix = as_array(H, dtype=float)
    if matrix.shape != (layout.size, layout.size):
        raise ValueError(
            f"H has shape {matrix.shape}, but x flattens to {layout.size} elements. "
            "Did you pass the same x used to compute H?"
        )

    # The existing helper preserves actual container types on both axes. Layout
    # views instead return dictionaries, so this main-API conversion stays local.
    structure = jax.tree.structure(x)
    rows = []
    for row_slice, row_shape in zip(layout.slices, layout.shapes):
        row = []
        for column_slice, column_shape in zip(layout.slices, layout.shapes):
            block = matrix[row_slice, column_slice]
            row.append(block.reshape(row_shape + column_shape))
        rows.append(structure.unflatten(row))
    return structure.unflatten(rows)
