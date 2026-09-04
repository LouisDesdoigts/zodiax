from collections.abc import Callable
from operator import index
from typing import Literal, Union
from warnings import warn

import equinox as eqx
import jax
import jax.numpy as np
from jax import Array
from jax.flatten_util import ravel_pytree

from .containers import GaussNewton, Hessian, Jacobian, TreeLayout

PyTree = Union[dict, list, tuple, eqx.Module]

__all__ = [
    "jacobian",
    "hessian",
    "gauss_newton",
    "hessian_to_pytree",
]


def _get_batch_sizes(n: int, nbatches: int) -> tuple[Array, int]:
    if isinstance(nbatches, bool):
        raise TypeError("nbatches must be a positive integer.")
    try:
        nbatches = index(nbatches)
    except TypeError as error:
        raise TypeError("nbatches must be a positive integer.") from error
    if nbatches < 1:
        raise ValueError("nbatches must be a positive integer.")

    # Convert "nbatches" into a fixed block size
    batch_size = (n + nbatches - 1) // nbatches
    total = nbatches * batch_size
    pad = total - n

    # Check that we dont have more batches than parameters
    if nbatches > n:
        raise ValueError(
            f"nbatches={nbatches} is too large for n={n} parameters. "
            f"Choose nbatches <= n, or set nbatches=1 for no batching."
        )

    # Pad the indices to a full grid and reshape into batches
    idx = np.arange(n, dtype=int)
    if pad:
        idx = np.pad(idx, (0, pad), constant_values=0)
    idx = idx.reshape(nbatches, batch_size)  # (nbatches, batch_size)

    return (idx, total)  # batch_size


def _materialise_columns(
    product: Callable,
    x_flat: Array,
    nbatches: int,
    *,
    batched: bool = False,
) -> Array:
    """Apply a linear product to basis vectors in fixed column blocks."""
    n = x_flat.size
    idx, total = _get_batch_sizes(n, nbatches)

    def step(carry, idxs):
        basis = jax.nn.one_hot(idxs, n, dtype=x_flat.dtype)
        outputs = product(basis) if batched else jax.vmap(product)(basis)
        columns = np.moveaxis(outputs, 0, -1)
        return carry, columns

    _, blocks = jax.lax.scan(step, None, idx)
    blocks = np.moveaxis(blocks, 0, -2)
    return blocks.reshape(*blocks.shape[:-2], total)[..., :n]


def _square_matrix(value, name: str, size: int) -> Array:
    """Validate one residual-space covariance matrix."""
    try:
        matrix = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be one floating array-like matrix.") from error
    if not np.issubdtype(matrix.dtype, np.floating):
        raise TypeError(f"{name} must be one floating array-like matrix.")
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
    must return one floating array-like value and is evaluated once eagerly to
    establish its fixed output shape before differentiation.

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
    # Every leaf in x is an explicit differentiation parameter.
    layout = TreeLayout.from_tree(x)
    x_flat, unflatten = ravel_pytree(x)
    _get_batch_sizes(x_flat.size, nbatches)
    try:
        output = np.asarray(f(x))
    except (TypeError, ValueError) as error:
        raise TypeError("f(x) must return one floating array-like value.") from error
    if not np.issubdtype(output.dtype, np.floating):
        raise TypeError("f(x) must return one floating array-like value.")

    def f_flat(z):
        return np.asarray(f(unflatten(z)))

    # Apply the requested memory transformations
    f_flat = jax.checkpoint(f_flat) if checkpoint else f_flat
    f_flat = jax.jit(f_flat) if jit else f_flat

    # Straight jax jacobian if only one batch (no batching overhead)
    if nbatches == 1:
        matrix = jax.jacobian(f_flat)(x_flat)
        return Jacobian(matrix, layout)

    # Use linearise to get the jvp without re-evaluating f for each column
    _, jvp = jax.linearize(f_flat, x_flat)
    jvp = jax.jit(jvp) if jit else jvp

    # Calculate and assemble the Jacobian columns
    matrix = _materialise_columns(jvp, x_flat, nbatches)

    # Package the realised result with its final-axis parameter layout
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
    savings. `f(x)` must return a scalar and is evaluated once eagerly to validate
    that contract before differentiation.

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
    # Every leaf in x is an explicit differentiation parameter.
    layout = TreeLayout.from_tree(x)
    x_flat, unflatten = ravel_pytree(x)
    _get_batch_sizes(x_flat.size, nbatches)
    try:
        output = np.asarray(f(x))
    except (TypeError, ValueError) as error:
        raise TypeError("f(x) must return one numerical scalar.") from error
    if output.shape != () or not np.issubdtype(output.dtype, np.floating):
        raise ValueError("f(x) must return one floating scalar.")
    if method not in ("fwd-rev", "rev-fwd"):
        raise ValueError("method must be 'fwd-rev' or 'rev-fwd'.")

    # Flatten input, checkpoint, and jit
    def f_flat(z):
        return f(unflatten(z))

    f_flat = jax.checkpoint(f_flat) if checkpoint else f_flat
    f_flat = jax.jit(f_flat) if jit else f_flat

    # Straight JAX Hessian for the default composition if there is only one block
    if method == "fwd-rev" and nbatches == 1:
        matrix = jax.hessian(f_flat)(x_flat)
        return Hessian(matrix, layout)

    if method == "fwd-rev":
        # Linearise the reverse-mode gradient once and reuse its forward product.
        _, hvp = jax.linearize(jax.grad(f_flat), x_flat)
    else:
        # Reverse-differentiate each scalar directional forward derivative.
        def hvp(vector):
            def directional(z):
                return jax.jvp(f_flat, (z,), (vector,))[1]

            return jax.grad(directional)(x_flat)

    hvp = jax.jit(hvp) if jit else hvp

    # Calculate and assemble the Hessian columns
    matrix = _materialise_columns(hvp, x_flat, nbatches)

    # Package the realised result with its shared parameter layout
    return Hessian(matrix, layout)


def gauss_newton(
    residual_fn: Callable,
    x: PyTree,
    *,
    cov: Array | None = None,
    inv_cov: Array | None = None,
    nbatches: int = 1,
    jit: bool = True,
    checkpoint: bool = False,
) -> GaussNewton:
    """Calculate a Gauss--Newton Hessian from unreduced residuals.

    This function materialises ``J.T @ C^-1 @ J`` using chunked JVP--VJP
    products, without materialising the residual Jacobian ``J``. Exactly one of
    ``cov`` and ``inv_cov`` may be supplied. When ``cov`` is supplied, its inverse
    action is evaluated with a linear solve; a supplied ``inv_cov`` is applied
    directly. Omitting both selects identity weighting.

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
    if cov is not None and inv_cov is not None:
        raise ValueError("Pass either cov or inv_cov, not both.")

    # Every leaf in x is an explicit differentiation parameter.
    layout = TreeLayout.from_tree(x)
    x_flat, unflatten = ravel_pytree(x)
    _get_batch_sizes(x_flat.size, nbatches)
    try:
        residual = np.asarray(residual_fn(x))
    except (TypeError, ValueError) as error:
        raise TypeError(
            "residual_fn(x) must return one floating residual array."
        ) from error
    if not np.issubdtype(residual.dtype, np.floating):
        raise TypeError("residual_fn(x) must return one floating residual array.")

    residual_size = residual.size
    cov = None if cov is None else _square_matrix(cov, "cov", residual_size)
    inv_cov = (
        None if inv_cov is None else _square_matrix(inv_cov, "inv_cov", residual_size)
    )

    def residual_flat(z):
        return np.asarray(residual_fn(unflatten(z))).reshape(-1)

    residual_flat = jax.checkpoint(residual_flat) if checkpoint else residual_flat
    residual_flat = jax.jit(residual_flat) if jit else residual_flat

    # Freeze one residual linearisation and transpose that exact linear map.
    _, jvp = jax.linearize(residual_flat, x_flat)
    vjp = jax.linear_transpose(jvp, x_flat)

    def product(basis):
        responses = jax.vmap(jvp)(basis)
        if cov is not None:
            responses = np.linalg.solve(cov, responses.T).T
        elif inv_cov is not None:
            responses = responses @ inv_cov.T
        return jax.vmap(lambda response: vjp(response)[0])(responses)

    product = jax.jit(product) if jit else product
    matrix = _materialise_columns(product, x_flat, nbatches, batched=True)
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
        H = H.matrix

    leaves, treedef = jax.tree_util.tree_flatten(x)

    # leaf sizes and shapes define the partition of the flat axis
    sizes = [int(np.size(leaf)) for leaf in leaves]
    shapes = [np.shape(leaf) for leaf in leaves]

    # build flat slices for each leaf
    starts = np.cumsum(np.array([0] + sizes[:-1], dtype=int))
    slices = [slice(int(s), int(s) + sz) for s, sz in zip(starts, sizes)]

    # sanity check (helps catch mismatched x vs H early)
    if H.shape != (sum(sizes), sum(sizes)):
        raise ValueError(
            f"H has shape {H.shape}, but x flattens to {sum(sizes)} elements. "
            "Did you pass the same x used to compute H?"
        )

    # assemble tree-of-trees
    rows = []
    for sli, shi in zip(slices, shapes):
        row = []
        for slj, shj in zip(slices, shapes):
            row.append(H[sli, slj].reshape(shi + shj))
        rows.append(treedef.unflatten(row))

    return treedef.unflatten(rows)
