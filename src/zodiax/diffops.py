import jax
import equinox as eqx
import jax.numpy as np
from jax import Array
from jax.flatten_util import ravel_pytree
from typing import Union

PyTree = Union[dict, list, tuple, eqx.Module]

__all__ = [
    "jacobian",
    "hessian",
    "hessian_to_pytree",
]


def _get_batch_sizes(n: int, nbatches: int) -> tuple[Array, int]:
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


def jacobian(
    f: callable,
    x: PyTree,
    nbatches: int = 1,
    jit: bool = True,
    checkpoint: bool = False,
) -> tuple[Array, callable]:
    """
    A batched version of `jax.jacobian` that computes the Jacobian in column blocks
    to reduce peak memory. Increase `nbatches` to reduce block size. Set
    `checkpoint=True` to trade extra computation for further memory savings.

    Parameters
    ----------
    f : callable
        The function to differentiate. Must accept a pytree of the same structure
        as `x`.
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
    J : Array
        The Jacobian of `f` at `x` in flattened coordinates, with shape
        `(*f(x).shape, n)` where `n = ravel_pytree(x)[0].size`.
    unflatten : callable
        Function that maps a flat vector of length `n` back to the pytree
        structure of `x`.
    """
    # Flatten params to allow pytree inputs
    x_flat, unflatten = ravel_pytree(x)
    n = x_flat.size

    # Flatten input, checkpoint, and jit
    def f_flat(z):
        return f(unflatten(z))

    f_flat = jax.checkpoint(f_flat) if checkpoint else f_flat
    f_flat = jax.jit(f_flat) if jit else f_flat

    # Straight jax jacobian if only one batch (no batching overhead)
    if nbatches == 1:
        J = jax.jacobian(f_flat)(x_flat)
        return J, unflatten

    # Get the batch indices and total size after padding
    idx, total = _get_batch_sizes(n, nbatches)

    # Use linearise to get the jvp without re-evaluating f for each column
    y0, jvp = jax.linearize(f_flat, x_flat)
    jvp = jax.jit(jvp) if jit else jvp

    # Define scan step to compute one block of columns for efficient jit
    def step(carry, idxs):
        V = jax.nn.one_hot(idxs, n, dtype=x_flat.dtype)  # (batch_size, n)
        return carry, np.moveaxis(jax.vmap(jvp)(V), 0, -1)  # (*y_shape, batch_size)

    # Calculate the jacobian blocks and reshape
    _, blocks = jax.lax.scan(step, None, idx)
    blocks = np.moveaxis(blocks, 0, -2)  # (*y_shape, nbatches, batch_size)
    J = blocks.reshape(*y0.shape, total)[..., :n]  # (*y_shape, n)

    # return outputs
    return J, unflatten


def hessian(
    f: callable,
    x: PyTree,
    nbatches: int = 1,
    jit: bool = True,
    checkpoint: bool = False,
) -> tuple[Array, callable]:
    """
    A batched version of `jax.hessian` that computes the Hessian in column blocks
    to reduce peak memory. Increase `nbatches` to reduce block size. Set
    `checkpoint=True` to trade extra computation for further memory savings.
    `f(x)` must return a scalar.

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

    Returns
    -------
    H : Array
        The Hessian of `f` at `x` in flattened coordinates, with shape `(n, n)`
        where `n = ravel_pytree(x)[0].size`.
    unflatten : callable
        Function that maps a flat vector of length `n` back to the pytree
        structure of `x`.
    """
    # Flatten params to allow pytree inputs
    x_flat, unflatten = ravel_pytree(x)
    n = x_flat.size

    # Flatten input, checkpoint, and jit
    def f_flat(z):
        return f(unflatten(z))

    f_flat = jax.checkpoint(f_flat) if checkpoint else f_flat
    f_flat = jax.jit(f_flat) if jit else f_flat

    # Straight jax hessian if only one batch (no batching overhead)
    if nbatches == 1:
        H = jax.hessian(f_flat)(x_flat)
        return H, unflatten

    # Get the batch indices and total size after padding
    idx, total = _get_batch_sizes(n, nbatches)

    # Linearise the gradient once to get a fast Hessian-vector product:
    # hvp(v) = H @ v  (forward-over-reverse)
    _, hvp = jax.linearize(jax.grad(f_flat), x_flat)
    hvp = jax.jit(hvp) if jit else hvp

    # Define scan step to compute one block of columns for efficient jit
    def step(carry, idxs):
        V = jax.nn.one_hot(idxs, n, dtype=x_flat.dtype)  # (batch_size, n)
        return carry, np.moveaxis(jax.vmap(hvp)(V), 0, -1)  # (n, batch_size)

    # Calculate Hessian blocks and reshape
    _, blocks = jax.lax.scan(step, None, idx)  # (nbatches, n, batch_size)
    H = blocks.transpose(1, 0, 2).reshape(n, total)[:, :n]

    # return outputs
    return H, unflatten


def hessian_to_pytree(H: Array, x: PyTree) -> PyTree:
    """
    Converts a flat `(n, n)` Hessian (computed w.r.t. `ravel_pytree(x)`) into a
    pytree-of-pytrees matching the structure of `x`. Assumes `H` was computed with
    the same `x` structure and leaf shapes, and that flattening was performed via
    `ravel_pytree(x)`.

    Parameters
    ----------
    H : Array
        The flat `(n, n)` Hessian matrix where `n = ravel_pytree(x)[0].size`.
    x : PyTree
        The pytree whose structure defines the block partition of `H`.

    Returns
    -------
    H_tree : PyTree
        A pytree-of-pytrees with the same structure as `x` twice over, where
        each leaf block `H_tree[i][j]` has shape `leaf_i.shape + leaf_j.shape`.
    """
    leaves, treedef = jax.tree_util.tree_flatten(x)

    # leaf sizes and shapes define the partition of the flat axis
    sizes = [int(np.size(leaf)) for leaf in leaves]
    shapes = [leaf.shape for leaf in leaves]

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
