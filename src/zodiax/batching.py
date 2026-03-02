import jax
import equinox as eqx
import jax.numpy as np
from jax import Array
from jax.flatten_util import ravel_pytree
from typing import Union

PyTree = Union[dict, list, tuple, eqx.Module]

__all__ = [
    "get_batch_sizes",
    "jacobian",
    "hessian",
    "hessian_to_pytree",
]


def get_batch_sizes(n: int, nbatches: int) -> tuple[Array, int]:
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
    A batched version of jax.jacobian designed to save memory by computing the Jacobian
    in column blocks. To lower memory usage, increase the number of batches (nbatches),
    which reduces the block size. If memory is still an issue, set checkpoint=True to
    checkpoint the function and save memory at the cost of extra computation.

    Return the Jacobian J and the unflatten function to map flat vectors back to x's
    structure.
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
    idx, total = get_batch_sizes(n, nbatches)

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
    A batched version of jax.hessian designed to save memory by computing the Hessian
    in column blocks. Increase nbatches to reduce block size. If memory is still an
    issue, set checkpoint=True to checkpoint f and save memory at the cost of extra
    computation.

    f(x) must return a scalar.

    Returns the Hessian H (n, n) in flattened coordinates and the unflatten function.
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
    idx, total = get_batch_sizes(n, nbatches)

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
    Convert a flat (n, n) Hessian (w.r.t. ravel_pytree(x)) into a pytree-of-pytrees.

    This assumes:
      - H was computed with the same x structure and leaf shapes
      - flattening was via ravel_pytree(x) (i.e. JAX pytree leaf order)

    Returns:
      H_tree: pytree where H_tree has x's structure twice, and each block has shape
              leaf_i.shape + leaf_j.shape
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
