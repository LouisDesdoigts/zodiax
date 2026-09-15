"""PyTree-aware derivative layouts and realised derivative containers.

Real numerical inputs, including integers, are converted to JAX's configured
default floating dtype at the array boundaries in this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite, prod
from numbers import Real
from operator import index as integer_index
from typing import Any

import equinox as eqx
import jax.numpy as np
from jax import Array
from jax.errors import ConcretizationTypeError
from jax import tree_util as jtu

from ..base import Base, _validate_mapping_keys
from ..numerics.arrays import as_array

# Both modules refer to the other through module attributes at call time.
from . import decompositions

__all__ = [
    "TreeLayout",
    "TreeVector",
    "TreeMatrix",
    "Derivative",
    "Jacobian",
    "Hessian",
    "GaussNewton",
    "Fisher",
]


def _path_string(key_path: tuple) -> str:
    """Convert one supported JAX key path to a Zodiax path string."""
    if len(key_path) == 0:
        return "value"

    parts = []
    for key in key_path:
        if isinstance(key, jtu.GetAttrKey):
            parts.append(key.name)
        elif isinstance(key, jtu.DictKey) and isinstance(key.key, str):
            if "." in key.key:
                raise ValueError(
                    f"Mapping key {key.key!r} contains '.'. Dots are reserved as "
                    "structural path separators."
                )
            parts.append(key.key)
        elif isinstance(key, jtu.SequenceKey):
            parts.append(str(key.idx))
        else:
            raise TypeError(
                "TreeLayout supports module attributes, string mapping keys, and "
                "sequence indices."
            )
    return ".".join(parts)


def _resolve_path(tree: Any, path: str) -> Any:
    """Resolve a stored path from a nested tree or flat path mapping."""
    if isinstance(tree, Mapping) and path in tree:
        return tree[path]
    if path == "value" and not isinstance(tree, (Mapping, list, tuple)):
        if not hasattr(tree, "value"):
            return tree

    value = tree
    try:
        for key in path.split("."):
            if isinstance(value, Mapping):
                value = value[key]
            elif isinstance(value, (list, tuple)):
                value = value[int(key)]
            else:
                value = getattr(value, key)
    except (AttributeError, IndexError, KeyError, ValueError) as error:
        raise ValueError(f"Tree does not contain layout path {path!r}.") from error
    return value


def _shape(value: Any, path: str) -> tuple[int, ...]:
    """Return the concrete shape of one real numerical leaf."""
    try:
        array = as_array(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"Leaf {path!r} must be real and array-like.") from error
    return tuple(int(size) for size in array.shape)


def _standard_deviation(value: Any, shape: tuple[int, ...]) -> Array:
    """Broadcast independent-noise standard deviations to an output shape."""
    value = as_array(value, dtype=float)
    try:
        value = np.broadcast_to(value, shape)
    except ValueError as error:
        raise ValueError(f"std must broadcast to output shape {shape}.") from error
    return value.reshape(-1)


def _eigenvalue_bounds(matrix: Array, rtol: float, atol: float) -> tuple[Array, Array]:
    """Return the smallest symmetric eigenvalue and its numerical tolerance."""
    # JAX's default (matrix + matrix.T) / 2 can overflow finite large entries.
    symmetric = matrix / 2 + matrix.T / 2
    eigenvalues = np.linalg.eigvalsh(symmetric, symmetrize_input=False)
    tolerance = np.maximum(atol, rtol * np.max(np.abs(eigenvalues)))
    return eigenvalues[0], tolerance


def _static_shape(shape: Sequence[int]) -> tuple[int, ...]:
    """Validate one explicit shape without silently coercing its dimensions."""
    try:
        dimensions = tuple(shape)
    except TypeError as error:
        raise TypeError("Every shape must be a sequence of integers.") from error

    output = []
    for size in dimensions:
        if isinstance(size, bool):
            raise TypeError("Every shape dimension must be an integer.")
        try:
            output.append(integer_index(size))
        except TypeError as error:
            raise TypeError("Every shape dimension must be an integer.") from error
    return tuple(output)


def _nested(mapping: Mapping[str, Any]) -> dict:
    """Expand a flat path mapping into nested dictionaries."""
    nested = {}
    for path, value in mapping.items():
        target = nested
        parts = path.split(".")
        for key in parts[:-1]:
            target = target.setdefault(key, {})
        target[parts[-1]] = value
    return nested


def _validate_paths(paths: tuple[str, ...]) -> None:
    """Validate that paths are unique leaves of one nested dictionary."""
    if len(paths) == 0:
        raise ValueError("paths must contain at least one leaf path.")
    if any(not isinstance(path, str) or not path for path in paths):
        raise TypeError("Every path must be a non-empty string.")
    if any("" in path.split(".") for path in paths):
        raise ValueError("Paths cannot contain empty components.")
    if len(set(paths)) != len(paths):
        raise ValueError("paths must be unique.")

    # Sort components so a parent precedes its descendants even when unrelated
    # names contain punctuation that sorts before a dot.
    ordered = sorted(path.split(".") for path in paths)
    for left, right in zip(ordered, ordered[1:]):
        if right[: len(left)] == left:
            raise ValueError("One layout path cannot be the parent of another.")


class TreeLayout(Base):
    """Static mapping between named PyTree leaves and one flat coordinate axis.

    A layout records only the order and shapes used to concatenate numerical leaves.
    Values and dtypes live on the corresponding vector or matrix, making the layout
    compact and serialisable. Plain dictionaries follow JAX's canonical key order
    when constructed with :meth:`from_tree`; :meth:`from_paths` preserves the
    supplied Zodiax path order.

    Parameters
    ----------
    paths : sequence of str
        Unique leaf paths in flat-coordinate order.
    shapes : sequence of tuple[int, ...]
        Original array shape for every path. Scalar leaves use ``()``.

    Examples
    --------
    ```python
    import jax.numpy as jnp
    import zodiax as zdx

    tree = {"weights": jnp.zeros(2), "bias": jnp.ones(())}
    layout = zdx.TreeLayout.from_tree(tree)
    flat = layout.flatten(tree)
    restored = layout.unflatten(flat, nested=True)
    ```
    """

    paths: tuple[str, ...] = eqx.field(static=True)
    shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)

    def __init__(
        self,
        paths: Sequence[str],
        shapes: Sequence[Sequence[int]],
    ):
        """Construct a layout from explicit, ordered leaf metadata.

        Parameters
        ----------
        paths : sequence of str
            Unique leaf paths. Dots delimit nested dictionary components.
        shapes : sequence of sequence of int
            Concrete leaf shapes in the same order as ``paths``.
        """
        paths = tuple(paths)
        shapes = tuple(_static_shape(shape) for shape in shapes)

        _validate_paths(paths)
        if len(paths) != len(shapes):
            raise ValueError("paths and shapes must have equal lengths.")
        if any(size < 0 for shape in shapes for size in shape):
            raise ValueError("Leaf shapes cannot contain negative sizes.")

        self.paths = paths
        self.shapes = shapes

    @property
    def sizes(self) -> tuple[int, ...]:
        """Flattened size of every leaf in coordinate order."""
        return tuple(prod(shape) for shape in self.shapes)

    @property
    def starts(self) -> tuple[int, ...]:
        """Inclusive start index of every leaf on the flat axis."""
        starts = []
        start = 0
        for size in self.sizes:
            starts.append(start)
            start += size
        return tuple(starts)

    @property
    def slices(self) -> tuple[slice, ...]:
        """Flat slice occupied by every leaf in coordinate order."""
        return tuple(
            slice(start, start + size) for start, size in zip(self.starts, self.sizes)
        )

    @property
    def size(self) -> int:
        """Total number of scalar coordinates represented by the layout."""
        return sum(self.sizes)

    @classmethod
    def from_tree(cls, tree: Any) -> "TreeLayout":
        """Construct a layout using the canonical JAX leaf order of a PyTree.

        Module attributes, string-keyed mappings, and list or tuple indices are
        represented as dotted Zodiax paths. Dots in literal mapping keys are rejected
        because dots are reserved as structural path separators.

        Parameters
        ----------
        tree : PyTree
            Parameter tree containing at least one real numerical array-like leaf.
            Every leaf is included; integer and floating inputs are accepted.

        Returns
        -------
        layout : TreeLayout
            Static description of the tree's numerical coordinates.
        """
        _validate_mapping_keys(tree)
        path_leaves, _ = jtu.tree_flatten_with_path(tree)
        if len(path_leaves) == 0:
            raise ValueError("tree must contain at least one numerical leaf.")

        paths, shapes = [], []
        for key_path, value in path_leaves:
            path = _path_string(key_path)
            paths.append(path)
            shapes.append(_shape(value, path))
        return cls(paths, shapes)

    @classmethod
    def from_paths(cls, tree: Any, paths: str | Sequence[str]) -> "TreeLayout":
        """Construct a layout from explicitly ordered Zodiax paths in a tree.

        Parameters
        ----------
        tree : PyTree
            Nested object containing the selected leaves.
        paths : str or sequence of str
            Leaf paths in the required flat-coordinate order.

        Returns
        -------
        layout : TreeLayout
            Static description of the selected numerical coordinates.
        """
        _validate_mapping_keys(tree)
        paths = (paths,) if isinstance(paths, str) else tuple(paths)

        shapes = []
        selected: dict[int, str] = {}
        for path in paths:
            value = _resolve_path(tree, path)
            # Object identity is meaningful for array leaves and detects selecting
            # one stored leaf through both its canonical path and an alias. Python
            # scalar interning must not make two independent scalar fields appear to
            # be the same leaf.
            if eqx.is_array(value):
                identity = id(value)
                if identity in selected:
                    raise ValueError(
                        f"Layout paths {selected[identity]!r} and {path!r} select "
                        "the same leaf, possibly through an alias."
                    )
                selected[identity] = path
            shapes.append(_shape(value, path))
        return cls(paths, shapes)

    def compatible(self, other: Any) -> bool:
        """Return whether another layout describes the same flat coordinates."""
        return (
            isinstance(other, TreeLayout)
            and self.paths == other.paths
            and self.shapes == other.shapes
        )

    def flatten(self, tree: Any) -> Array:
        """Concatenate matching tree leaves into one vector.

        Parameters
        ----------
        tree : PyTree or mapping[str, array-like]
            Nested tree or flat path mapping matching every stored path and shape.
            Selected real numerical leaves are converted to the configured default
            floating dtype, including integer inputs.

        Returns
        -------
        vector : Array
            Concatenated coordinates with shape ``(size,)`` in the configured
            default floating dtype.
        """
        leaves = []
        for path, shape in zip(self.paths, self.shapes):
            value = _resolve_path(tree, path)
            array = as_array(value, dtype=float)
            if array.shape != shape:
                raise ValueError(
                    f"Leaf {path!r} has shape {array.shape}, expected {shape}."
                )
            leaves.append(array.reshape(-1))
        return np.concatenate(leaves)

    def unflatten(self, vector: Any, *, nested: bool = False) -> dict:
        """Map a flat vector back to leaves with their stored shapes.

        Parameters
        ----------
        vector : array-like
            Flat real numerical coordinates with shape ``(size,)``, converted to
            the configured default floating dtype.
        nested : bool
            Return nested dictionaries when ``True``; otherwise return a flat
            path-keyed dictionary.

        Returns
        -------
        tree : dict
            Flat or nested dictionary of shaped leaves in the configured default
            floating dtype.
        """
        vector = as_array(vector, dtype=float)
        if vector.shape != (self.size,):
            raise ValueError(f"vector must have shape ({self.size},).")

        values = {}
        for path, shape, leaf_slice in zip(
            self.paths,
            self.shapes,
            self.slices,
        ):
            values[path] = vector[leaf_slice].reshape(shape)
        return _nested(values) if nested else values

    def split_axis(self, array: Any, axis: int, *, nested: bool = False) -> dict:
        """Replace one flat array axis with the stored path and leaf shapes.

        Parameters
        ----------
        array : array-like
            Real numerical array whose selected axis has length ``size``, converted
            to the configured default floating dtype.
        axis : int
            Axis to partition. Other axes retain their original ordering.
        nested : bool
            Return nested dictionaries when ``True``; otherwise return a flat
            path-keyed dictionary.

        Returns
        -------
        tree : dict
            Array leaves in which the selected axis is replaced by each stored leaf
            shape.
        """
        array = as_array(array, dtype=float)
        try:
            axis = integer_index(axis)
        except TypeError as error:
            raise TypeError("axis must be an integer.") from error
        axis = axis + array.ndim if axis < 0 else axis
        if not 0 <= axis < array.ndim:
            raise ValueError(f"axis must index an array with {array.ndim} dimensions.")
        if array.shape[axis] != self.size:
            raise ValueError(
                f"array axis {axis} has size {array.shape[axis]}, expected {self.size}."
            )

        values = {}
        for path, shape, leaf_slice in zip(self.paths, self.shapes, self.slices):
            index = [slice(None)] * array.ndim
            index[axis] = leaf_slice
            output_shape = array.shape[:axis] + shape + array.shape[axis + 1 :]
            values[path] = array[tuple(index)].reshape(output_shape)
        return _nested(values) if nested else values


class TreeVector(Base):
    """A flat vector associated with a serialisable :class:`TreeLayout`.

    Tree vectors provide flat and nested dictionary views without duplicating their
    numerical values. The vector is a dynamic JAX leaf and the layout contains only
    static coordinate topology.
    """

    vector: Array
    layout: TreeLayout

    def __init__(self, vector: Any, layout: TreeLayout):
        """Construct a tree-aware vector.

        Parameters
        ----------
        vector : array-like
            Flat real numerical vector with shape ``(layout.size,)``, converted to
            the configured default floating dtype.
        layout : TreeLayout
            Coordinate layout used to interpret the vector.
        """
        if not isinstance(layout, TreeLayout):
            raise TypeError("layout must be a TreeLayout.")

        vector = as_array(vector, dtype=float)
        if vector.shape != (layout.size,):
            raise ValueError(f"vector must have shape ({layout.size},).")

        self.vector = vector
        self.layout = layout

    @classmethod
    def from_tree(cls, tree: Any, layout: TreeLayout | None = None) -> "TreeVector":
        """Flatten a tree into a vector, deriving its layout when omitted."""
        layout = TreeLayout.from_tree(tree) if layout is None else layout
        return cls(layout.flatten(tree), layout)

    def flat_dict(self) -> dict:
        """Return path-keyed leaves with their original shapes."""
        return self.layout.unflatten(self.vector)

    def nested_dict(self) -> dict:
        """Return the vector as nested dictionaries following its stored paths."""
        return self.layout.unflatten(self.vector, nested=True)


class TreeMatrix(Base):
    """A square floating matrix indexed by one :class:`TreeLayout` on both axes.

    A block has shape ``row_leaf_shape + column_leaf_shape``. Concrete symmetric
    subclasses use the same compact ``matrix`` and ``layout`` state.
    """

    matrix: Array
    layout: TreeLayout
    _symmetric = False

    def __init__(self, matrix: Any, layout: TreeLayout):
        """Construct a square tree-aware matrix.

        Parameters
        ----------
        matrix : array-like
            Real numerical matrix with shape ``(layout.size, layout.size)``,
            converted to the configured default floating dtype. Symmetric subclasses
            replace it by ``(matrix + matrix.T) / 2``.
        layout : TreeLayout
            Coordinate layout represented by both matrix axes.
        """
        if not isinstance(layout, TreeLayout):
            raise TypeError("layout must be a TreeLayout.")

        matrix = as_array(matrix, dtype=float)
        shape = (layout.size, layout.size)
        if matrix.shape != shape:
            raise ValueError(f"matrix must have shape {shape}.")
        if self._symmetric:
            # Halve before adding so finite large entries do not overflow.
            matrix = matrix / 2 + matrix.T / 2

        self.matrix = matrix
        self.layout = layout

    @classmethod
    def from_tree(cls, matrix: Any, tree: Any) -> "TreeMatrix":
        """Attach a matrix to the coordinates of a parameter PyTree.

        This is the high-level constructor for realised matrices calculated
        outside Zodiax. The coordinate layout is derived internally from ``tree``.
        """
        return cls(matrix, TreeLayout.from_tree(tree))

    def rows(self, *, nested: bool = False) -> dict:
        """Split the first matrix axis into layout leaves."""
        return self.layout.split_axis(self.matrix, 0, nested=nested)

    def columns(self, *, nested: bool = False) -> dict:
        """Split the second matrix axis into layout leaves."""
        return self.layout.split_axis(self.matrix, 1, nested=nested)

    def blocks(self, *, nested: bool = False) -> dict:
        """Return path-paired blocks with row shape followed by column shape."""
        blocks = {}
        for row_path, row_shape, row_slice in zip(
            self.layout.paths,
            self.layout.shapes,
            self.layout.slices,
        ):
            row = {}
            for column_path, column_shape, column_slice in zip(
                self.layout.paths,
                self.layout.shapes,
                self.layout.slices,
            ):
                shape = row_shape + column_shape
                row[column_path] = self.matrix[row_slice, column_slice].reshape(shape)
            blocks[row_path] = _nested(row) if nested else row
        return _nested(blocks) if nested else blocks

    def diagonal(self) -> TreeVector:
        """Return the matrix diagonal mapped to layout leaves."""
        return TreeVector(np.diag(self.matrix), self.layout)

    def row_norms(self, ord: int | float | None = None) -> TreeVector:
        """Return norms across columns, mapped to layout leaves."""
        return TreeVector(np.linalg.norm(self.matrix, ord=ord, axis=1), self.layout)

    def column_norms(self, ord: int | float | None = None) -> TreeVector:
        """Return norms across rows, mapped to layout leaves."""
        return TreeVector(np.linalg.norm(self.matrix, ord=ord, axis=0), self.layout)

    def is_finite(self) -> Array:
        """Return whether every matrix entry is finite.

        The scalar Boolean result remains a JAX array and may be used inside a
        compiled diagnostic calculation.
        """
        return np.all(np.isfinite(self.matrix))

    def is_symmetric(
        self,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> Array:
        """Return whether the matrix is numerically symmetric."""
        return np.allclose(
            self.matrix,
            self.matrix.T,
            rtol=rtol,
            atol=atol,
        )

    def is_positive_semidefinite(
        self,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> Array:
        """Return whether the matrix is finite, symmetric, and numerically PSD."""
        if self.matrix.size == 0:
            return np.asarray(False)
        minimum, tolerance = _eigenvalue_bounds(self.matrix, rtol, atol)
        return (
            self.is_finite()
            & self.is_symmetric(rtol=rtol, atol=atol)
            & (minimum >= -tolerance)
        )

    def is_positive_definite(
        self,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> Array:
        """Return whether the matrix is finite, symmetric, and numerically PD."""
        if self.matrix.size == 0:
            return np.asarray(False)
        minimum, tolerance = _eigenvalue_bounds(self.matrix, rtol, atol)
        return (
            self.is_finite()
            & self.is_symmetric(rtol=rtol, atol=atol)
            & (minimum > tolerance)
        )

    def check(
        self,
        method: str = "eigh",
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> "TreeMatrix":
        """Eagerly validate suitability for a decomposition or projection.

        Parameters
        ----------
        method : {"eigh", "cholesky", "projection"}
            Validation contract. Eigendecomposition requires a finite symmetric
            matrix, Cholesky additionally requires numerical positive definiteness,
            and projection requires numerical positive semidefiniteness.
        rtol, atol : float
            Relative and absolute tolerances used for symmetry and definiteness.

        Returns
        -------
        matrix : TreeMatrix
            ``self``. Symmetric derivative subclasses can chain this validation
            with their decomposition methods.

        Notes
        -----
        This method transfers values to the host and is deliberately incompatible
        with JIT. Decomposition methods do not call it automatically.
        """
        if method not in {"eigh", "cholesky", "projection"}:
            raise ValueError("method must be 'eigh', 'cholesky', or 'projection'.")
        if isinstance(rtol, bool) or not isinstance(rtol, Real):
            raise TypeError("rtol must be a real scalar.")
        if isinstance(atol, bool) or not isinstance(atol, Real):
            raise TypeError("atol must be a real scalar.")
        rtol, atol = float(rtol), float(atol)
        if not isfinite(rtol) or rtol < 0:
            raise ValueError("rtol must be finite and non-negative.")
        if not isfinite(atol) or atol < 0:
            raise ValueError("atol must be finite and non-negative.")

        if self.layout.size == 0:
            raise ValueError("matrix must contain at least one coordinate.")

        # Only these explicit diagnostics transfer scalar results to Python.
        # The numerical kernels are shared with the JIT-compatible predicates.
        try:
            if not bool(self.is_finite()):
                raise ValueError("matrix must contain only finite values.")
            if not bool(self.is_symmetric(rtol=rtol, atol=atol)):
                raise ValueError(
                    "matrix must be symmetric within the requested tolerance."
                )
            if method == "eigh":
                return self

            minimum, tolerance = _eigenvalue_bounds(self.matrix, rtol, atol)
            if method == "cholesky" and bool(minimum <= tolerance):
                raise ValueError("matrix must be positive definite for Cholesky.")
            if method == "projection" and bool(minimum < -tolerance):
                raise ValueError("matrix must be positive semidefinite for projection.")
        except ConcretizationTypeError as error:
            raise TypeError(
                "check() is eager-only and cannot be used inside JIT."
            ) from error
        return self


class Derivative(Base):
    """Base for realised Jacobian, Hessian, and Gauss--Newton results.

    ``matrix`` stores the numerical result and ``layout`` tracks its parameter
    coordinates. The final matrix axis follows that layout; leading axes describe
    the function output or, for second derivatives, another parameter axis.
    """

    matrix: Array
    layout: TreeLayout

    def __init__(self, matrix: Any, layout: TreeLayout):
        """Store a real derivative array with a final parameter axis."""
        if not isinstance(layout, TreeLayout):
            raise TypeError("layout must be a TreeLayout.")

        matrix = as_array(matrix, dtype=float)
        if matrix.ndim < 1 or matrix.shape[-1] != layout.size:
            raise ValueError(f"matrix must have a final axis of size {layout.size}.")

        self.matrix = matrix
        self.layout = layout


class Jacobian(Derivative):
    """Jacobian array associated with one tree-structured parameter space.

    The matrix has shape ``output_shape + (layout.size,)``. Its leading axes preserve
    the differentiated function's array output shape, while the final axis contains
    flattened parameter coordinates. The function and temporary JAX transformation
    closures are never stored.

    Examples
    --------
    ```python
    import jax.numpy as jnp
    import zodiax as zdx

    parameters = {"weights": jnp.array([1.0, 2.0])}

    def model(values):
        return values["weights"] ** 2

    jacobian = zdx.jacobian(model, parameters)
    fisher = jacobian.fisher(std=0.1)
    columns = jacobian.columns()
    ```
    """

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Shape of the differentiated function's array output."""
        return self.matrix.shape[:-1]

    def columns(self, *, nested: bool = False) -> dict:
        """Split the final parameter axis into layout leaves."""
        return self.layout.split_axis(self.matrix, -1, nested=nested)

    def row_norms(self, ord: int | float | None = None) -> Array:
        """Return parameter-axis norms with shape ``output_shape``."""
        return np.linalg.norm(self.matrix, ord=ord, axis=-1)

    def column_norms(self, ord: int | float | None = None) -> TreeVector:
        """Return output-space norms mapped to parameter leaves."""
        matrix = self.matrix.reshape((prod(self.output_shape), self.layout.size))
        return TreeVector(np.linalg.norm(matrix, ord=ord, axis=0), self.layout)

    def fisher(
        self,
        *,
        std: Any | None = None,
        covariance: Any | None = None,
    ) -> "Fisher":
        """Construct Gaussian Fisher information from this model Jacobian.

        Parameters
        ----------
        std : array-like, optional
            Positive standard deviation of independent Gaussian noise. A scalar or
            an array broadcastable to the Jacobian output shape is accepted and
            uses the elementwise weighting fast path. Inputs are converted to the
            configured default floating dtype.
        covariance : array-like, optional
            Full data covariance in flattened output coordinates. Mutually
            exclusive with ``std``. Inputs are converted to the configured default
            floating dtype. Omitting both selects identity weighting.

        Returns
        -------
        fisher : Fisher
            Fisher information carrying the same parameter-coordinate metadata.
        """
        if std is not None and covariance is not None:
            raise ValueError("Pass either std or covariance, not both.")

        output_size = prod(self.output_shape)
        jacobian = self.matrix.reshape((output_size, self.layout.size))
        if std is not None:
            std = _standard_deviation(std, self.output_shape)
            weighted = jacobian / std[:, None]
            matrix = weighted.T @ weighted
        elif covariance is not None:
            covariance = as_array(covariance, dtype=float)
            shape = (output_size, output_size)
            if covariance.shape != shape:
                raise ValueError(f"covariance must have shape {shape}.")
            matrix = jacobian.T @ np.linalg.solve(covariance, jacobian)
        else:
            matrix = jacobian.T @ jacobian
        return Fisher(matrix, self.layout)


class _SymmetricTreeMatrix(TreeMatrix):
    """Shared decompositions for symmetric matrices on one parameter layout."""

    _symmetric = True

    def eigh(self) -> "decompositions.EigenDecomposition":
        """Return the symmetric eigendecomposition of this matrix.

        Eigenvalues follow :func:`jax.numpy.linalg.eigh` and are stored in
        ascending order. The corresponding eigenvectors are the columns of the
        returned ``vectors`` matrix.
        """
        return decompositions.EigenDecomposition.from_matrix(self)

    def cholesky(self) -> "decompositions.CholeskyDecomposition":
        """Return a lower-triangular Cholesky decomposition.

        The matrix must be positive definite. Positive-semidefinite Fisher and
        Gauss--Newton matrices should normally use :meth:`projection` with
        ``method="eigh"`` so unresolved modes can be discarded safely.
        """
        return decompositions.CholeskyDecomposition.from_matrix(self)

    def projection(
        self,
        *,
        method: str = "eigh",
        rtol: float = 1e-5,
        damping: float = 0.0,
        equilibrate: bool = True,
    ) -> "decompositions.ParameterProjection":
        """Construct a latent-to-parameter local metric reparameterisation.

        For a full-rank, undamped matrix ``M``, the projection ``P`` satisfies
        ``P.T @ M @ P = I`` and ``P @ P.T = inv(M)`` up to floating-point
        accuracy. ``method="eigh"`` supports positive-semidefinite matrices by
        setting modes below ``rtol`` to zero. ``method="cholesky"`` requires a
        positive-definite matrix and ignores ``rtol``.

        Hessians may be indefinite, so applying this method to a :class:`Hessian`
        is only valid where the realised Hessian is locally positive semidefinite.
        """
        return decompositions.ParameterProjection.from_matrix(
            self,
            method=method,
            rtol=rtol,
            damping=damping,
            equilibrate=equilibrate,
        )

    def parameterise(
        self,
        origin: Any,
        *,
        method: str = "eigh",
        rtol: float = 1e-5,
        damping: float = 0.0,
        equilibrate: bool = True,
    ) -> "decompositions.LocalParameterisation":
        """Bind local metric coordinates directly to a parameter PyTree.

        The returned callable maps a latent coordinate vector to a PyTree centred
        on ``origin``. Parameter flattening and reconstruction remain internal.
        """
        return self.projection(
            method=method,
            rtol=rtol,
            damping=damping,
            equilibrate=equilibrate,
        ).bind(origin)


# These results are both symmetric parameter matrices and derivatives. The square
# matrix branch supplies their constructor; Derivative groups the calculation results.
class Hessian(_SymmetricTreeMatrix, Derivative):
    """Symmetric Hessian on one tree-structured parameter space.

    Hessians may be indefinite. Construction enforces exact symmetry by storing
    ``(matrix + matrix.T) / 2``. :func:`zodiax.hessian` consumes rather than stores
    the differentiated function and temporary JAX transformation closures.
    """


class GaussNewton(_SymmetricTreeMatrix, Derivative):
    """Gauss--Newton Hessian approximation on one parameter space.

    A Gauss--Newton matrix has the form ``J.T @ P @ J``, where ``J`` is the
    Jacobian of an unreduced residual function and ``P`` is its inverse
    covariance. It is not generally the exact Hessian of the corresponding scalar
    objective. Construction enforces exact symmetry by storing
    ``(matrix + matrix.T) / 2``. Positive semidefiniteness is expected when ``P``
    is positive semidefinite, but is not projected or validated.
    """


class Fisher(_SymmetricTreeMatrix):
    """Symmetric Fisher information on one tree-structured parameter space.

    Construction enforces exact symmetry by storing ``(matrix + matrix.T) / 2``.
    Positive semidefiniteness is expected but is not projected or validated.
    """
