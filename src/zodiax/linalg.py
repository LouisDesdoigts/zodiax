"""PyTree-aware vectors, matrices, Jacobians, Hessians, and Fisher information."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import prod
from operator import index as integer_index
from typing import Any

import equinox as eqx
import jax.numpy as np
import numpy as onp
from jax import Array
from jax import tree_util as jtu

from .base import Module
from .numerics.arrays import _as_float_array, _validate_float_array
from .stats import gauss_hessian

__all__ = [
    "TreeLayout",
    "TreeVector",
    "TreeMatrix",
    "Jacobian",
    "Hessian",
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
    for key in path.split("."):
        if isinstance(value, Mapping):
            value = value[key]
        elif isinstance(value, (list, tuple)):
            value = value[int(key)]
        else:
            value = getattr(value, key)
    return value


def _shape(value: Any, path: str) -> tuple[int, ...]:
    """Return the concrete shape of one floating leaf."""
    try:
        array = _as_float_array(value)
    except TypeError as error:
        raise TypeError(f"Leaf {path!r} must be floating and array-like.") from error
    return tuple(int(size) for size in array.shape)


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

    split_paths = [path.split(".") for path in paths]
    for index, left in enumerate(split_paths):
        for right in split_paths[index + 1 :]:
            shortest = min(len(left), len(right))
            if left[:shortest] == right[:shortest]:
                raise ValueError("One layout path cannot be the parent of another.")


class TreeLayout(Module):
    """Static mapping between named PyTree leaves and one flat coordinate axis.

    A layout records only the order and shapes used to concatenate floating leaves.
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

    tree = {"position": jnp.zeros(2), "flux": jnp.ones(())}
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
        *,
        alias: Any = None,
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
        self.alias = alias

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
        represented as dotted Zodiax paths. Literal dots in mapping keys therefore
        carry the same nested-path meaning as elsewhere in Zodiax.

        Parameters
        ----------
        tree : PyTree
            Tree containing at least one floating array-like leaf. Every leaf must
            have a floating dtype.

        Returns
        -------
        layout : TreeLayout
            Static description of the tree's numerical coordinates.
        """
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
            Nested object or flat path mapping containing the selected leaves.
        paths : str or sequence of str
            Leaf paths in the required flat-coordinate order.

        Returns
        -------
        layout : TreeLayout
            Static description of the selected numerical coordinates.
        """
        paths = (paths,) if isinstance(paths, str) else tuple(paths)

        shapes = []
        selected: dict[int, str] = {}
        for path in paths:
            try:
                value = _resolve_path(tree, path)
            except (AttributeError, IndexError, KeyError, ValueError) as error:
                raise ValueError(
                    f"Tree does not contain layout path {path!r}."
                ) from error
            # Object identity is meaningful for array leaves and detects selecting
            # one stored leaf through both its canonical path and an alias. Python
            # scalar interning must not make two independent scalar fields appear to
            # be the same leaf.
            if isinstance(value, (Array, onp.ndarray)):
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
            Every selected leaf must have a floating dtype.

        Returns
        -------
        vector : Array
            Concatenated coordinates with shape ``(size,)``. Leaf dtypes follow normal
            JAX promotion rules.
        """
        leaves = []
        for path, shape in zip(self.paths, self.shapes):
            try:
                value = _resolve_path(tree, path)
            except (AttributeError, IndexError, KeyError, ValueError) as error:
                raise ValueError(
                    f"Tree does not contain layout path {path!r}."
                ) from error
            array = _as_float_array(value)
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
            Flat numerical coordinates with shape ``(size,)``.
        nested : bool
            Return nested dictionaries when ``True``; otherwise return a flat
            path-keyed dictionary.

        Returns
        -------
        tree : dict
            Flat or nested dictionary of shaped leaves in the vector's dtype.
        """
        vector = _as_float_array(vector)
        if vector.ndim != 1 or vector.size != self.size:
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
            Numerical array whose selected axis has length ``size``.
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
        array = _as_float_array(array)
        if not isinstance(axis, int):
            raise TypeError("axis must be an integer.")
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

    def __zodiax_validate__(self) -> None:
        """Validate constructor-free archive reconstruction."""
        if type(self.paths) is not tuple or type(self.shapes) is not tuple:
            raise TypeError("paths and shapes must be tuples.")
        _validate_paths(self.paths)
        if len(self.paths) != len(self.shapes):
            raise ValueError("paths and shapes must have equal lengths.")
        for shape in self.shapes:
            if type(shape) is not tuple:
                raise TypeError("Every shape must be a tuple of integers.")
            if any(type(size) is not int for size in shape):
                raise TypeError("Every shape dimension must be an integer.")
            if any(size < 0 for size in shape):
                raise ValueError("Leaf shapes cannot contain negative sizes.")


class TreeVector(Module):
    """A flat vector associated with a serialisable :class:`TreeLayout`.

    Tree vectors provide flat and nested dictionary views without duplicating their
    numerical values. The vector is a dynamic JAX leaf and the layout contains only
    static coordinate topology.
    """

    vector: Array
    layout: TreeLayout

    def __init__(self, vector: Any, layout: TreeLayout, *, alias: Any = None):
        """Construct a tree-aware vector.

        Parameters
        ----------
        vector : array-like
            Flat numerical vector with shape ``(layout.size,)``.
        layout : TreeLayout
            Coordinate layout used to interpret the vector.
        """
        if not isinstance(layout, TreeLayout):
            raise TypeError("layout must be a TreeLayout.")

        vector = _as_float_array(vector)
        if vector.ndim != 1 or vector.size != layout.size:
            raise ValueError(f"vector must have shape ({layout.size},).")

        self.vector = vector
        self.layout = layout
        self.alias = alias

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

    def __zodiax_validate__(self) -> None:
        """Validate constructor-free archive reconstruction."""
        if not isinstance(self.layout, TreeLayout):
            raise TypeError("layout must be a TreeLayout.")
        _validate_float_array(self.vector, "vector")
        if self.vector.shape != (self.layout.size,):
            raise ValueError(f"vector must have shape ({self.layout.size},).")


class TreeMatrix(Module):
    """A square floating matrix indexed by one :class:`TreeLayout` on both axes.

    A block has shape ``row_leaf_shape + column_leaf_shape``. Concrete symmetric
    subclasses use the same compact ``matrix`` and ``layout`` state.
    """

    matrix: Array
    layout: TreeLayout
    _symmetric = False

    def __init__(self, matrix: Any, layout: TreeLayout, *, alias: Any = None):
        """Construct a square tree-aware matrix.

        Parameters
        ----------
        matrix : array-like
            Floating matrix with shape ``(layout.size, layout.size)``. Symmetric
            subclasses replace it by ``(matrix + matrix.T) / 2``.
        layout : TreeLayout
            Coordinate layout represented by both matrix axes.
        """
        if not isinstance(layout, TreeLayout):
            raise TypeError("layout must be a TreeLayout.")

        matrix = _as_float_array(matrix)
        shape = (layout.size, layout.size)
        if matrix.ndim != 2 or matrix.shape != shape:
            raise ValueError(f"matrix must have shape {shape}.")
        matrix = (matrix + matrix.T) / 2 if self._symmetric else matrix

        self.matrix = matrix
        self.layout = layout
        self.alias = alias

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

    def __zodiax_validate__(self) -> None:
        """Validate constructor-free archive reconstruction."""
        if not isinstance(self.layout, TreeLayout):
            raise TypeError("layout must be a TreeLayout.")
        _validate_float_array(self.matrix, "matrix")
        shape = (self.layout.size, self.layout.size)
        if self.matrix.shape != shape:
            raise ValueError(f"matrix must have shape {shape}.")
        if self._symmetric and not onp.array_equal(
            onp.asarray(self.matrix),
            onp.asarray(self.matrix.T),
            equal_nan=True,
        ):
            raise ValueError("matrix must be symmetric.")


class Jacobian(Module):
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

    parameters = {"position": jnp.array([1.0, 2.0])}

    def model(values):
        return values["position"] ** 2

    jacobian = zdx.jacobian(model, parameters)
    columns = jacobian.columns()
    ```
    """

    matrix: Array
    layout: TreeLayout

    def __init__(self, matrix: Any, layout: TreeLayout, *, alias: Any = None):
        """Construct a Jacobian from its realised derivative array.

        Parameters
        ----------
        matrix : array-like
            Floating derivative array with shape
            ``output_shape + (layout.size,)``.
        layout : TreeLayout
            Parameter coordinates represented by the final matrix axis.
        """
        if not isinstance(layout, TreeLayout):
            raise TypeError("layout must be a TreeLayout.")

        matrix = _as_float_array(matrix)
        if matrix.ndim < 1 or matrix.shape[-1] != layout.size:
            raise ValueError(f"matrix must have a final axis of size {layout.size}.")

        self.matrix = matrix
        self.layout = layout
        self.alias = alias

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

    def __zodiax_validate__(self) -> None:
        """Validate constructor-free archive reconstruction."""
        if not isinstance(self.layout, TreeLayout):
            raise TypeError("layout must be a TreeLayout.")
        _validate_float_array(self.matrix, "matrix")
        if self.matrix.ndim < 1 or self.matrix.shape[-1] != self.layout.size:
            raise ValueError(
                f"matrix must have a final axis of size {self.layout.size}."
            )


class Hessian(TreeMatrix):
    """Symmetric Hessian on one tree-structured parameter space.

    Hessians may be indefinite. Construction enforces exact symmetry by storing
    ``(matrix + matrix.T) / 2``. :func:`zodiax.hessian` consumes rather than stores
    the differentiated function and temporary JAX transformation closures.
    """

    _symmetric = True

    def __init__(self, matrix: Any, layout: TreeLayout):
        """Construct a symmetric Hessian.

        Parameters
        ----------
        matrix : array-like
            Floating matrix with shape ``(layout.size, layout.size)``. The stored
            matrix is algebraically symmetrised.
        layout : TreeLayout
            Parameter coordinates represented by both matrix axes.
        """
        super().__init__(matrix, layout)


class Fisher(TreeMatrix):
    """Symmetric Fisher information on one tree-structured parameter space.

    Construction enforces exact symmetry by storing ``(matrix + matrix.T) / 2``.
    Positive semidefiniteness is expected but is not projected or validated.
    """

    _symmetric = True

    def __init__(self, matrix: Any, layout: TreeLayout):
        """Construct symmetric Fisher information.

        Parameters
        ----------
        matrix : array-like
            Floating matrix with shape ``(layout.size, layout.size)``. The stored
            matrix is algebraically symmetrised.
        layout : TreeLayout
            Parameter coordinates represented by both matrix axes.
        """
        super().__init__(matrix, layout)

    @classmethod
    def from_jacobian(
        cls,
        jacobian: Any,
        layout: TreeLayout | None = None,
        *,
        covariance: Any | None = None,
    ) -> "Fisher":
        """Construct Gaussian Fisher information ``J.T @ C^-1 @ J``.

        Parameters
        ----------
        jacobian : Jacobian or array-like
            Tree-aware model Jacobian, or a floating array with shape
            ``output_shape + (layout.size,)``.
        layout : TreeLayout, optional
            Parameter-coordinate layout for an array-like Jacobian. It is inferred
            from a :class:`Jacobian` object and may otherwise not be omitted.
        covariance : array-like, optional
            Data covariance with shape ``(n_data, n_data)``, where ``n_data`` is the
            flattened output size. Identity weighting is used when omitted; ordering
            follows row-major flattening of the output axes.

        Returns
        -------
        fisher : Fisher
            Symmetrised Gaussian Fisher information and its parameter layout.
        """
        if isinstance(jacobian, Jacobian):
            if layout is not None:
                if not isinstance(layout, TreeLayout):
                    raise TypeError("layout must be a TreeLayout or None.")
                if not layout.compatible(jacobian.layout):
                    raise ValueError(
                        "layout must be compatible with the Jacobian parameter "
                        "layout."
                    )
            layout = jacobian.layout
            jacobian = jacobian.matrix
        elif not isinstance(layout, TreeLayout):
            raise TypeError("layout must be a TreeLayout when jacobian is array-like.")

        jacobian = _as_float_array(jacobian)
        if jacobian.ndim < 1 or jacobian.shape[-1] != layout.size:
            raise ValueError(f"jacobian must have a final axis of size {layout.size}.")
        output_size = prod(jacobian.shape[:-1])
        jacobian = jacobian.reshape((output_size, layout.size))

        if covariance is not None:
            covariance = _as_float_array(covariance)
            shape = (jacobian.shape[0], jacobian.shape[0])
            if covariance.shape != shape:
                raise ValueError(f"covariance must have shape {shape}.")

        matrix = gauss_hessian(jacobian, covariance)
        return cls(matrix, layout)
