"""Reusable decompositions of tree-aware symmetric derivative matrices.

Real numerical inputs, including integers, are converted to JAX's configured
default floating dtype at the array boundaries in this module.
"""

from __future__ import annotations

from collections.abc import Callable
from numbers import Real
from typing import Any

import equinox as eqx
import jax.numpy as np
import jax.scipy.linalg as jspl
from jax import Array
from jax import tree_util as jtu
from jax.flatten_util import ravel_pytree
from jax.typing import ArrayLike

from ..base import Base, PyTree
from ..numerics.arrays import as_array
from . import containers

# Containers expose these decompositions as methods. Import the module rather than
# individual classes so both modules can finish loading before methods use them.

__all__ = [
    "Decomposition",
    "EigenDecomposition",
    "CholeskyDecomposition",
    "ParameterProjection",
    "LocalParameterisation",
]


def _nonnegative(value: Any, name: str) -> float:
    """Return one finite, non-negative Python scalar."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar.")
    value = float(value)
    if not 0 <= value < float("inf"):
        raise ValueError(f"{name} must be finite and non-negative.")
    return value


def _symmetric_matrix(value: Any) -> containers.TreeMatrix:
    """Require one symmetric tree-aware matrix container."""
    if not isinstance(value, containers.TreeMatrix) or not value._symmetric:
        raise TypeError("matrix must be a symmetric TreeMatrix.")
    if value.layout.size == 0:
        raise ValueError("A decomposition needs at least one parameter coordinate.")
    return value


def _vector(
    value: ArrayLike | containers.TreeVector,
    layout: containers.TreeLayout,
    name: str,
) -> Array:
    """Return a compatible vector in the configured default floating dtype."""
    if isinstance(value, containers.TreeVector):
        if not layout.compatible(value.layout):
            raise ValueError(f"{name} layout must be compatible with this layout.")
        value = value.vector
    value = as_array(value, dtype=float)
    if value.shape != (layout.size,):
        raise ValueError(f"{name} must have shape ({layout.size},).")
    return value


def _tree_vector(
    value: PyTree | ArrayLike, layout: containers.TreeLayout, name: str
) -> tuple[Array, Callable[[Array], PyTree | Array]]:
    """Ravel a compatible PyTree in the configured floating dtype."""
    candidate = containers.TreeLayout.from_tree(value)
    if not layout.compatible(candidate):
        raise ValueError(f"{name} must match the projection parameter structure.")

    # Normalise before ravel so reconstruction uses the final floating dtypes.
    normalised = jtu.tree_map(lambda leaf: as_array(leaf, dtype=float), value)
    return ravel_pytree(normalised)


class Decomposition(Base):
    """Base for matrix factors and the local coordinate maps built from them.

    Concrete subclasses declare their own stored fields and numerical operations.
    They inherit Base's immutable path access and updates.
    """


class EigenDecomposition(Decomposition):
    """Symmetric eigendecomposition associated with a parameter layout.

    ``values`` are ascending and ``vectors[:, i]`` is the eigenvector associated
    with ``values[i]``, matching :func:`jax.numpy.linalg.eigh`.
    Real numerical inputs are stored in the configured default floating dtype.
    """

    values: Array
    vectors: Array
    layout: containers.TreeLayout

    def __init__(
        self, values: ArrayLike, vectors: ArrayLike, layout: containers.TreeLayout
    ):
        if not isinstance(layout, containers.TreeLayout):
            raise TypeError("layout must be a TreeLayout.")
        values = as_array(values, dtype=float)
        vectors = as_array(vectors, dtype=float)
        if values.shape != (layout.size,):
            raise ValueError(f"values must have shape ({layout.size},).")
        shape = (layout.size, layout.size)
        if vectors.shape != shape:
            raise ValueError(f"vectors must have shape {shape}.")

        self.values = values
        self.vectors = vectors
        self.layout = layout

    @classmethod
    def from_matrix(cls, matrix: containers.TreeMatrix) -> "EigenDecomposition":
        """Decompose one symmetric tree-aware matrix."""
        matrix = _symmetric_matrix(matrix)
        # The container already owns symmetry. Averaging it again inside JAX
        # can overflow large finite entries by forming A + A.T first.
        values, vectors = np.linalg.eigh(matrix.matrix, symmetrize_input=False)
        return cls(values, vectors, matrix.layout)

    def reconstruct(self) -> containers.TreeMatrix:
        """Reconstruct the represented symmetric matrix."""
        matrix = (self.vectors * self.values[None, :]) @ self.vectors.T
        return containers.TreeMatrix(matrix, self.layout)


class CholeskyDecomposition(Decomposition):
    """Lower-triangular Cholesky factor associated with a parameter layout.

    Real numerical factor inputs are stored in the configured default floating dtype.
    """

    factor: Array
    layout: containers.TreeLayout

    def __init__(self, factor: ArrayLike, layout: containers.TreeLayout):
        if not isinstance(layout, containers.TreeLayout):
            raise TypeError("layout must be a TreeLayout.")
        factor = as_array(factor, dtype=float)
        shape = (layout.size, layout.size)
        if factor.shape != shape:
            raise ValueError(f"factor must have shape {shape}.")

        self.factor = factor
        self.layout = layout

    @classmethod
    def from_matrix(cls, matrix: containers.TreeMatrix) -> "CholeskyDecomposition":
        """Factor one symmetric positive-definite tree-aware matrix."""
        matrix = _symmetric_matrix(matrix)
        factor = np.linalg.cholesky(matrix.matrix, symmetrize_input=False)
        return cls(factor, matrix.layout)

    @property
    def log_determinant(self) -> Array:
        """Log determinant of the represented positive-definite matrix."""
        return 2 * np.log(np.diag(self.factor)).sum()

    def reconstruct(self) -> containers.TreeMatrix:
        """Reconstruct the represented positive-definite matrix."""
        return containers.TreeMatrix(self.factor @ self.factor.T, self.layout)

    def solve(self, value: ArrayLike | containers.TreeVector) -> containers.TreeVector:
        """Solve the represented linear system for one compatible vector."""
        value = _vector(value, self.layout, "value")
        intermediate = jspl.solve_triangular(self.factor, value, lower=True)
        solution = jspl.solve_triangular(
            self.factor.T,
            intermediate,
            lower=False,
        )
        return containers.TreeVector(solution, self.layout)


class ParameterProjection(Decomposition):
    """Latent-to-parameter factor for a local positive-semidefinite metric.

    The square, fixed-shape ``matrix`` maps a latent vector ``u`` to a parameter
    step ``matrix @ u``. Eigendecomposition-based projections retain zero columns
    for discarded modes so their shape remains stable under JIT. Rank selection is
    discrete; differentiation through an eigenvalue crossing is not supported.
    Repeated eigenvalues can also make derivatives of the eigenvectors undefined.
    Applying an already constructed projection remains a linear operation.
    When constructing a factor directly, retained columns must be independent and
    discarded columns must be zero; solving trusts this declared rank.
    Real numerical matrix and eigenvalue inputs are stored in the configured default
    floating dtype; the retained-mode mask remains boolean.
    """

    matrix: Array
    retained: Array
    eigenvalues: Array | None
    layout: containers.TreeLayout
    method: str = eqx.field(static=True)

    def __init__(
        self,
        matrix: ArrayLike,
        retained: ArrayLike,
        layout: containers.TreeLayout,
        *,
        method: str,
        eigenvalues: ArrayLike | None = None,
    ):
        if not isinstance(layout, containers.TreeLayout):
            raise TypeError("layout must be a TreeLayout.")
        if method not in {"eigh", "cholesky"}:
            raise ValueError("method must be 'eigh' or 'cholesky'.")

        matrix = as_array(matrix, dtype=float)
        shape = (layout.size, layout.size)
        if matrix.shape != shape:
            raise ValueError(f"matrix must have shape {shape}.")

        retained = as_array(retained, dtype=bool)
        if retained.shape != (layout.size,):
            raise ValueError(f"retained must have shape ({layout.size},).")

        if eigenvalues is not None:
            eigenvalues = as_array(eigenvalues, dtype=float)
            if eigenvalues.shape != (layout.size,):
                raise ValueError(f"eigenvalues must have shape ({layout.size},).")
        if method == "eigh" and eigenvalues is None:
            raise ValueError("eigenvalues are required for method='eigh'.")
        if method == "cholesky" and eigenvalues is not None:
            raise ValueError("eigenvalues must be None for method='cholesky'.")

        self.matrix = matrix
        self.retained = retained
        self.eigenvalues = eigenvalues
        self.layout = layout
        self.method = method

    @classmethod
    def from_matrix(
        cls,
        matrix: containers.TreeMatrix,
        *,
        method: str = "eigh",
        rtol: float = 1e-5,
        damping: float = 0.0,
        equilibrate: bool = True,
    ) -> "ParameterProjection":
        """Construct a local inverse-square-root factor of ``matrix``.

        Parameters
        ----------
        matrix : symmetric TreeMatrix
            Positive-semidefinite metric in parameter coordinates.
        method : {"eigh", "cholesky"}
            ``"eigh"`` thresholds unresolved modes. ``"cholesky"`` requires a
            full-rank positive-definite metric.
        rtol : float
            Relative eigenvalue threshold. Ignored by the Cholesky method.
        damping : float
            Non-negative damping in the decomposed coordinates. For ``"eigh"`` it
            is relative to the largest eigenvalue; for ``"cholesky"`` it is added
            directly to the diagonal. With the default equilibration these
            coordinates are dimensionless.
        equilibrate : bool
            Decompose a unit-diagonal metric so rank decisions do not depend on
            parameter units.
        """
        matrix = _symmetric_matrix(matrix)
        if method not in {"eigh", "cholesky"}:
            raise ValueError("method must be 'eigh' or 'cholesky'.")
        rtol = _nonnegative(rtol, "rtol")
        damping = _nonnegative(damping, "damping")
        if type(equilibrate) is not bool:
            raise TypeError("equilibrate must be a bool.")

        metric = matrix.matrix
        size = matrix.layout.size

        # Express each parameter in units set by its diagonal curvature. Zero
        # diagonal entries use unit scale; their modes are handled below.
        if equilibrate:
            diagonal = np.diag(metric)
            positive_diagonal = np.where(diagonal > 0, diagonal, 1)
            scale = 1 / np.sqrt(positive_diagonal)
        else:
            scale = np.ones((size,), dtype=metric.dtype)
        equilibrated = scale[:, None] * metric * scale[None, :]
        equilibrated = 0.5 * equilibrated + 0.5 * equilibrated.T

        if method == "cholesky":
            # If A = L L.T in these coordinates, L**(-T) maps unit latent
            # coordinates to steps with identity curvature.
            regularised = equilibrated + damping * np.eye(
                size,
                dtype=metric.dtype,
            )
            factor = np.linalg.cholesky(regularised, symmetrize_input=False)
            inverse_factor = jspl.solve_triangular(
                factor.T,
                np.eye(size, dtype=metric.dtype),
                lower=False,
            )
            projection = scale[:, None] * inverse_factor
            retained = np.ones((size,), dtype=bool)
            return cls(
                projection,
                retained,
                matrix.layout,
                method=method,
            )

        # Decide the rank once, in equilibrated coordinates. Keep zero columns
        # for discarded modes so the PyTree and array shapes stay fixed under JIT.
        eigenvalues, eigenvectors = np.linalg.eigh(equilibrated, symmetrize_input=False)
        tiny = np.finfo(metric.dtype).tiny
        largest = np.maximum(eigenvalues[-1], tiny)
        retained = eigenvalues > rtol * largest
        regularised = np.where(
            retained,
            eigenvalues + damping * largest,
            1,
        )
        inverse_scale = np.where(retained, 1 / np.sqrt(regularised), 0)
        projection = scale[:, None] * eigenvectors * inverse_scale[None, :]
        return cls(
            projection,
            retained,
            matrix.layout,
            method=method,
            eigenvalues=eigenvalues,
        )

    @property
    def rank(self) -> Array:
        """Number of active latent modes."""
        return self.retained.sum(dtype=int)

    @property
    def covariance(self) -> containers.TreeMatrix:
        """Return ``P @ P.T``, the inverse metric on the retained subspace.

        Discarded modes contribute zero to this operator. This does not imply
        that unconstrained physical directions have zero statistical uncertainty.
        """
        return containers.TreeMatrix(self.matrix @ self.matrix.T, self.layout)

    def apply(self, value: ArrayLike) -> containers.TreeVector:
        """Map one flat latent vector to a tree-aware parameter step."""
        value = _vector(value, self.layout, "value")
        return containers.TreeVector(self.matrix @ value, self.layout)

    def solve(self, value: ArrayLike | containers.TreeVector) -> Array:
        """Return the minimum-norm latent vector for a parameter step.

        Minimise ``||P @ u - value||`` using the rank already recorded in this
        projection. Discarded latent entries are zero, including for a rank-zero
        projection. No second tolerance discards modes when parameter units differ.
        The result is an array of latent coordinates, not a parameter TreeVector.

        Differentiation with respect to ``value`` is linear. Differentiating the
        SVD used here with respect to ``matrix`` has the usual repeated-singular-value
        limitations.
        """
        value = _vector(value, self.layout, "value")
        left, singular_values, right_transpose = np.linalg.svd(
            self.matrix, full_matrices=False
        )

        # SVD orders singular values from largest to smallest. Trust the rank
        # chosen when constructing the projection, rather than applying rcond again.
        active = np.arange(singular_values.size) < self.rank
        denominators = np.where(active, singular_values, 1)
        coordinates = (left.T @ value) / denominators
        latent = right_transpose.T @ np.where(active, coordinates, 0)
        return np.where(self.retained, latent, 0)

    def pullback(self, gradient: ArrayLike | containers.TreeVector) -> Array:
        """Map a parameter-space gradient covector into latent coordinates."""
        gradient = _vector(gradient, self.layout, "gradient")
        return self.matrix.T @ gradient

    def natural_gradient(
        self, gradient: ArrayLike | containers.TreeVector
    ) -> containers.TreeVector:
        """Apply the represented inverse metric to a parameter gradient."""
        latent_gradient = self.pullback(gradient)
        return containers.TreeVector(self.matrix @ latent_gradient, self.layout)

    def natural_norm(self, gradient: ArrayLike | containers.TreeVector) -> Array:
        """Return ``sqrt(g.T @ covariance @ g)`` for a parameter gradient."""
        latent_gradient = self.pullback(gradient)
        return np.sqrt(np.maximum(np.vdot(latent_gradient, latent_gradient), 0))

    def bind(self, origin: PyTree | ArrayLike) -> "LocalParameterisation":
        """Bind an origin PyTree for direct latent-to-parameter conversion."""
        return LocalParameterisation(self, origin)

    def __call__(self, value: ArrayLike) -> containers.TreeVector:
        """Alias for :meth:`apply`."""
        return self.apply(value)


class LocalParameterisation(Decomposition):
    """A projection bound to an origin parameter PyTree.

    Calling the object maps a flat latent vector to a parameter PyTree with the
    exact structure and static metadata of ``origin``. Output numerical leaves use
    the configured default floating dtype. The underlying ravel and reconstruction
    functions are created ephemerally and are never stored.
    """

    projection: ParameterProjection
    origin: PyTree | ArrayLike

    def __init__(self, projection: ParameterProjection, origin: PyTree | ArrayLike):
        if not isinstance(projection, ParameterProjection):
            raise TypeError("projection must be a ParameterProjection.")
        _tree_vector(origin, projection.layout, "origin")
        self.projection = projection
        self.origin = origin

    @property
    def zeros(self) -> Array:
        """Zero latent vector representing the stored origin."""
        return np.zeros((self.projection.layout.size,), dtype=float)

    @property
    def variance(self) -> PyTree | Array:
        """Diagonal of the retained covariance, with the structure of ``origin``."""
        _, unravel = _tree_vector(self.origin, self.projection.layout, "origin")
        diagonal = np.sum(self.projection.matrix**2, axis=1)
        return unravel(diagonal)

    @property
    def standard_deviation(self) -> PyTree | Array:
        """Square root of retained variances, structured like ``origin``."""
        _, unravel = _tree_vector(self.origin, self.projection.layout, "origin")
        diagonal = np.sum(self.projection.matrix**2, axis=1)
        return unravel(np.sqrt(np.maximum(diagonal, 0)))

    def step(self, latent: ArrayLike) -> PyTree | Array:
        """Return a parameter-step PyTree for one latent vector."""
        _, unravel = _tree_vector(
            self.origin,
            self.projection.layout,
            "origin",
        )
        latent = _vector(latent, self.projection.layout, "latent")
        return unravel(self.projection.matrix @ latent)

    def __call__(self, latent: ArrayLike) -> PyTree | Array:
        """Map a latent vector to a parameter PyTree around ``origin``."""
        origin, unravel = _tree_vector(
            self.origin,
            self.projection.layout,
            "origin",
        )
        latent = _vector(latent, self.projection.layout, "latent")
        return unravel(origin + self.projection.matrix @ latent)

    def encode(self, parameters: PyTree | ArrayLike) -> Array:
        """Return the minimum-norm latent coordinates of ``parameters``."""
        origin, _ = _tree_vector(self.origin, self.projection.layout, "origin")
        parameters, _ = _tree_vector(
            parameters,
            self.projection.layout,
            "parameters",
        )
        return self.projection.solve(parameters - origin)

    def pullback(self, gradient: PyTree | ArrayLike) -> Array:
        """Map a gradient PyTree into latent-coordinate gradients."""
        gradient, _ = _tree_vector(
            gradient,
            self.projection.layout,
            "gradient",
        )
        return self.projection.pullback(gradient)

    def natural_gradient(self, gradient: PyTree | ArrayLike) -> PyTree | Array:
        """Apply the inverse metric and return a parameter-structured PyTree."""
        gradient, unravel = _tree_vector(
            gradient,
            self.projection.layout,
            "gradient",
        )
        natural = self.projection.natural_gradient(gradient)
        return unravel(natural.vector)

    def natural_norm(self, gradient: PyTree | ArrayLike) -> Array:
        """Return the inverse-metric norm of a gradient PyTree."""
        gradient, _ = _tree_vector(
            gradient,
            self.projection.layout,
            "gradient",
        )
        return self.projection.natural_norm(gradient)
