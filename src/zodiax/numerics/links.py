"""Single-owner numerical links for Zodiax PyTrees.

``Linked`` values own numerical leaves exactly once, while key-only ``Deferred``
values refer to those owners. Population replaces both forms in an ephemeral tree,
so every use remains connected to the owner's single input tracer.
"""

from __future__ import annotations

from numbers import Number
from typing import Any
from uuid import uuid4

import equinox as eqx
import jax.tree_util as jtu
import numpy as onp
import wadler_lindig as wl
from jax import Array

from .arrays import as_array
from .expressions import Expression, resolve

__all__ = ["Deferred", "Linked", "populate", "validate_links"]


def _short_key(key: str) -> str:
    """Shorten generated UUID-like keys without hiding semantic identities."""
    if len(key) == 32 and all(character in "0123456789abcdef" for character in key):
        return f"{key[:8]}…{key[-4:]}"
    return key


def _validate_key(key: Any) -> None:
    """Validate one stable link identity without generating a replacement."""
    if not isinstance(key, str):
        raise TypeError("key must be a string.")
    if not key:
        raise ValueError("key must not be empty.")


def _link_pdoc(
    owner: Any,
    name: str,
    fields: list[tuple[str, Any]],
    **kwargs: Any,
) -> Any:
    """Build a compact Equinox-compatible document for a link node."""
    alias = getattr(owner, "alias", None)
    if alias is not None:
        fields = [("alias", alias), *fields]
    return wl.bracketed(
        begin=wl.TextDoc(f"{name}("),
        docs=wl.named_objs(fields, **kwargs),
        sep=wl.comma,
        end=wl.TextDoc(")"),
        indent=kwargs["indent"],
    )


class Linked(Expression):
    """Own a value that can be referenced elsewhere without duplicating leaves.

    Parameters
    ----------
    value
        The canonical array, PyTree, or Expression value. Its dynamic leaves occur
        only here in the unresolved model. Python numerical scalars and NumPy/JAX
        arrays are normalised to strongly typed JAX arrays; ambiguous list, tuple,
        and mapping PyTrees are retained as containers.
    key
        Optional stable, serialisable identity for the link. If omitted, a UUID
        is generated once at construction and retained by serialisation and tree
        transformations. Supplying a semantic key makes independently rebuilt
        model topologies identical.
    """

    value: Any
    key: str = eqx.field(static=True, repr=False)

    def __init__(
        self,
        value: Any,
        *,
        key: str | None = None,
        alias: Any = None,
    ):
        if key is None:
            key = uuid4().hex
        elif not isinstance(key, str):
            raise TypeError("key must be a string or None.")
        _validate_key(key)

        if isinstance(value, (Number, Array, onp.ndarray, onp.generic)):
            value = as_array(value)

        self.alias = alias
        self.value = value
        self.key = key

    def defer(self) -> Deferred:
        """Return a zero-dynamic-leaf deferred link to this linked value."""
        return Deferred(self.key)

    def evaluate(self, **context: Any) -> Any:
        """Evaluate the canonical value directly.

        Deferred links still require :func:`populate`; evaluating an unpopulated
        link raises a diagnostic error.
        """
        return resolve(self.value, **context)

    def __pdoc__(self, **kwargs: Any) -> Any:
        return _link_pdoc(
            self,
            "Linked",
            [("key", _short_key(self.key)), ("value", self.value)],
            **kwargs,
        )

    def __zodiax_validate__(self) -> None:
        """Validate constructor-free archive reconstruction."""
        _validate_key(self.key)


class Deferred(Expression):
    """A static link replaced by :func:`populate` before evaluation."""

    key: str = eqx.field(static=True, repr=False)

    def __init__(self, key: str):
        _validate_key(key)
        self.key = key

    def evaluate(self, **context: Any) -> Any:
        del context
        raise ValueError(
            f"Link {self.key!r} has not been populated. Call populate() on the "
            "containing model inside the evaluated function."
        )

    def __pdoc__(self, **kwargs: Any) -> Any:
        return _link_pdoc(
            self,
            "Deferred",
            [("key", _short_key(self.key))],
            **kwargs,
        )

    def __zodiax_validate__(self) -> None:
        """Validate constructor-free archive reconstruction."""
        _validate_key(self.key)


def validate_links(tree: Any) -> None:
    """Validate linked ownership, deferred links, and dependency cycles.

    Raises
    ------
    ValueError
        If multiple owners use one key, a deferred link is dangling, or linked
        owners form a dependency cycle.
    """
    owners, deferred = _collect_links(tree)
    _check_dangling(owners, deferred)

    populator = _Populator(owners)
    for key in owners:
        populator.resolve_key(key)


def populate(tree: Any) -> Any:
    """Return an ephemeral tree with linked owners and deferred links populated.

    This is a two-pass operation. The first pass discovers every owner and
    validates the complete topology, so forward deferred links are supported. The
    second pass replaces both an owner and all its deferred links with the owner's
    recursively populated value.

    The unresolved input remains unchanged and contains each linked numerical
    subtree exactly once. Calling this function inside a differentiated
    computation therefore makes all populated uses contribute gradients to the
    canonical owner.
    """
    owners, deferred = _collect_links(tree)
    _check_dangling(owners, deferred)

    populator = _Populator(owners)
    for key in owners:
        populator.resolve_key(key)
    return populator.tree(tree)


def _is_link_node(value: Any) -> bool:
    return isinstance(value, (Linked, Deferred))


def _collect_links(tree: Any) -> tuple[dict[str, Linked], set[str]]:
    owners: dict[str, Linked] = {}
    deferred: set[str] = set()

    def scan(subtree: Any) -> None:
        # Treat link nodes as leaves for this pass. Linked values are scanned in
        # a separate recursive call so the owner itself and nested dependencies
        # are both visible.
        nodes = jtu.tree_leaves(subtree, is_leaf=_is_link_node)
        for node in nodes:
            if isinstance(node, Linked):
                if node.key in owners:
                    raise ValueError(f"Link {node.key!r} has multiple Linked owners.")
                owners[node.key] = node
                scan(node.value)
            elif isinstance(node, Deferred):
                deferred.add(node.key)

    scan(tree)
    return owners, deferred


def _check_dangling(owners: dict[str, Linked], deferred: set[str]) -> None:
    dangling = deferred.difference(owners)
    if dangling:
        keys = ", ".join(repr(key) for key in sorted(dangling))
        raise ValueError(f"Dangling deferred link(s): {keys}.")


class _Populator:
    """Resolve a validated link registry while detecting dependency cycles."""

    def __init__(self, owners: dict[str, Linked]):
        self.owners = owners
        self.values: dict[str, Any] = {}
        self.active: tuple[str, ...] = ()

    def tree(self, tree: Any) -> Any:
        return jtu.tree_map(self._replace, tree, is_leaf=_is_link_node)

    def _replace(self, node: Any) -> Any:
        if isinstance(node, (Linked, Deferred)):
            return self.resolve_key(node.key)
        return node

    def resolve_key(self, key: str) -> Any:
        if key in self.values:
            return self.values[key]

        if key in self.active:
            start = self.active.index(key)
            cycle = self.active[start:] + (key,)
            chain = " -> ".join(cycle)
            raise ValueError(f"Linked dependency cycle detected: {chain}.")

        owner = self.owners[key]
        previous = self.active
        self.active = previous + (key,)
        try:
            value = self.tree(owner.value)
        finally:
            self.active = previous

        self.values[key] = value
        return value
