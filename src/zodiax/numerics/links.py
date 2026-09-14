"""Shared numerical values stored once and referenced by key.

Linked holds the value; Deferred holds only its owner's key. Population replaces
both with that value in a temporary tree, leaving the original model unchanged.
Normal model.resolve calls perform this step before evaluating expressions.
"""

from __future__ import annotations

from numbers import Number
from typing import Any
from uuid import uuid4

import equinox as eqx
import jax.tree_util as jtu

from .arrays import as_array
from .expressions import Expression, resolve

__all__ = ["Deferred", "Linked", "populate", "validate_links"]


def populate(tree: Any) -> Any:
    """Return a copy with each Linked and Deferred replaced by its owner's value.

    First collect all owners, including those nested inside other linked values.
    Then replace links recursively. This allows references to appear before their
    owners in the tree. Duplicate owners, missing owners, and cycles raise ValueError.

    Population connects shared values without evaluating expressions. Call it
    inside the differentiated function so every use contributes to the gradient
    of the value stored once in the original model. Model.resolve does this
    automatically before evaluating available expressions.
    """
    # Pass 1: build a registry of owners and record the keys used by references.
    # These collections belong to this call; nothing is stored on the model.
    owners: dict[str, Linked] = {}
    deferred: set[str] = set()

    # Normally JAX traverses Linked.value and skips Deferred's static-only fields.
    # Treating both classes as leaves lets us inspect the link objects themselves.
    is_link = lambda value: isinstance(value, (Linked, Deferred))

    def collect(subtree: Any) -> None:
        """Register links in this subtree, including links inside owned values."""
        for node in jtu.tree_leaves(subtree, is_leaf=is_link):
            if isinstance(node, Linked):
                # Even repeating the same owner object duplicates its tree leaves.
                if node.key in owners:
                    raise ValueError(f"Link {node.key!r} has multiple Linked owners.")
                owners[node.key] = node
                collect(node.value)
            elif isinstance(node, Deferred):
                deferred.add(node.key)

    # Finish collecting before checking references: an owner may appear later in
    # the tree. The set difference finds keys with no owner anywhere in this tree.
    collect(tree)
    missing = deferred.difference(owners)
    if missing:
        keys = ", ".join(repr(key) for key in sorted(missing))
        raise ValueError(f"Dangling deferred link(s): {keys}.")

    # Pass 2: replace each owner and reference with the owner's populated value.
    # Cache completed values by key so shared subtrees are only expanded once.
    populated: dict[str, Any] = {}

    # Track owners whose values are still being populated. Nested calls share
    # this list; reaching an unfinished owner again indicates a dependency cycle.
    active: list[str] = []

    def replace(node: Any) -> Any:
        """Replace one link, recursively populating its owned value if needed."""
        if not is_link(node):
            return node

        key = node.key
        if key in populated:
            return populated[key]

        if key in active:
            # Keep the circular part of the path, for example a -> b -> a.
            start = active.index(key)
            cycle = active[start:] + [key]
            chain = " -> ".join(cycle)
            raise ValueError(f"Linked dependency cycle detected: {chain}.")

        # Populate dependencies before caching this value. Mark the owner active
        # during that work, then restore the caller's path even if a child fails.
        active.append(key)
        try:
            value = jtu.tree_map(replace, owners[key].value, is_leaf=is_link)
        finally:
            active.pop()

        populated[key] = value
        return value

    # Rebuild the outer tree without changing the original. Each link is one
    # replacement slot, so its value can be an array or an entire populated PyTree.
    return jtu.tree_map(replace, tree, is_leaf=is_link)


def validate_links(tree: Any) -> None:
    """Check for duplicate owners, missing owners, and dependency cycles.

    Use the same traversal as population and discard the temporary result.
    Numerical expressions are not evaluated.
    """
    populate(tree)


class Linked(Expression):
    """Store a shared array, Expression, or PyTree once in a model.

    Place this owner in one tree position and use defer() for every other use.
    Repeating the same Linked object in two positions still duplicates its leaves.

    Numerical scalars and NumPy/JAX arrays become strongly typed JAX arrays at
    construction. Containers retain their structure and contents; supply arrays
    for numerical leaves that should participate in JAX transformations.

    The static string key identifies the owner and its references. A UUID is
    generated when omitted. Supply a descriptive key to give independently
    constructed models the same link identity. Keys survive tree transformations
    and serialisation.
    """

    value: Any
    key: str = eqx.field(static=True)

    def __init__(
        self,
        value: Any,
        *,
        key: str | None = None,
        alias: Any = None,
    ):
        if key is None:
            key = uuid4().hex
        if not isinstance(key, str):
            raise TypeError("key must be a string.")
        if not key:
            raise ValueError("key must not be empty.")

        if isinstance(value, Number) or eqx.is_array(value):
            value = as_array(value)

        self.alias = alias
        self.value = value
        self.key = key

    def defer(self) -> Deferred:
        """Return a reference containing only this owner's static key."""
        return Deferred(self.key)

    def evaluate(self, **context: Any) -> Any:
        """Resolve the owned value when evaluating this wrapper directly.

        Normal model resolution replaces the wrapper during population first.
        """
        return resolve(self.value, **context)


class Deferred(Expression):
    """A reference populated from the Linked owner with the same key.

    Resolve a tree containing both the owner and its references. A detached
    reference cannot find its owner from the key alone.
    """

    key: str = eqx.field(static=True)

    def __init__(self, key: str):
        if not isinstance(key, str):
            raise TypeError("key must be a string.")
        if not key:
            raise ValueError("key must not be empty.")

        self.alias = None
        self.key = key

    def evaluate(self, **context: Any) -> Any:
        """Report a reference that reached evaluation without being populated."""
        raise ValueError(
            f"Link {self.key!r} has not been populated. Call resolve() on a tree "
            "containing both this reference and its Linked owner."
        )
