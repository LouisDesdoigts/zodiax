"""Composable numerical expressions and recursive resolution."""

from __future__ import annotations

from abc import abstractmethod
from contextvars import ContextVar
from typing import Any

import jax.tree_util as jtu

from ..base import Module, _is_runtime_field

__all__ = ["Expression", "resolve"]


_ACTIVE_EXPRESSIONS: ContextVar[tuple[int, ...]] = ContextVar(
    "zodiax_active_expressions", default=()
)


class Expression(Module):
    """Abstract value evaluated from stored leaves and call-local context.

    Subclasses should remain pure and compatible with the JAX transformations
    promised by their consumer. An Expression may contain another Expression; use
    :func:`resolve` to evaluate nested compositions completely.
    """

    @abstractmethod
    def evaluate(self, **context: Any) -> Any:
        """Evaluate this value using the supplied named context."""

    def __call__(self, **context: Any) -> Any:
        """Resolve this expression using the supplied named context."""
        return resolve(self, **context)


def _runtime_path(root: Any, path: tuple[Any, ...]) -> bool:
    """Return whether one concrete PyTree path crosses a runtime field."""
    value = root
    for key in path:
        if isinstance(key, jtu.GetAttrKey):
            if isinstance(value, Module) and _is_runtime_field(value, key.name):
                return True
            value = object.__getattribute__(value, key.name)
        elif isinstance(key, jtu.SequenceKey):
            value = value[key.idx]
        elif isinstance(key, jtu.DictKey):
            value = value[key.key]
        else:
            # Custom PyTree keys do not describe a public structural field, so they
            # cannot introduce Zodiax runtime metadata themselves.
            return False
    return False


def resolve(value: Any, *, runtime: bool = False, **context: Any) -> Any:
    """Recursively evaluate Expression values in a value or PyTree.

    Resolution proceeds from each outer Expression into its returned value, then
    through any Expression leaves in that result. Ordinary leaves and the input
    object are not mutated. Runtime-declared module fields remain opaque by default;
    ``runtime=True`` explicitly evaluates them using the supplied context.
    """
    if type(runtime) is not bool:
        raise TypeError("runtime must be a bool.")
    return _resolve(value, context, runtime)


def _resolve(
    value: Any,
    context: dict[str, Any],
    runtime: bool,
) -> Any:
    if isinstance(value, Expression):
        marker = id(value)
        active = _ACTIVE_EXPRESSIONS.get()
        if marker in active:
            raise ValueError("Expression evaluation contains a cycle.")
        if len(active) >= 100:
            raise ValueError("Expression resolution exceeded 100 nested evaluations.")

        token = _ACTIVE_EXPRESSIONS.set(active + (marker,))
        try:
            output = value.evaluate(**context)
            return _resolve(output, context, runtime)
        finally:
            _ACTIVE_EXPRESSIONS.reset(token)

    is_expression = lambda leaf: isinstance(leaf, Expression)

    def evaluate(path: tuple[Any, ...], leaf: Any) -> Any:
        if not isinstance(leaf, Expression):
            return leaf
        if not runtime and _runtime_path(value, path):
            return leaf
        return _resolve(leaf, context, runtime)

    return jtu.tree_map_with_path(evaluate, value, is_leaf=is_expression)
