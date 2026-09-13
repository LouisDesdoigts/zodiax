"""Composable numerical expressions and recursive resolution."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from contextvars import ContextVar
import inspect
from typing import Any

import jax.tree_util as jtu
from jax.core import get_opaque_trace_state

from ..base import Module

__all__ = ["Expression", "resolve"]


_ACTIVE_EXPRESSIONS: ContextVar[tuple[int, ...]] = ContextVar(
    "zodiax_active_expressions", default=()
)
"""IDs of expressions currently being evaluated, used to detect nested calls and cycles.

ContextVar lets nested resolve calls share this stack within the current thread
or task, without extra arguments on evaluate. Each evaluation adds its ID and
restores the previous stack in finally, even if it fails. This bookkeeping is
separate from **context, which supplies scientific inputs such as coordinates.
"""


_RESOLVED_EXPRESSIONS: ContextVar[tuple[Any, dict] | None] = ContextVar(
    "zodiax_resolved_expressions", default=None
)
"""Temporary results shared by nested resolve calls to avoid repeating child work.

Stores the JAX trace state and a dictionary keyed by expression and context-value
IDs. Entries keep those objects alive so their IDs remain valid. Results are
reused only within the same JAX trace, so temporary tracers cannot escape a nested
transformation through this cache. The outer resolve call clears it when finished.
"""


def _missing_context(
    calculation: Callable[..., Any], context: dict[str, Any], *args: Any
) -> tuple[str, ...]:
    """Find required named arguments absent from a calculation's supplied inputs."""
    # Read the ordinary Python signature and match the supplied arguments. Any
    # positional inputs, such as fwd's x, are supplied through args so they are
    # already accounted for when checking the remaining named context inputs.
    # bind_partial allows missing required arguments, but still rejects invalid ones.
    signature = inspect.signature(calculation)
    supplied = signature.bind_partial(*args, **context).arguments

    # A context dependency must accept a keyword and have no default value.
    # Positional-only arguments and *args/**kwargs are not named context inputs.
    keyword_kinds = (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )
    missing = []
    for name, parameter in signature.parameters.items():
        accepts_keyword = parameter.kind in keyword_kinds
        required = parameter.default is inspect.Parameter.empty
        provided = name in supplied

        if accepts_keyword and required and not provided:
            missing.append(name)

    return tuple(missing)


def _has_unresolved_operand(value: Any) -> bool:
    """Detect unresolved operands outside ordinary Module boundaries."""
    # A Module may supply context to its remaining expressions when called later.
    # Expressions are Modules too, but remain unresolved numerical operands.
    is_module = lambda leaf: isinstance(leaf, Module)
    for leaf in jtu.tree_leaves(value, is_leaf=is_module):
        if isinstance(leaf, Expression):
            return True

    return False


def _evaluate_expression(expression: Expression, context: dict[str, Any]) -> Any:
    """Try the calculation, then prepare children if it cannot finish yet."""
    # Reject recursive self-dependencies before starting another evaluation.
    marker = id(expression)
    active = _ACTIVE_EXPRESSIONS.get()
    if marker in active:
        raise ValueError("Expression evaluation contains a cycle.")
    if len(active) >= 100:
        raise ValueError("Expression resolution exceeded 100 nested evaluations.")

    # Keep the owner active during evaluation and any later child preparation.
    token = _ACTIVE_EXPRESSIONS.set(active + (marker,))
    try:
        # The calculation chooses its required operands and their local context.
        # Stored callable transforms and unused fields need not become arrays.
        if not _missing_context(expression.evaluate, context):
            try:
                output = expression.evaluate(**context)
                return _resolve_tree(output, context)
            except _Unresolved:
                # A requested numerical operand could not provide a value. Keep
                # partial progress through the child-preparation step below.
                pass

        # Missing context or a deferred operand prevents the owner's calculation,
        # but independent children can still resolve. Return the rebuilt owner.
        return expression.resolve_children(**context)
    finally:
        # Restore the caller's stack on success, early return, or any exception.
        _ACTIVE_EXPRESSIONS.reset(token)


def _resolve_tree(value: Any, context: dict[str, Any]) -> Any:
    """Resolve expression leaves while preserving ordinary PyTree structure."""
    cache = None
    cache_state = _RESOLVED_EXPRESSIONS.get()
    if cache_state is not None:
        trace_state, values = cache_state
        if trace_state == get_opaque_trace_state():
            cache = values
    context_ids = []
    for name in sorted(context):
        context_ids.append((name, id(context[name])))
    context_key = tuple(context_ids)

    def resolve_leaf(leaf: Any) -> Any:
        if not isinstance(leaf, Expression):
            return leaf

        key = (id(leaf), context_key)
        if cache is not None and key in cache:
            _, _, resolved = cache[key]
            return resolved

        resolved = _evaluate_expression(leaf, context)
        if cache is not None:
            cache[key] = (leaf, context, resolved)
            if isinstance(resolved, Expression):
                # A later operand request may refer to the partially prepared
                # copy rather than its original. It has the same pending result.
                prepared_key = (id(resolved), context_key)
                cache[prepared_key] = (resolved, context, resolved)
        return resolved

    # Each expression controls its own calculation. Ordinary Modules retain their
    # PyTree structure while their expression fields are resolved independently.
    is_expression = lambda leaf: isinstance(leaf, Expression)
    return jtu.tree_map(resolve_leaf, value, is_leaf=is_expression)


def resolve(
    value: Any,
    *,
    populate: bool = True,
    **context: Any,
) -> Any:
    """Populate links and partially evaluate context-ready expressions.

    Link population runs once at the outermost resolution by default. Expressions
    evaluate when their required named context is available, choosing the operands
    and local context they need. If the calculation cannot finish, inherited child
    preparation retains a copy with available descendants resolved.
    Ordinary Modules may retain expressions for later local evaluation, including
    when resolved from inside another expression. Independent branches continue
    resolving, and ordinary evaluation errors are not suppressed.

    ``populate`` is keyword-only: use ``resolve(value, populate=False)`` to skip
    link population. ``**context`` forwards named scientific inputs to expressions.
    """
    if type(populate) is not bool:
        raise TypeError("populate must be a bool.")

    # Share results within this call, and release them even if evaluation fails.
    nested = bool(_ACTIVE_EXPRESSIONS.get())
    cache_token = None
    if not nested:
        cache_token = _RESOLVED_EXPRESSIONS.set((get_opaque_trace_state(), {}))
    try:
        # Connect shared owners once, at the outermost resolution boundary.
        if populate and not nested:
            # links defines Expression subclasses, so importing it at module scope
            # would require Expression before this module has finished defining it.
            from .links import populate as populate_links

            value = populate_links(value)

        resolved = _resolve_tree(value, context)

        # This check applies only to an operand actually requested by a calculation,
        # not to every field stored on its parent. Modules may remain partial.
        if nested and _has_unresolved_operand(resolved):
            raise _Unresolved("A required expression remains unresolved.")
        return resolved
    finally:
        if cache_token is not None:
            _RESOLVED_EXPRESSIONS.reset(cache_token)


class _Unresolved(ValueError):
    """Interrupt an incomplete calculation so its owning expression is retained.

    Only this dedicated signal is caught by the expression evaluator. Ordinary
    exceptions, including other ValueErrors, remain visible to the caller.
    """


class Expression(Module):
    """Abstract value evaluated from stored leaves and call-local context.

    An ordinary Module contains values; an Expression represents a value. Calling
    ``resolve`` on a ready Expression invokes its ``evaluate`` method and returns
    the result. Resolving an ordinary Module instead retains the Module and replaces
    its ready expression fields with their results.

    Subclasses resolve required operands inside ``evaluate``, supplying their local
    context. Unused expression fields and stored callable transforms do not prevent
    evaluation. When a calculation cannot finish, inherited ``resolve_children``
    prepares available descendants using the supplied context.
    Each context name must denote the same scientific input throughout the tree.
    Use distinct names for derived inputs, such as ``coordinates`` for incoming
    coordinates and ``sample_coordinates`` for a child's transformed coordinates.
    Implementations remain pure and compatible with their promised JAX transforms.
    """

    def resolve_children(self, **context: Any) -> Expression:
        """Return a copy with available children resolved, retaining this expression.

        The resolver calls this when evaluation lacks a required input. Child
        expressions that still need inputs are retained without preventing their
        siblings from resolving. Static fields remain part of the tree structure.
        The outer ``resolve`` call handles link population before this method runs.

        The default forwards the supplied context unchanged. Callers and class
        construction define which inputs are valid at this stage; it does not infer
        coordinate frames or enforce scientific input combinations. Customising
        this method is optional, not required to implement an expression.
        A child requiring a distinctly named derived input remains deferred until
        its owner supplies that input; the incoming context cannot stand in for it.
        """

        # Map over immediate children: keep this receiver and recursively resolve
        # each dynamic field. Each child is one slot, so its replacement can have
        # a different structure, such as an array replacing a Map expression.
        def resolve_child(child: Any) -> Any:
            return _resolve_tree(child, context)

        is_child = lambda value: value is not self
        return jtu.tree_map(resolve_child, self, is_leaf=is_child)

    @abstractmethod
    def evaluate(self, **context: Any) -> Any:
        """Resolve required operands in their local context and return this value.

        Declare required named context explicitly in the signature. Missing context
        retains this expression with its prepared children. Resolve numerical
        operands with their appropriate context inside this method, so a missing
        required value defers the calculation. Stored operators may instead be
        called with explicit inputs. A direct call runs the calculation itself.
        """

    def __call__(self, **context: Any) -> Any:
        """Resolve this expression using the supplied named context."""
        return resolve(self, **context)
