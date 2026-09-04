"""Explicit symbolic callables for portable, non-importing reconstruction."""

from __future__ import annotations

from collections.abc import Callable
import inspect
from threading import RLock
from typing import Any

import equinox as eqx
import jax.nn as jnn
import jax.random as jr

from ._schema import _validate_callable_identifier

__all__ = ["register_callable"]


_LOCK = RLock()
_BY_IDENTIFIER: dict[str, Callable[..., Any]] = {}
_BY_IDENTITY: dict[int, tuple[Callable[..., Any], str]] = {}


def register_callable(
    identifier: str,
    function: Callable[..., Any] | None = None,
):
    """Register a trusted symbolic callable under a stable identifier.

    Registration is explicit and never causes archive-directed imports. The same
    registration must be performed before loading an archive in a new process.
    This function may also be used as a decorator.

    Parameters
    ----------
    identifier : str
        Stable dotted identifier, for example ``"my_package.activations.square"``.
    function : callable, optional
        Stateless callable represented by ``identifier``.

    Returns
    -------
    function : callable
        The registered function, making decorator use transparent.
    """
    _validate_callable_identifier(identifier, "callable identifier")

    if function is None:

        def decorator(value):
            return register_callable(identifier, value)

        return decorator

    if not callable(function):
        raise TypeError("function must be callable.")

    with _LOCK:
        existing = _BY_IDENTIFIER.get(identifier)
        if existing is not None and existing is not function:
            raise ValueError(
                f"Callable identifier {identifier!r} is already registered."
            )

        identity = id(function)
        existing_identity = _BY_IDENTITY.get(identity)
        if (
            existing_identity is not None
            and existing_identity[0] is function
            and existing_identity[1] != identifier
        ):
            raise ValueError(
                f"This callable is already registered as {existing_identity[1]!r}."
            )

        _BY_IDENTIFIER[identifier] = function
        _BY_IDENTITY[identity] = (function, identifier)
    return function


def _callable_identifier(function: Callable[..., Any]) -> str | None:
    """Return the registered identifier for one exact callable object."""
    with _LOCK:
        registered = _BY_IDENTITY.get(id(function))
        if registered is None or registered[0] is not function:
            return None
        return registered[1]


def _resolve_callable(identifier: str, path: str) -> Callable[..., Any]:
    """Resolve only a callable already present in the trusted registry."""
    with _LOCK:
        function = _BY_IDENTIFIER.get(identifier)
    if function is None:
        raise ValueError(
            f"Callable {identifier!r} at {path} is not registered. Register it "
            "explicitly before loading."
        )
    return function


def _register_builtin_callables() -> None:
    """Register stable symbols used by built-in numerical definitions."""
    random_functions = (
        "bernoulli",
        "categorical",
        "choice",
        "multivariate_normal",
        "normal",
        "permutation",
        "poisson",
        "randint",
        "truncated_normal",
        "uniform",
    )
    for name in random_functions:
        register_callable(f"jax.random.{name}", getattr(jr, name))

    try:
        parameters = inspect.signature(eqx.nn.MLP).parameters
    except (TypeError, ValueError):
        parameters = {}

    # Use Zodiax-owned wire identifiers rather than Equinox's private lambda name.
    register_callable("jax.nn.relu", jnn.relu)
    final_activation = parameters.get("final_activation")
    if final_activation is not None:
        identity = final_activation.default
        if (
            identity is not inspect.Parameter.empty
            and identity is not jnn.relu
            and callable(identity)
        ):
            register_callable("equinox.nn.identity", identity)


_register_builtin_callables()
