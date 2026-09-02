"""Generic JAX-random expressions driven by explicit numerical state."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import blake2s
from operator import index as integer_index
from typing import Any
from uuid import uuid4

import equinox as eqx
import jax.numpy as np
import jax.random as jr
import wadler_lindig as wl

from .arrays import as_array
from .expressions import Expression, resolve
from .state import State

__all__ = ["Random"]


_JAX_STATIC_ARGUMENTS = frozenset(
    ("axis", "dtype", "independent", "method", "mode", "replace", "shape")
)


def _validate_stream(stream: Any) -> str:
    """Return one stable, serialisable random-stream identity."""
    if not isinstance(stream, str):
        raise TypeError("stream must be a string.")
    if not stream:
        raise ValueError("stream must not be empty.")
    return stream


def _stream_words(stream: str) -> tuple[int, int, int, int]:
    """Map a stable stream identity onto four JAX fold-in words."""
    digest = blake2s(stream.encode(), digest_size=16).digest()
    return tuple(
        int.from_bytes(digest[start : start + 4], "little") for start in range(0, 16, 4)
    )


def _short_stream(stream: str) -> str:
    """Keep generated stream identities legible in normal object prints."""
    if len(stream) == 32 and all(
        character in "0123456789abcdef" for character in stream
    ):
        return f"{stream[:8]}…{stream[-4:]}"
    return stream


def _callable_name(fn: Callable[..., Any]) -> str:
    """Return one concise diagnostic name for a stored callable."""
    name = getattr(fn, "__name__", None)
    if isinstance(name, str):
        return name
    return type(fn).__name__


def _static_names(value: Any, kwargs: dict[str, Any]) -> set[str]:
    """Return explicit plus conventional JAX trace-time argument names."""
    if value is None:
        names = set()
    elif isinstance(value, str):
        names = {value}
    else:
        try:
            names = set(value)
        except TypeError as error:
            raise TypeError(
                "static must be a string, string iterable, or None."
            ) from error
        if any(not isinstance(name, str) for name in names):
            raise TypeError("static argument names must be strings.")

    missing = names.difference(kwargs)
    if missing:
        listed = ", ".join(repr(name) for name in sorted(missing))
        raise ValueError(f"Static Random argument(s) were not supplied: {listed}.")
    return names.union(_JAX_STATIC_ARGUMENTS.intersection(kwargs))


def _static_argument(name: str, value: Any) -> Any:
    """Canonicalise common JAX configuration values as hashable metadata."""
    if name == "shape":
        if value is None:
            return None
        values = value if isinstance(value, (tuple, list)) else (value,)
        shape = []
        for size in values:
            if isinstance(size, bool):
                raise TypeError("Random shape must contain integers.")
            try:
                size = integer_index(size)
            except TypeError as error:
                raise TypeError("Random shape must contain integers.") from error
            if size < 0:
                raise ValueError("Random shape must contain nonnegative sizes.")
            shape.append(size)
        return tuple(shape)
    if name == "dtype":
        try:
            return np.dtype(value).name
        except (TypeError, ValueError) as error:
            raise TypeError("Random dtype must be understood by JAX.") from error
    if name == "axis":
        if isinstance(value, bool):
            raise TypeError("Random axis must be an integer.")
        try:
            return integer_index(value)
        except TypeError as error:
            raise TypeError("Random axis must be an integer.") from error
    if name in ("independent", "replace") and type(value) is not bool:
        raise TypeError(f"Random {name} must be a bool.")
    if name in ("method", "mode") and not isinstance(value, str):
        raise TypeError(f"Random {name} must be a string.")

    try:
        hash(value)
    except TypeError as error:
        raise TypeError(f"Static Random argument {name!r} must be hashable.") from error
    return value


def _pdoc(owner: Any, fields: list[tuple[str, Any]], **kwargs: Any) -> Any:
    """Build a compact Equinox-compatible random-expression representation."""
    alias = getattr(owner, "alias", None)
    if alias is not None:
        fields = [("alias", alias), *fields]
    return wl.bracketed(
        begin=wl.TextDoc("Random("),
        docs=wl.named_objs(fields, **kwargs),
        sep=wl.comma,
        end=wl.TextDoc(")"),
        indent=kwargs["indent"],
    )


class Random(Expression):
    """Evaluate any JAX-style key-first random callable.

    ``fn`` receives a normal JAX key as its first positional argument. Positional
    and named callable arguments are stored as ordinary PyTrees and may themselves
    contain numerical expressions. By default the root ``key`` and current
    ``index`` are read from call-local :class:`State`; either may instead be stored
    explicitly.

    Every instance owns a stable static ``stream`` identity. Resolution derives its
    key exclusively with :func:`jax.random.fold_in`, so the result remains natively
    interoperable with JAX random functions. A State without ``index`` denotes a
    fixed realisation at index zero. Conventional JAX trace-time options such as
    ``shape``, ``dtype``, and ``axis`` are stored as static metadata automatically;
    ``static=`` identifies any additional named callable options that must also be
    known while tracing.
    """

    fn: Callable[..., Any] | None = eqx.field(default=None, static=True, repr=False)
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = eqx.field(default_factory=dict)
    options: tuple[tuple[str, Any], ...] = eqx.field(
        default=(), static=True, repr=False
    )
    key: Any = None
    index: Any = None
    stream: str = eqx.field(default="", static=True, repr=False)

    def __init__(
        self,
        fn: Callable[..., Any],
        *args: Any,
        key: Any = None,
        index: Any = None,
        stream: str | None = None,
        static: Any = None,
        alias: Any = None,
        **kwargs: Any,
    ):
        if not callable(fn):
            raise TypeError("fn must be callable.")
        if "key" in kwargs:
            raise TypeError("Pass the random root through Random key=, not fn kwargs.")
        if stream is None:
            stream = uuid4().hex

        static_names = _static_names(static, kwargs)
        options = tuple(
            (name, _static_argument(name, kwargs[name]))
            for name in sorted(static_names)
        )

        self.alias = alias
        self.fn = fn
        self.args = tuple(as_array(value) for value in args)
        self.kwargs = {
            name: as_array(kwargs[name])
            for name in sorted(kwargs)
            if name not in static_names
        }
        self.options = options
        self.key = as_array(key)
        self.index = as_array(index)
        self.stream = _validate_stream(stream)

    def _root_key(self, state: State | None, context: dict[str, Any]) -> Any:
        """Resolve an explicit key or read the conventional State root key."""
        if self.key is not None:
            return resolve(self.key, **context)
        if not isinstance(state, State):
            raise TypeError("Random requires key= or a State containing key.")
        try:
            key = state.value("key")
        except KeyError as error:
            raise ValueError("Random State must contain a key entry.") from error
        return resolve(key, state=state, **context)

    def _index_words(
        self,
        state: State | None,
        context: dict[str, Any],
    ) -> tuple[Any, ...]:
        """Resolve an index into one or two words accepted by JAX ``fold_in``."""
        if self.index is not None:
            index = resolve(self.index, **context)
        elif isinstance(state, State):
            try:
                index = state.value("index")
            except KeyError:
                index = 0
            else:
                index = resolve(index, state=state, **context)
        else:
            index = 0

        index = np.asarray(index)
        if index.ndim != 0 or not np.issubdtype(index.dtype, np.integer):
            raise TypeError("Random index must be a scalar integer array.")
        if index.dtype.itemsize <= 4:
            return (np.asarray(index, dtype=np.uint32),)
        if index.dtype.itemsize <= 8:
            # ``fold_in`` consumes one uint32 at a time. Preserve a 64-bit index
            # without concretising it by folding its low and high words in a fixed
            # order. These bit operations remain valid under ``jit`` and ``scan``.
            index = np.asarray(index, dtype=np.uint64)
            low = np.asarray(index, dtype=np.uint32)
            high = np.asarray(index >> np.uint64(32), dtype=np.uint32)
            return (low, high)
        raise TypeError("Random index must use a 64-bit or narrower dtype.")

    def evaluate(self, *, state: State | None = None, **context: Any) -> Any:
        """Derive one ordinary JAX key and invoke the stored random callable."""
        nested = dict(context)
        if state is not None:
            nested["state"] = state

        key = self._root_key(state, nested)
        index_words = self._index_words(state, nested)
        try:
            for word in index_words:
                key = jr.fold_in(key, word)
            for word in _stream_words(self.stream):
                key = jr.fold_in(key, word)
        except (TypeError, ValueError) as error:
            raise TypeError("Random key must be one scalar JAX PRNG key.") from error

        args = resolve(self.args, **nested)
        kwargs = dict(self.options)
        kwargs.update(resolve(self.kwargs, **nested))
        return self.fn(key, *args, **kwargs)

    def __pdoc__(self, **kwargs: Any) -> Any:
        fields = [("fn", _callable_name(self.fn))]
        if self.args:
            fields.append(("args", self.args))
        fields.extend(sorted((*self.options, *self.kwargs.items())))
        if self.key is not None:
            fields.append(("key", self.key))
        if self.index is not None:
            fields.append(("index", self.index))
        fields.append(("stream", _short_stream(self.stream)))
        return _pdoc(self, fields, **kwargs)

    def __zodiax_validate__(self) -> None:
        """Validate constructor-free archive reconstruction."""
        if not callable(self.fn):
            raise TypeError("fn must be callable.")
        if type(self.args) is not tuple:
            raise TypeError("args must be stored as a tuple.")
        if type(self.kwargs) is not dict:
            raise TypeError("kwargs must be stored as a dict.")
        if tuple(self.kwargs) != tuple(sorted(self.kwargs)):
            raise ValueError("kwargs must be stored in canonical key order.")
        if type(self.options) is not tuple:
            raise TypeError("options must be stored as a tuple.")
        if any(
            type(option) is not tuple
            or len(option) != 2
            or not isinstance(option[0], str)
            for option in self.options
        ):
            raise TypeError("options must contain (name, value) pairs.")
        option_names = tuple(name for name, _ in self.options)
        if option_names != tuple(sorted(option_names)):
            raise ValueError("options must be stored in canonical key order.")
        if len(option_names) != len(set(option_names)):
            raise ValueError("options must contain unique names.")
        if set(option_names).intersection(self.kwargs):
            raise ValueError("Random options and kwargs must not overlap.")
        for name, value in self.options:
            _static_argument(name, value)
        _validate_stream(self.stream)
