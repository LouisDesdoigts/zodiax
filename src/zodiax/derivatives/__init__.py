"""Structured derivatives, curvature matrices, and reusable decompositions."""

from . import containers, decompositions, operations

_MODULES = (containers, decompositions, operations)

for _module in _MODULES:
    globals().update({name: getattr(_module, name) for name in _module.__all__})

__all__ = [name for module in _MODULES for name in module.__all__]
