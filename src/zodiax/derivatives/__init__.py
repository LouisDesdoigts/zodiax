"""Structured Jacobian, exact Hessian, and curvature representations."""

from . import containers, operations

_MODULES = (containers, operations)

for _module in _MODULES:
    globals().update({name: getattr(_module, name) for name in _module.__all__})

__all__ = [name for module in _MODULES for name in module.__all__]

del _module
