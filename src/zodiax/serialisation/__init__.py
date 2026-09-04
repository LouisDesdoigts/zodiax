"""Validated Zodiax archives and automatically generated object definitions."""

from ._callables import register_callable
from .serialisation import ObjectDefinition, load, save

__all__ = ["ObjectDefinition", "save", "load", "register_callable"]
