"""Simplification subpackage for AGraph expressions.

Provides both cheap stack reduction (:func:`reduce`) and full algebraic
CAS simplification (:func:`simplify`).
"""

from ._reduce import get_utilized_commands, reduce
from ._simplify import simplify

__all__ = [
    "get_utilized_commands",
    "reduce",
    "simplify",
]
