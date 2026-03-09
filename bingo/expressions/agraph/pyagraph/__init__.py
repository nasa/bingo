"""Pure-Python AGraph expression implementation.

This sub-package contains the Python implementation of the acyclic-graph
expression, its evaluation engine, and algebraic simplification pipeline.
A future C++ implementation will live in a sibling package (e.g.
``cppagraph``).
"""

from .expression import AGraphExpression
from .data_container import DataContainer

__all__ = [
    "AGraphExpression",
    "DataContainer",
]
