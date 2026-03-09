"""Pure-Python AGraph expression implementation.

This sub-package contains the Python implementation of the acyclic-graph
expression, its evaluation engine, and algebraic simplification pipeline.
A future C++ implementation will live in a sibling package (e.g.
``cppagraph``).
"""

from .expression import AGraphExpression
from .data_container import DataContainer
from .operators import (  # noqa: F401 — re-exported for public use
    # Operator ID constants
    VARIABLE,
    CONSTANT,
    INTEGER,
    ADDITION,
    SUBTRACTION,
    MULTIPLICATION,
    DIVISION,
    POWER,
    SAFE_POWER,
    SQUARE,
    CUBE,
    SQRT,
    ABS,
    EXPONENTIAL,
    LOGARITHM,
    SIN,
    COS,
    TAN,
    ARCSIN,
    ARCCOS,
    ARCTAN,
    SINH,
    COSH,
    TANH,
    # Operator property sets
    TERMINAL_IDS,
    ARITY_2_IDS,
    # NumPy lookup arrays
    IS_TERMINAL_ARRAY,
    IS_ARITY_2_ARRAY,
    # Name mapping
    OPERATOR_NAMES,
)

__all__ = [
    "AGraphExpression",
    "DataContainer",
    # Operator ID constants
    "VARIABLE",
    "CONSTANT",
    "INTEGER",
    "ADDITION",
    "SUBTRACTION",
    "MULTIPLICATION",
    "DIVISION",
    "POWER",
    "SAFE_POWER",
    "SQUARE",
    "CUBE",
    "SQRT",
    "ABS",
    "EXPONENTIAL",
    "LOGARITHM",
    "SIN",
    "COS",
    "TAN",
    "ARCSIN",
    "ARCCOS",
    "ARCTAN",
    "SINH",
    "COSH",
    "TANH",
    # Operator property sets
    "TERMINAL_IDS",
    "ARITY_2_IDS",
    # NumPy lookup arrays
    "IS_TERMINAL_ARRAY",
    "IS_ARITY_2_ARRAY",
    # Name mapping
    "OPERATOR_NAMES",
]
