"""C++ accelerated AGraph expression implementation.

This sub-package provides a C++17/Eigen implementation of the acyclic-graph
expression engine.  It exposes the same public API as the pure-Python sibling
package :mod:`~bingo.expressions.agraph.pyagraph`.

If the compiled extension ``_cppagraph`` is not available (e.g. the C++
build was skipped), importing this package will raise :exc:`ImportError`.
The parent package's ``__init__.py`` handles the fallback to pyagraph.
"""

# pylint: disable=import-error,no-name-in-module
from ._cppagraph import (  # type: ignore[import-not-found]
    # DataContainer class
    DataContainer,
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
    # Evaluation engine
    evaluate,
    evaluate_with_derivative,
    evaluate_with_const_hessian,
    CachedEvaluator,
    # Simplification / stack reduction
    get_utilized_commands,
    reduce,
    # CAS simplification pipeline
    cas_simplify,
    # Expression class
    AGraphExpression,
    # Scoring metrics
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    bic_score,
    laplace_nmll_score,
)

__all__ = [
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
    # Evaluation engine
    "evaluate",
    "evaluate_with_derivative",
    "evaluate_with_const_hessian",
    "CachedEvaluator",
    # Simplification / stack reduction
    "get_utilized_commands",
    "reduce",
    # CAS simplification pipeline
    "cas_simplify",
    # Expression class
    "AGraphExpression",
    # Scoring metrics
    "mean_absolute_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "bic_score",
    "laplace_nmll_score",
]
