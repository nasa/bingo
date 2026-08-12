"""Evaluation subpackage for AGraph expressions.

Re-exports the public API so callers can do::

    from bingo.expressions.agraph.pyagraph.evaluation import evaluate
"""

from .evaluation import (
    evaluate,
    evaluate_with_const_hessian,
    evaluate_with_derivative,
)
from .cached_evaluation import CachedEvaluator

__all__ = [
    "evaluate",
    "evaluate_with_const_hessian",
    "evaluate_with_derivative",
    "CachedEvaluator",
]
