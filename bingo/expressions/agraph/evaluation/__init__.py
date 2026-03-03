"""Evaluation subpackage for AGraph expressions.

Re-exports the public API so callers can do::

    from bingo.expressions.agraph.evaluation import evaluate
"""

from .evaluation import evaluate, evaluate_with_derivative

__all__ = [
    "evaluate",
    "evaluate_with_derivative",
]
