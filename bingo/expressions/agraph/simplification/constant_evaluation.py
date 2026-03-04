"""Numeric evaluation and collapse of constant-valued CAS subtrees.

After constant folding has merged constant indices structurally, this
module walks the CAS tree and *numerically evaluates* any subtree whose
leaves are exclusively CONSTANT and INTEGER nodes.  The result replaces
the subtree with a single ``CONSTANT`` node whose value is the computed
float.

If the evaluation produces a non-finite result (NaN / inf), the subtree
is replaced with one of the existing CONSTANT terminals from that
subtree so that the expression still collapses — no information about
the tree shape is preserved, matching the old behaviour.

Evaluation semantics (protected operators, etc.) mirror
:mod:`bingo.expressions.agraph.evaluation.operator_eval` exactly.
"""

import math

import numpy as np

from ..operators import (
    CONSTANT,
    INTEGER,
    VARIABLE,
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
)
from .cas_expression import CASExpression


# ------------------------------------------------------------------ #
#  Scalar evaluation map — mirrors operator_eval.py protections       #
# ------------------------------------------------------------------ #


def _safe_sqrt(x):
    return math.sqrt(abs(x))


def _safe_log(x):
    v = abs(x)
    if v == 0.0:
        return float("-inf")
    return math.log(v)


def _safe_power(base, exp):
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.power(base, exp))


def _safe_safe_power(base, exp):
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.power(abs(base), exp))


_UNARY_EVAL = {
    SQUARE: lambda x: x * x,
    CUBE: lambda x: x * x * x,
    SQRT: _safe_sqrt,
    ABS: abs,
    EXPONENTIAL: math.exp,
    LOGARITHM: _safe_log,
    SIN: math.sin,
    COS: math.cos,
    TAN: math.tan,
    ARCSIN: math.asin,
    ARCCOS: math.acos,
    ARCTAN: math.atan,
    SINH: math.sinh,
    COSH: math.cosh,
    TANH: math.tanh,
}

_BINARY_EVAL = {
    ADDITION: lambda a, b: a + b,
    SUBTRACTION: lambda a, b: a - b,
    MULTIPLICATION: lambda a, b: a * b,
    DIVISION: lambda a, b: (
        a / b
        if b != 0.0
        else (math.copysign(math.inf, a) if a != 0.0 else float("nan"))
    ),
    POWER: _safe_power,
    SAFE_POWER: _safe_safe_power,
}


# ------------------------------------------------------------------ #
#  Tree evaluation                                                    #
# ------------------------------------------------------------------ #


def _find_first_constant(expression):
    """Return the first CONSTANT leaf found in *expression* (DFS)."""
    if expression.operator == CONSTANT:
        return expression
    if expression.operator in (INTEGER, VARIABLE):
        return None
    for operand in expression.operands:
        found = _find_first_constant(operand)
        if found is not None:
            return found
    return None


def count_constant_leaves(expression):
    """Return the number of distinct CONSTANT leaf indices in *expression*."""
    indices = set()
    _collect_constant_indices(expression, indices)
    return len(indices)


def _collect_constant_indices(expression, indices):
    if expression.operator == CONSTANT:
        indices.add(expression.operands[0])
        return
    if expression.operator in (INTEGER, VARIABLE):
        return
    for operand in expression.operands:
        _collect_constant_indices(operand, indices)


def _evaluate_subtree(expression, constants):
    """Recursively compute the scalar float value of a constant subtree.

    Parameters
    ----------
    expression : CASExpression
        Must be ``is_constant_valued``.
    constants : tuple of float
        The current constant-value table.

    Returns
    -------
    float
        The numeric value (may be inf / nan on degenerate inputs).
    """
    op = expression.operator

    if op == CONSTANT:
        idx = expression.operands[0]
        return float(constants[idx]) if idx < len(constants) else 1.0

    if op == INTEGER:
        return float(expression.operands[0])

    operands = expression.operands

    # Unary
    fn_u = _UNARY_EVAL.get(op)
    if fn_u is not None:
        child_val = _evaluate_subtree(operands[0], constants)
        if not math.isfinite(child_val):
            return child_val
        try:
            return fn_u(child_val)
        except (ValueError, OverflowError, ZeroDivisionError):
            return float("nan")

    # Binary
    fn_b = _BINARY_EVAL.get(op)
    if fn_b is not None:
        left = _evaluate_subtree(operands[0], constants)
        right = _evaluate_subtree(operands[1], constants)
        try:
            return fn_b(left, right)
        except (ValueError, OverflowError, ZeroDivisionError):
            return float("nan")

    # N-ary (flattened ADDITION / MULTIPLICATION from CAS)
    if op == ADDITION:
        total = 0.0
        for child in operands:
            total += _evaluate_subtree(child, constants)
        return total

    if op == MULTIPLICATION:
        total = 1.0
        for child in operands:
            total *= _evaluate_subtree(child, constants)
        return total

    # Unknown operator — shouldn't happen, but be safe
    return float("nan")


# ------------------------------------------------------------------ #
#  Public API                                                         #
# ------------------------------------------------------------------ #


def evaluate_constant_subtrees(expression, constants, integers=None):
    """Replace constant-valued subtrees with single CONSTANT nodes.

    Walks the CAS tree; when a non-terminal subtree is entirely
    ``is_constant_valued``, its numeric value is computed and the
    subtree is replaced by ``CASExpression(CONSTANT, [new_index])``
    with the computed value appended to a growing constants list.

    If evaluation produces a non-finite result (NaN / inf), the subtree
    is replaced with one of the existing CONSTANT terminals from that
    subtree instead.

    Parameters
    ----------
    expression : CASExpression
    constants : tuple of float
    integers : tuple of int, optional
        Not used directly (INTEGER values are embedded in the CAS tree)
        but accepted for pipeline uniformity.

    Returns
    -------
    tuple
        ``(new_expression, new_constants)`` where *new_constants* is
        the original tuple extended with any computed values.
    """
    const_list = list(constants)
    result = _collapse_recursive(expression, const_list)
    return result, tuple(const_list)


def _has_constant_leaf(expression):
    """Return True if *expression* contains at least one CONSTANT leaf."""
    if expression.operator == CONSTANT:
        return True
    if expression.operator in (INTEGER, VARIABLE):
        return False
    return any(_has_constant_leaf(op) for op in expression.operands)


def _collapse_recursive(expression, const_list):
    """Bottom-up walk: collapse constant-valued subtrees."""
    op = expression.operator

    # Terminals are already as simple as possible.
    if op in (CONSTANT, INTEGER, VARIABLE):
        return expression

    # Recurse into children first (bottom-up).
    orig_operands = expression.operands
    new_operands = [_collapse_recursive(child, const_list) for child in orig_operands]

    # Rebuild if any child changed.
    if all(n is o for n, o in zip(new_operands, orig_operands)):
        node = expression
    else:
        node = CASExpression(op, new_operands)

    # If this node is constant-valued and not already a bare terminal,
    # evaluate it numerically and collapse — but only when the subtree
    # contains at least one CONSTANT leaf.  Pure-integer subtrees like
    # sin(INTEGER(1)) are left as-is to mirror the old implementation.
    if node.is_constant_valued and _has_constant_leaf(node):
        value = _evaluate_subtree(node, tuple(const_list))
        if math.isfinite(value):
            new_idx = len(const_list)
            const_list.append(value)
            return CASExpression(CONSTANT, [new_idx])
        # Non-finite — replace with an existing CONSTANT terminal
        # from the subtree.
        fallback = _find_first_constant(node)
        if fallback is not None:
            return fallback

    return node
