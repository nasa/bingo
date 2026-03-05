"""Optional expression modifications applied after automatic simplification.

These are not strictly required for correctness but produce a canonical
form better suited to the AGraph stack representation.

Module-level flags control each modification:

``INSERT_SUBTRACTION``
    Convert ``a + (-1)*b`` → ``a - b``.  Default ``True``.

``INSERT_DIVISION``
    Convert ``a * b^(-1)`` → ``a / b`` and standalone ``b^(-1)`` →
    ``1 / b``.  Default ``True``.

``INSERT_SQUARE_CUBE``
    Convert ``x^2`` → ``SQUARE(x)`` and ``x^3`` → ``CUBE(x)`` so
    that integer-exponent powers use the compact unary operators.
    Default ``True``.

``REPLACE_INTEGER_POWERS``
    Convert ``a^n`` (positive integer *n* ≥ 4) into expanded
    multiplications.  Exponents 2 and 3 are handled by
    ``INSERT_SQUARE_CUBE`` when enabled.  Default ``True``.

``REPLACE_INTEGERS_WITH_CONSTANTS``
    Convert integer coefficients (2*x, 3*x, …) into constant-valued
    expressions so that they become optimisable.  Default ``False``.
"""

from ..operators import (
    INTEGER,
    CONSTANT,
    VARIABLE,
    ADDITION,
    MULTIPLICATION,
    SUBTRACTION,
    DIVISION,
    POWER,
    SQUARE,
    CUBE,
)
from .cas_expression import CASExpression, _NEG_ONE, _ONE

INSERT_SUBTRACTION = True
INSERT_DIVISION = True
INSERT_SQUARE_CUBE = True
REPLACE_INTEGER_POWERS = False
REPLACE_INTEGERS_WITH_CONSTANTS = False

NEGATIVE_ONE = _NEG_ONE
ONE_EXPR = _ONE
SOME_BIG_INT = 1_000_000
_TERMINAL_OPS = frozenset({INTEGER, CONSTANT, VARIABLE})


def optional_modifications(expression):
    """Apply optional post-simplification modifications.

    Parameters
    ----------
    expression : CASExpression

    Returns
    -------
    CASExpression
    """
    if INSERT_SUBTRACTION:
        expression = _insert_subtraction(expression)
    if INSERT_DIVISION:
        expression = _insert_division(expression)
    if INSERT_SQUARE_CUBE:
        expression = _insert_square_cube(expression)
    if REPLACE_INTEGER_POWERS:
        expression = _replace_integer_powers(expression)
    if REPLACE_INTEGERS_WITH_CONSTANTS:
        expression = _replace_integers_with_constants(expression)
    return expression


# ------------------------------------------------------------------ #
#  a + (-1)*b → a - b                                                 #
# ------------------------------------------------------------------ #


def _insert_subtraction(expression):
    operator = expression.operator
    if operator in _TERMINAL_OPS:
        return expression

    orig_operands = expression.operands
    operands_w_subtraction = [_insert_subtraction(operand) for operand in orig_operands]
    if operator != ADDITION:
        if all(n is o for n, o in zip(operands_w_subtraction, orig_operands)):
            return expression
        return CASExpression(operator, operands_w_subtraction)

    additive_operands = []
    subtractive_operands = []
    for operand in operands_w_subtraction:
        coeff = operand.coefficient
        if coeff is NEGATIVE_ONE or coeff == NEGATIVE_ONE:
            term = operand.term
            if len(term.operands) == 1:
                subtractive_operands.append(term.operands[0])
            else:
                subtractive_operands.append(term)
        else:
            additive_operands.append(operand)

    if len(subtractive_operands) == 0:
        return CASExpression(ADDITION, additive_operands)

    if len(additive_operands) == 0:
        return CASExpression(
            MULTIPLICATION,
            [
                NEGATIVE_ONE,
                CASExpression(ADDITION, subtractive_operands),
            ],
        )

    if len(subtractive_operands) == 1:
        subtractive_exp = subtractive_operands[0]
    else:
        subtractive_exp = CASExpression(ADDITION, subtractive_operands)

    if len(additive_operands) == 1:
        additive_exp = additive_operands[0]
    else:
        additive_exp = CASExpression(ADDITION, additive_operands)

    return CASExpression(SUBTRACTION, [additive_exp, subtractive_exp])


# ------------------------------------------------------------------ #
#  a * b^(-1) → a / b                                                 #
# ------------------------------------------------------------------ #


def _is_inverse(expr):
    """Return True if *expr* is ``something^(-1)``."""
    return (
        expr.operator == POWER
        and expr.operands[1].operator == INTEGER
        and expr.operands[1].operands[0] == -1
    )


def _insert_division(expression):
    operator = expression.operator
    if operator in _TERMINAL_OPS:
        return expression

    orig_operands = expression.operands

    # Standalone inverse: POWER(x, -1) → DIVISION(1, x)
    if operator == POWER and _is_inverse(expression):
        base = _insert_division(orig_operands[0])
        return CASExpression(DIVISION, [ONE_EXPR, base])

    if operator != MULTIPLICATION:
        new_operands = [_insert_division(op) for op in orig_operands]
        if all(n is o for n, o in zip(new_operands, orig_operands)):
            return expression
        return CASExpression(operator, new_operands)

    # Split product operands into numerator and denominator groups.
    # Check _is_inverse on ORIGINAL operands BEFORE recursing.
    numerator_ops = []
    denominator_ops = []
    for operand in orig_operands:
        if _is_inverse(operand):
            # Recurse on the base only (strip the ^(-1))
            denominator_ops.append(_insert_division(operand.operands[0]))
        else:
            numerator_ops.append(_insert_division(operand))

    if not denominator_ops:
        if all(n is o for n, o in zip(numerator_ops, orig_operands)):
            return expression
        return CASExpression(MULTIPLICATION, numerator_ops)

    # Build numerator
    if len(numerator_ops) == 0:
        numerator = ONE_EXPR
    elif len(numerator_ops) == 1:
        numerator = numerator_ops[0]
    else:
        numerator = CASExpression(MULTIPLICATION, numerator_ops)

    # Build denominator
    if len(denominator_ops) == 1:
        denominator = denominator_ops[0]
    else:
        denominator = CASExpression(MULTIPLICATION, denominator_ops)

    return CASExpression(DIVISION, [numerator, denominator])


# ------------------------------------------------------------------ #
#  x^2 → SQUARE(x),  x^3 → CUBE(x)                                   #
# ------------------------------------------------------------------ #


def _insert_square_cube(expression):
    operator = expression.operator
    if operator in _TERMINAL_OPS:
        return expression

    orig_operands = expression.operands
    new_operands = [_insert_square_cube(op) for op in orig_operands]

    if operator == POWER and new_operands[1].operator == INTEGER:
        exp_val = new_operands[1].operands[0]
        if exp_val == 2:
            return CASExpression(SQUARE, [new_operands[0]])
        if exp_val == 3:
            return CASExpression(CUBE, [new_operands[0]])

    if all(n is o for n, o in zip(new_operands, orig_operands)):
        return expression
    return CASExpression(operator, new_operands)


# ------------------------------------------------------------------ #
#  a^n → expanded multiplication  (n ≥ 4 only when SQUARE/CUBE       #
#  insertion has already handled 2 and 3)                              #
# ------------------------------------------------------------------ #


def _replace_integer_powers(expression):
    operator = expression.operator
    if operator in _TERMINAL_OPS:
        return expression

    orig_operands = expression.operands
    operands_w_replaced = [
        _replace_integer_powers(operand) for operand in orig_operands
    ]

    if (
        operator != POWER
        or operands_w_replaced[1].operator != INTEGER
        or operands_w_replaced[1].operands[0] <= 0
    ):
        if all(n is o for n, o in zip(operands_w_replaced, orig_operands)):
            return expression
        return CASExpression(operator, operands_w_replaced)

    power = operands_w_replaced[1].operands[0]
    base = operands_w_replaced[0]

    return CASExpression(MULTIPLICATION, [base] * power)


# ------------------------------------------------------------------ #
#  integer coefficients → constants                                   #
# ------------------------------------------------------------------ #


def _replace_integers_with_constants(expression):
    operator = expression.operator
    if operator in [CONSTANT, VARIABLE]:
        return expression
    if operator == INTEGER:
        return CASExpression(CONSTANT, [SOME_BIG_INT + expression.operands[0]])

    orig_operands = expression.operands
    operands_w_replaced = [
        _replace_integers_with_constants(operand) for operand in orig_operands
    ]
    if all(n is o for n, o in zip(operands_w_replaced, orig_operands)):
        return expression
    return CASExpression(operator, operands_w_replaced)
