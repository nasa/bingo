"""Optional expression modifications applied after automatic simplification.

These are not strictly required for correctness but produce a canonical
form better suited to the AGraph stack representation.

Module-level flags control each modification:

``INSERT_SUBTRACTION``
    Convert ``a + (-1)*b`` → ``a - b``.  Default ``True``.

``REPLACE_INTEGER_POWERS``
    Convert ``a^n`` (small positive integer *n*) into expanded
    multiplications / SQUARE / CUBE operators.  Default ``True``.

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
    POWER,
    SQUARE,
    CUBE,
)
from .cas_expression import CASExpression

INSERT_SUBTRACTION = True
REPLACE_INTEGER_POWERS = True
REPLACE_INTEGERS_WITH_CONSTANTS = False

NEGATIVE_ONE = CASExpression(INTEGER, [-1])
SOME_BIG_INT = 1_000_000


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
    if operator in [INTEGER, CONSTANT, VARIABLE]:
        return expression

    operands_w_subtraction = [
        _insert_subtraction(operand) for operand in expression.operands
    ]
    if operator != ADDITION:
        return CASExpression(operator, operands_w_subtraction)

    additive_operands = []
    subtractive_operands = []
    for operand in operands_w_subtraction:
        if operand.coefficient == NEGATIVE_ONE:
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
                NEGATIVE_ONE.copy(),
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
#  a^n → SQUARE / CUBE / expanded multiplication                     #
# ------------------------------------------------------------------ #


def _replace_integer_powers(expression):
    operator = expression.operator
    if operator in [INTEGER, CONSTANT, VARIABLE]:
        return expression

    operands_w_replaced = [
        _replace_integer_powers(operand) for operand in expression.operands
    ]

    if (
        operator != POWER
        or operands_w_replaced[1].operator != INTEGER
        or operands_w_replaced[1].operands[0] <= 0
    ):
        return CASExpression(operator, operands_w_replaced)

    power = operands_w_replaced[1].operands[0]
    base = operands_w_replaced[0]

    if power == 2:
        return CASExpression(SQUARE, [base, base])
    if power == 3:
        return CASExpression(CUBE, [base, base])

    return CASExpression(MULTIPLICATION, [base] * power)


# ------------------------------------------------------------------ #
#  integer coefficients → constants                                   #
# ------------------------------------------------------------------ #


def _replace_integers_with_constants(expression):
    operator = expression.operator
    if operator in [CONSTANT, VARIABLE]:
        return expression
    if operator == INTEGER:
        return CASExpression(
            CONSTANT, [SOME_BIG_INT + expression.operands[0]]
        )

    operands_w_replaced = [
        _replace_integers_with_constants(operand)
        for operand in expression.operands
    ]
    return CASExpression(operator, operands_w_replaced)
