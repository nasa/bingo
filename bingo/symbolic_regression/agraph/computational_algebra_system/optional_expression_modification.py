from ..operator_definitions import *
from .expression import Expression

INSERT_SUBTRACTION = True
REPLACE_INTEGER_POWERS = True

NEGATIVE_ONE = Expression(INTEGER, [-1])


def optional_modifications(expression):
    if INSERT_SUBTRACTION:
        expression = _insert_subtraction(expression)
    if REPLACE_INTEGER_POWERS:
        expression = _replace_integer_powers(expression)
    return expression


def _insert_subtraction(expression):
    operator = expression.operator
    if operator in [INTEGER, CONSTANT, VARIABLE]:
        return expression

    operands_w_subtraction = [_insert_subtraction(operand)
                              for operand in expression.operands]
    if operator != ADDITION:
        return Expression(operator, operands_w_subtraction)

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
        return Expression(ADDITION, additive_operands)

    if len(additive_operands) == 0:
        return Expression(MULTIPLICATION, [NEGATIVE_ONE.copy(),
                                           Expression(ADDITION,
                                                      subtractive_operands)])

    return Expression(SUBTRACTION,
                      [Expression(ADDITION, additive_operands),
                       Expression(ADDITION, subtractive_operands)])


def _replace_integer_powers(expression):
    operator = expression.operator
    if operator in [INTEGER, CONSTANT, VARIABLE]:
        return expression

    operands_w_replaced = [_replace_integer_powers(operand)
                           for operand in expression.operands]

    if operator != POWER:
        return Expression(operator, operands_w_replaced)

    if operands_w_replaced[1].operator != INTEGER:
        return Expression(operator, operands_w_replaced)

    power = operands_w_replaced[1].operands[0]
    return Expression(MULTIPLICATION, [operands_w_replaced[0]] * power)


