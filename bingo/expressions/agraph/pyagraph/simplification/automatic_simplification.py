"""Core of the built-in computer algebra simplification algorithm.

Based on the algorithm presented in Chapter 3 of Joel Cohen's book [1].

References
----------
.. [1] Joel S. Cohen (2003) *Computer Algebra and Symbolic Computation*.
"""

from ..operators import (
    CONSTANT,
    INTEGER,
    VARIABLE,
    POWER,
    MULTIPLICATION,
    ADDITION,
    SUBTRACTION,
    DIVISION,
    SIN,
    COS,
    TAN,
    LOGARITHM,
    EXPONENTIAL,
    ABS,
    SQRT,
    SAFE_POWER,
    SINH,
    COSH,
    TANH,
    ARCCOS,
    ARCSIN,
    ARCTAN,
    SQUARE,
    CUBE,
)
from .cas_expression import (
    CASExpression,
    _init_singletons,
    _ZERO,
    _ONE,
    _TWO,
    _NEG_ONE,
)

# Ensure singletons are initialised before use.
_init_singletons()

# Module-level aliases for readability.
NEGATIVE_ONE = _NEG_ONE
ZERO = _ZERO
ONE = _ONE


def automatic_simplify(expression):
    """Recursively simplify a CAS expression.

    Parameters
    ----------
    expression : CASExpression

    Returns
    -------
    CASExpression
    """
    if expression.operator in [CONSTANT, INTEGER, VARIABLE]:
        return expression

    expr_w_simp_operands = expression.map(automatic_simplify)

    return SIMPLIFICATION_FUNCTIONS[expr_w_simp_operands.operator](expr_w_simp_operands)


# ------------------------------------------------------------------ #
#  Power                                                              #
# ------------------------------------------------------------------ #


def simplify_power(expression):
    """Simplification of power operators."""
    base, exponent = expression.operands
    if base.is_one():
        return ONE
    if base.is_zero() and exponent.operator == INTEGER and exponent.operands[0] > 0:
        return ZERO
    if exponent.operator in [INTEGER, CONSTANT]:
        return _simplify_constant_power(base, exponent)
    return expression


def _simplify_constant_power(base, exponent):
    if exponent.is_one():
        return base
    if exponent.is_zero():
        return ONE

    if (
        base.operator == INTEGER
        and exponent.operator == INTEGER
        and exponent.operands[0] > 0
    ):
        return CASExpression(INTEGER, [base.operands[0] ** exponent.operands[0]])

    if base.operator == POWER:  # multiply constant powers
        base_base = base.operands[0]
        base_exponent = base.operands[1]
        mult_exp = CASExpression(MULTIPLICATION, [base_exponent, exponent])
        new_exponent = simplify_product(mult_exp)
        if base_exponent.operator in [INTEGER, CONSTANT]:
            return _simplify_constant_power(base_base, new_exponent)
        return CASExpression(POWER, [base_base, new_exponent])

    if base.operator == MULTIPLICATION:  # distribute constant powers

        def _temp_simp_const_power(bas):
            return _simplify_constant_power(bas, exponent)

        return simplify_product(base.map(_temp_simp_const_power))

    return CASExpression(POWER, [base, exponent])


# ------------------------------------------------------------------ #
#  Product                                                            #
# ------------------------------------------------------------------ #


def simplify_product(expression):
    """Simplification of multiplication operators."""
    operands = expression.operands
    if ZERO in operands:
        return ZERO
    if len(operands) == 1:
        return operands[0]

    recursively_simplified_operands = _simplify_product_rec(operands)
    if len(recursively_simplified_operands) == 0:
        return ONE
    if len(recursively_simplified_operands) == 1:
        return recursively_simplified_operands[0]
    return CASExpression(MULTIPLICATION, recursively_simplified_operands)


def _simplify_product_rec(operands):
    if len(operands) == 2:
        op_1, op_2 = operands
        if op_1.operator == INTEGER and op_2.operator == INTEGER:
            new_integer = op_1.operands[0] * op_2.operands[0]
            simpl_const_prod = CASExpression(INTEGER, [new_integer])
            if simpl_const_prod.is_one():
                return []
            return [simpl_const_prod]

        if MULTIPLICATION not in (op_1.operator, op_2.operator):
            if op_1.is_one():
                return [op_2]
            if op_2.is_one():
                return [op_1]

            if op_1.base == op_2.base:
                e1, e2 = op_1.exponent, op_2.exponent
                # Fast path: 1+1=2 (very common — two identical terms)
                if e1 is _ONE and e2 is _ONE:
                    new_exponent = _TWO
                else:
                    new_exponent = CASExpression(ADDITION, [e1, e2])
                    new_exponent = simplify_sum(new_exponent)
                combined_op = CASExpression(POWER, [op_1.base, new_exponent])
                combined_op = simplify_power(combined_op)

                if combined_op.is_one():
                    return []
                return [combined_op]

            if op_2 < op_1:
                return [op_2, op_1]

            return operands

        if op_1.operator == MULTIPLICATION:
            to_merge_1 = op_1.operands
        else:
            to_merge_1 = [op_1]
        if op_2.operator == MULTIPLICATION:
            to_merge_2 = op_2.operands
        else:
            to_merge_2 = [op_2]
        return _merge_products(to_merge_1, to_merge_2)

    rest_simplified = _simplify_product_rec(operands[1:])
    if operands[0].operator == MULTIPLICATION:
        return _merge_products(operands[0].operands, rest_simplified)
    return _merge_products([operands[0]], rest_simplified)


def _merge_products(operands_1, operands_2):
    result = []
    i, j = 0, 0
    n1, n2 = len(operands_1), len(operands_2)
    while i < n1 and j < n2:
        simplified = _simplify_product_rec([operands_1[i], operands_2[j]])
        slen = len(simplified)
        if slen == 0:
            i += 1
            j += 1
        elif slen == 1:
            result.append(simplified[0])
            i += 1
            j += 1
        elif simplified[0] is operands_1[i] or simplified[0] == operands_1[i]:
            result.append(simplified[0])
            i += 1
        else:
            result.append(simplified[0])
            j += 1
    # Append remaining
    if i < n1:
        result.extend(operands_1[i:])
    if j < n2:
        result.extend(operands_2[j:])
    return result


# ------------------------------------------------------------------ #
#  Sum                                                                #
# ------------------------------------------------------------------ #


def simplify_sum(expression):
    """Simplification of addition operators."""
    operands = expression.operands
    if len(operands) == 1:
        return operands[0]

    recursively_simplified_operands = _simplify_sum_rec(operands)
    if len(recursively_simplified_operands) == 0:
        return ZERO
    if len(recursively_simplified_operands) == 1:
        return recursively_simplified_operands[0]
    return CASExpression(ADDITION, recursively_simplified_operands)


def _simplify_sum_rec(operands):
    if len(operands) == 2:
        op_1, op_2 = operands
        if op_1.operator == INTEGER and op_2.operator == INTEGER:
            new_integer = op_1.operands[0] + op_2.operands[0]
            simpl_const_sum = CASExpression(INTEGER, [new_integer])
            if simpl_const_sum.is_zero():
                return []
            return [simpl_const_sum]

        if ADDITION not in (op_1.operator, op_2.operator):
            if op_1.is_zero():
                return [op_2]
            if op_2.is_zero():
                return [op_1]

            if op_1.same_term(op_2):
                c1, c2 = op_1.coefficient, op_2.coefficient
                # Fast path: 1+1=2 (very common — two identical terms)
                if c1 is _ONE and c2 is _ONE:
                    new_coefficient = _TWO
                else:
                    new_coefficient = CASExpression(ADDITION, [c1, c2])
                    new_coefficient = simplify_sum(new_coefficient)
                combined_op = CASExpression(
                    MULTIPLICATION, [new_coefficient, op_1.term]
                )
                combined_op = simplify_product(combined_op)

                if combined_op.is_zero():
                    return []
                return [combined_op]

            if op_2 < op_1:
                return [op_2, op_1]

            return operands

        if op_1.operator == ADDITION:
            to_merge_1 = op_1.operands
        else:
            to_merge_1 = [op_1]
        if op_2.operator == ADDITION:
            to_merge_2 = op_2.operands
        else:
            to_merge_2 = [op_2]
        return _merge_sums(to_merge_1, to_merge_2)

    rest_simplified = _simplify_sum_rec(operands[1:])
    if operands[0].operator == ADDITION:
        return _merge_sums(operands[0].operands, rest_simplified)
    return _merge_sums([operands[0]], rest_simplified)


def _merge_sums(operands_1, operands_2):
    result = []
    i, j = 0, 0
    n1, n2 = len(operands_1), len(operands_2)
    while i < n1 and j < n2:
        simplified = _simplify_sum_rec([operands_1[i], operands_2[j]])
        slen = len(simplified)
        if slen == 0:
            i += 1
            j += 1
        elif slen == 1:
            result.append(simplified[0])
            i += 1
            j += 1
        elif simplified[0] is operands_1[i] or simplified[0] == operands_1[i]:
            result.append(simplified[0])
            i += 1
        else:
            result.append(simplified[0])
            j += 1
    # Append remaining
    if i < n1:
        result.extend(operands_1[i:])
    if j < n2:
        result.extend(operands_2[j:])
    return result


# ------------------------------------------------------------------ #
#  Quotient and Difference                                            #
# ------------------------------------------------------------------ #


def simplify_quotient(expression):
    """Simplification of division operators."""
    numerator, denominator = expression.operands
    denominator_inv = CASExpression(POWER, [denominator, NEGATIVE_ONE])
    denominator_inv = simplify_power(denominator_inv)
    quotient_as_product = CASExpression(MULTIPLICATION, [numerator, denominator_inv])
    return simplify_product(quotient_as_product)


def simplify_difference(expression):
    """Simplification of subtraction operators."""
    first, second = expression.operands
    new_operands = [first]
    if second.operator == ADDITION:
        for operand in second.operands:
            negative_operand = CASExpression(MULTIPLICATION, [NEGATIVE_ONE, operand])
            new_operands.append(simplify_product(negative_operand))
    else:
        negative_second = CASExpression(MULTIPLICATION, [NEGATIVE_ONE, second])
        new_operands.append(simplify_product(negative_second))
    difference_as_sum = CASExpression(ADDITION, new_operands)
    return simplify_sum(difference_as_sum)


# ------------------------------------------------------------------ #
#  Trigonometric                                                      #
# ------------------------------------------------------------------ #


def simplify_sin(expression):
    """Simplification of sin operators."""
    operand = expression.operands[0]
    if operand.is_zero():
        return ZERO
    if operand.operator == ARCSIN:
        return operand.operands[0]
    return expression


def simplify_cos(expression):
    """Simplification of cos operators."""
    operand = expression.operands[0]
    if operand.is_zero():
        return ONE
    if operand.operator == ARCCOS:
        return operand.operands[0]
    return expression


def simplify_tan(expression):
    """Simplification of tan operators."""
    operand = expression.operands[0]
    if operand.is_zero():
        return ZERO
    if operand.operator == ARCTAN:
        return operand.operands[0]
    return expression


def simplify_logarithm(expression):
    """Simplification of log operators."""
    operand = expression.operands[0]
    if operand.is_one():
        return ZERO
    if operand.operator == EXPONENTIAL:
        return operand.operands[0]
    return expression


def simplify_exponential(expression):
    """Simplification of exp operators."""
    operand = expression.operands[0]
    if operand.is_zero():
        return ONE
    if operand.operator == LOGARITHM:
        return operand.operands[0]
    return expression


# ------------------------------------------------------------------ #
#  Hyperbolic                                                         #
# ------------------------------------------------------------------ #


def simplify_sinh(expression):
    """Simplification of hyperbolic sin operators."""
    if expression.operands[0].is_zero():
        return ZERO
    return expression


def simplify_cosh(expression):
    """Simplification of hyperbolic cos operators."""
    if expression.operands[0].is_zero():
        return ONE
    return expression


def simplify_tanh(expression):
    """Simplification of hyperbolic tan operators."""
    if expression.operands[0].is_zero():
        return ZERO
    return expression


# ------------------------------------------------------------------ #
#  Inverse trigonometric                                              #
# ------------------------------------------------------------------ #


def simplify_asin(expression):
    """Simplification of inverse sin operators."""
    if expression.operands[0].is_zero():
        return ZERO
    return expression


def simplify_acos(expression):
    """Simplification of inverse cos operators."""
    if expression.operands[0].is_one():
        return ZERO
    return expression


def simplify_atan(expression):
    """Simplification of inverse tan operators."""
    if expression.operands[0].is_zero():
        return ZERO
    return expression


# ------------------------------------------------------------------ #
#  SQRT / ABS                                                         #
# ------------------------------------------------------------------ #


def simplify_sqrt(expression):
    """Simplification of sqrt operators."""
    operand = expression.operands[0]
    if operand.is_zero():
        return ZERO
    if operand.is_one():
        return ONE
    # sqrt(x^2) → abs(x)  (common in symbolic regression)
    if (
        operand.operator == POWER
        and operand.operands[1].operator == INTEGER
        and operand.operands[1].operands[0] == 2
    ):
        return CASExpression(ABS, [operand.operands[0]])
    return expression


def simplify_abs(expression):
    """Simplification of absolute-value operators."""
    operand = expression.operands[0]
    if operand.is_zero():
        return ZERO
    if operand.is_one():
        return ONE
    if operand.operator == INTEGER:
        return CASExpression(INTEGER, [abs(operand.operands[0])])
    if operand.operator == ABS:
        return expression.operands[0]  # abs(abs(x)) → abs(x)
    return expression


# ------------------------------------------------------------------ #
#  Square / Cube  (kept for stacks that still contain these opcodes)  #
# ------------------------------------------------------------------ #


def simplify_square(expression):
    """Simplification of square operators.

    Assumes non-negative domain for ``sqrt(x)^2 → x`` (common in
    symbolic regression).
    """
    operand = expression.operands[0]
    if operand.is_zero():
        return ZERO
    if operand.is_one():
        return ONE
    if operand.operator == INTEGER:
        value = operand.operands[0]
        return CASExpression(INTEGER, [value**2])
    if operand.operator == SQRT:
        return operand.operands[0]
    return expression


def simplify_cube(expression):
    """Simplification of cube operators."""
    operand = expression.operands[0]
    if operand.is_zero():
        return ZERO
    if operand.is_one():
        return ONE
    if operand.operator == INTEGER:
        value = operand.operands[0]
        return CASExpression(INTEGER, [value**3])
    return expression


# ------------------------------------------------------------------ #
#  No-op                                                              #
# ------------------------------------------------------------------ #


def no_simplification(expression):
    """No simplification performed."""
    return expression


# ------------------------------------------------------------------ #
#  Dispatch table                                                     #
# ------------------------------------------------------------------ #

SIMPLIFICATION_FUNCTIONS = {
    POWER: simplify_power,
    MULTIPLICATION: simplify_product,
    ADDITION: simplify_sum,
    DIVISION: simplify_quotient,
    SUBTRACTION: simplify_difference,
    SIN: simplify_sin,
    COS: simplify_cos,
    TAN: simplify_tan,
    LOGARITHM: simplify_logarithm,
    EXPONENTIAL: simplify_exponential,
    ABS: simplify_abs,
    SQRT: simplify_sqrt,
    SAFE_POWER: simplify_power,
    SINH: simplify_sinh,
    COSH: simplify_cosh,
    TANH: simplify_tanh,
    ARCSIN: simplify_asin,
    ARCCOS: simplify_acos,
    ARCTAN: simplify_atan,
    SQUARE: simplify_square,
    CUBE: simplify_cube,
}
