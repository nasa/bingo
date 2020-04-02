from .operator_definitions import *
from .expression import Expression


NEGATIVE_ONE = Expression(INTEGER, [-1])
ZERO = Expression(INTEGER, [0])
ONE = Expression(INTEGER, [1])


def automatic_simplify(expression):
    print("!", expression)
    if expression.operator in [CONSTANT, INTEGER, CONSTSYMBOL, VARIABLE]:
        print("<", expression)
        return expression

    expr_w_simp_operands = expression.map(automatic_simplify)
    print("!s", expr_w_simp_operands)

    if expr_w_simp_operands.operator == POWER:
        return simplify_power(expr_w_simp_operands)

    if expr_w_simp_operands.operator == MULTIPLICATION:
        print("<", simplify_product(expr_w_simp_operands))
        return simplify_product(expr_w_simp_operands)

    if expr_w_simp_operands.operator == ADDITION:
        return simplify_sum(expr_w_simp_operands)

    if expr_w_simp_operands.operator == DIVISION:
        return simplify_quotient(expr_w_simp_operands)

    if expr_w_simp_operands.operator == SUBTRACTION:
        return simplify_difference(expr_w_simp_operands)

    raise NotImplementedError


def simplify_power(expression):
    base, exponent = expression.operands
    if base == ONE:
        return ONE.copy()
    if base == ZERO and exponent.operator == INTEGER \
            and exponent.operands[0] > 0:
        return ZERO.copy()
    if exponent.operator in [INTEGER, CONSTANT, CONSTSYMBOL]:
        return _simplify_constant_power(base, exponent)
    return expression


def _simplify_constant_power(base, exponent):
    if exponent == ONE:
        return base
    if exponent == ZERO:
        return ONE.copy()

    if base.operator == INTEGER and exponent.operator == INTEGER \
            and exponent.operands[0] > 0:
        return Expression(INTEGER, [base.operands[0]**exponent.operands[0]])

    if base.operator == POWER:  # multiply constant powers
        base_base = base.operands[0]
        base_exponent = base.operands[1]
        mult_exp = Expression(MULTIPLICATION, [base_exponent, exponent])
        new_exponent = simplify_product(mult_exp)
        if base_exponent in [INTEGER, CONSTANT, CONSTSYMBOL]:
            return _simplify_constant_power(base_base, new_exponent)
        return Expression(POWER, [base_base, new_exponent])

    if base.operator == MULTIPLICATION:  # distribute constant powers
        def temp_simp_const_power(bas):
            exp = exponent.copy()
            return _simplify_constant_power(bas, exp)
        return simplify_product(base.map(temp_simp_const_power))

    return Expression(POWER, [base, exponent])


def simplify_product(expression):
    operands = expression.operands
    if ZERO in operands:
        return ZERO.copy()
    if len(operands) == 1:
        return operands[0]

    recursively_simplified_operands = _simplify_product_rec(operands)
    if len(recursively_simplified_operands) == 0:
        return ONE.copy()
    if len(recursively_simplified_operands) == 1:
        return recursively_simplified_operands[0]
    return Expression(MULTIPLICATION, recursively_simplified_operands)


def _simplify_product_rec(operands):
    if len(operands) == 2:
        op_1, op_2 = operands
        if op_1.operator == INTEGER and op_2.operator == INTEGER:
            new_integer = op_1.operands[0] * op_2.operands[0]
            simpl_const_prod = Expression(INTEGER, [new_integer])
            if simpl_const_prod == ONE:
                return []
            return [simpl_const_prod]

        if op_1.operator != MULTIPLICATION and op_2.operator != MULTIPLICATION:
            if op_1 == ONE:
                return [op_2]
            if op_2 == ONE:
                return [op_1]

            if op_1.base == op_2.base:
                new_exponent = Expression(ADDITION,
                                          [op_1.exponent, op_2.exponent])
                new_exponent = simplify_sum(new_exponent)
                combined_op = Expression(POWER, [op_1.base, new_exponent])
                combined_op = simplify_power(combined_op)

                if combined_op == ONE:
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
    if len(operands_1) == 0:
        return operands_2
    if len(operands_2) == 0:
        return operands_1

    simplified_firsts = _simplify_product_rec([operands_1[0], operands_2[0]])
    if len(simplified_firsts) == 0:
        return _merge_products(operands_1[1:], operands_2[1:])
    if len(simplified_firsts) == 1:
        return simplified_firsts + _merge_products(operands_1[1:],
                                                   operands_2[1:])
    if simplified_firsts[0] == operands_1[0]:
        return [simplified_firsts[0]] + _merge_products(operands_1[1:],
                                                        operands_2)
    return [simplified_firsts[0]] + _merge_products(operands_1, operands_2[1:])


def simplify_sum(expression):
    operands = expression.operands
    if len(operands) == 1:
        return operands[0]

    recursively_simplified_operands = _simplify_sum_rec(operands)
    if len(recursively_simplified_operands) == 0:
        return ZERO.copy()
    if len(recursively_simplified_operands) == 1:
        return recursively_simplified_operands[0]
    return Expression(ADDITION, recursively_simplified_operands)


def _simplify_sum_rec(operands):
    if len(operands) == 2:
        op_1, op_2 = operands
        if op_1.operator == INTEGER and op_2.operator == INTEGER:
            new_integer = op_1.operands[0] + op_2.operands[0]
            simpl_const_sum = Expression(INTEGER, [new_integer])
            if simpl_const_sum == ZERO:
                return []
            return [simpl_const_sum]

        if op_1.operator != ADDITION and op_2.operator != ADDITION:
            if op_1 == ZERO:
                return [op_2]
            if op_2 == ZERO:
                return [op_1]

            if op_1.term == op_2.term:
                new_coefficient = Expression(ADDITION,
                                             [op_1.coefficient,
                                              op_2.coefficient])
                new_coefficient = simplify_sum(new_coefficient)
                combined_op = Expression(MULTIPLICATION,
                                         [new_coefficient, op_1.term])
                combined_op = simplify_product(combined_op)

                if combined_op == ZERO:
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
    if len(operands_1) == 0:
        return operands_2
    if len(operands_2) == 0:
        return operands_1

    simplified_firsts = _simplify_sum_rec([operands_1[0], operands_2[0]])
    if len(simplified_firsts) == 0:
        return _merge_sums(operands_1[1:], operands_2[1:])
    if len(simplified_firsts) == 1:
        return simplified_firsts + _merge_sums(operands_1[1:], operands_2[1:])
    if simplified_firsts[0] == operands_1[0]:
        return [simplified_firsts[0]] + _merge_sums(operands_1[1:], operands_2)
    return [simplified_firsts[0]] + _merge_sums(operands_1, operands_2[1:])


def simplify_quotient(expression):
    numerator, denominator = expression.operands
    denominator_inv = Expression(POWER, [denominator, NEGATIVE_ONE.copy()])
    denominator_inv = simplify_power(denominator_inv)
    quotient_as_product = Expression(MULTIPLICATION,
                                     [numerator, denominator_inv])
    return simplify_product(quotient_as_product)


def simplify_difference(expression):
    first, second = expression.operands
    negative_second = Expression(MULTIPLICATION, [NEGATIVE_ONE.copy(), second])
    negative_second = simplify_product(negative_second)
    difference_as_sum = Expression(ADDITION, [first, negative_second])
    return simplify_sum(difference_as_sum)
