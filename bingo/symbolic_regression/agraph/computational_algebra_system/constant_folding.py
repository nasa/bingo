from collections import defaultdict
from itertools import combinations

from .operator_definitions import *
from .expression import Expression


def fold_constants(expression):
    expression = _group_constants(expression)
    print("grouping constants: ", expression)

    check_for_folding = True
    while check_for_folding:
        print("iter", expression)
        check_for_folding = False
        constants = _get_constants(expression)
        for const_subset in _subsets(list(constants)):
            print(const_subset)
            insertion_points = _find_insertion_points(expression, const_subset)
            # print(insertion_points)
            replacements = _generate_replacement_instructions(const_subset,
                                                              constants,
                                                              insertion_points)
            # print(replacements)
            if len(replacements) > 0:
                expression = _perform_constant_folding(expression,
                                                       replacements)
                check_for_folding = True
                break

    return expression


def _group_constants(expression):
    if expression.operator in [CONSTANT, CONSTSYMBOL, INTEGER, VARIABLE]:
        return expression

    new_operands = [_group_constants(operand)
                    for operand in expression.operands]

    if expression.operator in [MULTIPLICATION, ADDITION]:
        const_operands = [operand for operand in new_operands
                          if operand.is_constant_valued]
        non_const_operands = [operand for operand in new_operands
                              if not operand.is_constant_valued]
        if len(const_operands) > 1 and len(non_const_operands) > 0:
            const_expr = Expression(expression.operator, const_operands)
            return Expression(expression.operator,
                              [const_expr] + non_const_operands)

    return Expression(expression.operator, new_operands)


def _generate_replacement_instructions(const_subset, constants,
                                       insertion_points):
    if len(insertion_points) > len(const_subset):
        return {}

    replacements = defaultdict(dict)
    constants_to_insert = set()
    expressions_to_replace = set()
    for const_num, (_, insertions) in zip(const_subset,
                                          insertion_points.items()):
        const_to_insert = constants[const_num]
        for (parent, children) in insertions:
            for i, child in enumerate(children):
                expressions_to_replace.add(child)
                if i == 0:
                    replacements[parent][child] = const_to_insert
                    constants_to_insert.add(const_to_insert)
                else:
                    replacements[parent][child] = None
                    constants_to_insert.add(None)

    if constants_to_insert == expressions_to_replace:
        return {}
    return replacements


def _get_constants(expression):
    if expression.operator == CONSTANT:
        return {expression.operands[0]: expression}

    if expression.operator in [CONSTSYMBOL, INTEGER, VARIABLE]:
        return {}

    constants = {}
    for operand in expression.operands:
        constants.update(_get_constants(operand))
    return constants


def _subsets(constants):
    for i in range(1, len(constants) + 1):
        for c in combinations(constants, i):
            yield set(c)


def _find_insertion_points(expression, constants):
    insertion_points = defaultdict(set)
    entire_expr_is_const = \
        _recursive_insertion_point_search(expression, constants,
                                          insertion_points, parent=None)
    if entire_expr_is_const:
        insertion_points[None].add((None, frozenset([expression])))
    return insertion_points


def _recursive_insertion_point_search(expression, constants, insertion_points,
                                      parent):
    if expression.operator == CONSTANT:
        return expression.operands[0] in constants

    if expression.operator in [CONSTSYMBOL, INTEGER, VARIABLE]:
        return False

    operands_are_constant_based = []
    for operand in expression.operands:
        is_constant_based = \
            _recursive_insertion_point_search(operand, constants,
                                              insertion_points, expression)
        operands_are_constant_based.append(is_constant_based)

    if all(operands_are_constant_based):
        return True

    if not any(operands_are_constant_based):
        return False

    if expression.is_constant_valued:
        insertion_points[expression].add((parent, frozenset([expression])))
    else:
        constant_operands = \
            frozenset([operand for operand, is_const
                       in zip(expression.operands, operands_are_constant_based)
                       if is_const])
        insertion_points[expression].add((expression, constant_operands))

    return False


def _perform_constant_folding(expression, replacements):
    if None in replacements:
        return replacements[None][expression].copy()

    return _recursive_expreson_replacement(expression, replacements)


def _recursive_expreson_replacement(expression, replacements):
    if expression not in replacements:
        if expression.operator in [CONSTANT, CONSTSYMBOL, INTEGER, VARIABLE]:
            return expression
        return expression.map(lambda x:
                              _perform_constant_folding(x, replacements))
    new_operands = _get_new_operands_with_replacements(expression,
                                                       replacements)
    return Expression(expression.operator, new_operands)


def _get_new_operands_with_replacements(expression, replacements):
    new_operands = []
    for operand in expression.operands:
        replacements_for_expr = replacements[expression]
        if operand in replacements_for_expr:
            operand_replacement = replacements_for_expr[operand]
            if operand_replacement is not None:
                new_operands.append(operand_replacement.copy())
        else:
            new_operands.append(_perform_constant_folding(operand,
                                                          replacements))
    return new_operands