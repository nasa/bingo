"""Constant folding for CAS expressions.

Reduces the number of distinct constants in an expression by grouping
and merging constant-valued sub-expressions.
"""

from collections import defaultdict
from itertools import combinations

from ..operators import CONSTANT, INTEGER, VARIABLE, MULTIPLICATION, ADDITION
from .cas_expression import CASExpression


_TERMINAL_OPS = frozenset({CONSTANT, INTEGER, VARIABLE})
_ASSOC_OPS = frozenset({MULTIPLICATION, ADDITION})


def fold_constants(expression):
    """Fold constant-valued sub-expressions together.

    Repeatedly scans the tree to combine constants so that the resulting
    expression uses as few ``CONSTANT`` nodes as possible.

    Parameters
    ----------
    expression : CASExpression

    Returns
    -------
    CASExpression
    """
    expression = _group_constants(expression)

    check_for_folding = True
    while check_for_folding:
        check_for_folding = False
        # Single fused DFS: discover constants and insertion points
        # together instead of two separate traversals.
        constants, insertion_points_map = _fused_discovery(expression)
        for const_subset in _subsets(list(constants)):
            insertion_points = _filter_insertion_points(
                expression, const_subset, insertion_points_map
            )
            replacements = _generate_replacement_instructions(
                const_subset, constants, insertion_points
            )
            if len(replacements) > 0:
                expression = _perform_constant_folding(expression, replacements)
                check_for_folding = True
                break

    return expression


# ------------------------------------------------------------------ #
#  Grouping                                                           #
# ------------------------------------------------------------------ #


def _group_constants(expression):
    if expression.operator in _TERMINAL_OPS:
        return expression

    orig_operands = expression.operands
    new_operands = [_group_constants(operand) for operand in orig_operands]

    if expression.operator in _ASSOC_OPS:
        const_operands = [op for op in new_operands if op.is_constant_valued]
        non_const_operands = [op for op in new_operands if not op.is_constant_valued]
        if len(const_operands) > 1 and len(non_const_operands) > 0:
            const_expr = CASExpression(expression.operator, const_operands)
            return CASExpression(expression.operator, [const_expr] + non_const_operands)

    if all(n is o for n, o in zip(new_operands, orig_operands)):
        return expression
    return CASExpression(expression.operator, new_operands)


# ------------------------------------------------------------------ #
#  Replacement instructions                                           #
# ------------------------------------------------------------------ #


def _generate_replacement_instructions(const_subset, constants, insertion_points):
    if len(insertion_points) > len(const_subset):
        return {}

    replacements = defaultdict(dict)
    constants_to_insert = set()
    expressions_to_replace = set()
    for const_num, (_, insertions) in zip(const_subset, insertion_points.items()):
        const_to_insert = constants[const_num]
        for parent, children in insertions:
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


# ------------------------------------------------------------------ #
#  Fused constant discovery + insertion point computation             #
# ------------------------------------------------------------------ #

_I_SET = frozenset({"i"})


def _subsets(constants):
    for i in range(1, len(constants) + 1):
        for comb in combinations(constants, i):
            yield set(comb)


def _fused_discovery(expression):
    """Single DFS that discovers constants and computes insertion data.

    Returns
    -------
    constants : dict
        Mapping ``{const_index: CASExpression}`` for every distinct
        ``CONSTANT`` node found in *expression*.
    ip_map : dict
        Mapping ``{node: (solely_flags, others_flags, const_operands)}``
        for every non-terminal node, where *solely_flags* and
        *others_flags* are tuples of bools parallel to the node's
        operands, and *const_operands* is a frozenset of operands whose
        ``depends_on`` is a subset of the target constants plus ``"i"``.
    """
    constants = {}
    ip_map = {}
    _fused_dfs(expression, constants, ip_map)
    return constants, ip_map


def _fused_dfs(expression, constants, ip_map):
    """Populate *constants* and *ip_map* in one bottom-up walk."""
    op = expression.operator
    if op == CONSTANT:
        constants[expression.operands[0]] = expression
        return
    if op in _TERMINAL_OPS:  # INTEGER or VARIABLE
        return

    operands = expression.operands
    for operand in operands:
        _fused_dfs(operand, constants, ip_map)

    # Pre-compute per-operand flags that _is_insertion_point and
    # the insertion-point logic both need.  We cache these so that
    # _filter_insertion_points can reuse them without re-walking.
    deps = expression.depends_on
    ip_map[expression] = (deps, operands)


def _filter_insertion_points(expression, constants, ip_map):
    """Build insertion points for a specific *constants* subset.

    Uses the cached data in *ip_map* from the fused DFS, avoiding a
    second full traversal and redundant ``depends_on`` set operations.
    """
    deps = expression.depends_on
    const_set = constants
    const_and_i = const_set | _I_SET

    if not deps.isdisjoint(const_set) and deps <= const_and_i:
        return {expression: [(None, frozenset([expression]))]}

    insertion_points = defaultdict(set)
    _filter_ip_recurse(
        expression, const_set, const_and_i, insertion_points, ip_map, parent=None
    )
    return insertion_points


def _filter_ip_recurse(
    expression, const_set, const_and_i, insertion_points, ip_map, parent
):
    if expression not in ip_map:
        return  # terminal

    deps, operands = ip_map[expression]

    for operand in operands:
        _filter_ip_recurse(
            operand, const_set, const_and_i, insertion_points, ip_map, expression
        )

    # Check if this node is an insertion point: at least one operand
    # depends solely on the target constants, and at least one depends
    # on something else.
    any_solely = False
    any_others = False
    for operand in operands:
        od = operand.depends_on
        has_consts = not od.isdisjoint(const_set)
        has_other = not od <= const_and_i
        if has_consts and not has_other:
            any_solely = True
        if has_other:
            any_others = True
        if any_solely and any_others:
            break
    if not (any_solely and any_others):
        return

    if expression.is_constant_valued:
        insertion_points[expression].add((parent, frozenset([expression])))
    else:
        constant_operands = frozenset(
            operand for operand in operands if operand.depends_on <= const_and_i
        )
        insertion_points[expression].add((expression, constant_operands))


# ------------------------------------------------------------------ #
#  Folding execution                                                  #
# ------------------------------------------------------------------ #


def _perform_constant_folding(expression, replacements):
    if None in replacements:
        return replacements[None][expression]
    return _recursive_expression_replacement(expression, replacements)


def _recursive_expression_replacement(expression, replacements):
    if expression not in replacements:
        if expression.operator in _TERMINAL_OPS:
            return expression
        return expression.map(lambda x: _perform_constant_folding(x, replacements))
    new_operands = _get_new_operands_with_replacements(expression, replacements)
    return CASExpression(expression.operator, new_operands)


def _get_new_operands_with_replacements(expression, replacements):
    new_operands = []
    for operand in expression.operands:
        replacements_for_expr = replacements[expression]
        if operand in replacements_for_expr:
            operand_replacement = replacements_for_expr[operand]
            if operand_replacement is not None:
                new_operands.append(operand_replacement)
        else:
            new_operands.append(_perform_constant_folding(operand, replacements))
    return new_operands
