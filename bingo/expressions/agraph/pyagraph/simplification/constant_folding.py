"""Constant folding for CAS expressions.

Reduces the number of distinct constants in an expression by grouping
and merging constant-valued sub-expressions.  The folding is purely
structural — constant *values* are not tracked or computed.
"""

from collections import defaultdict
from itertools import combinations

from ..operators import CONSTANT, INTEGER, VARIABLE, MULTIPLICATION, ADDITION
from .cas_expression import CASExpression


_TERMINAL_OPS = frozenset({CONSTANT, INTEGER, VARIABLE})
_ASSOC_OPS = frozenset({MULTIPLICATION, ADDITION})
_EXHAUSTIVE_FOLDING_MAX_CONSTANTS = 7


def fold_constants(expression):
    """Fold constant-valued sub-expressions together.

    Repeatedly scans the tree to combine constants. For expressions with at
    most seven distinct constants, every non-empty subset is considered.
    Larger expressions use a deterministic local policy that prioritizes
    reducing distinct constant identities, with constant leaves as a
    tie-breaker. The folding is purely structural — no constant values are
    tracked or computed.

    Parameters
    ----------
    expression : CASExpression

    Returns
    -------
    CASExpression
    """
    expression = _group_constants(expression)

    while True:
        # Single fused DFS: discover constants and insertion points
        # together instead of two separate traversals.
        cas_constants, insertion_points_map = _fused_discovery(expression)
        if len(cas_constants) <= _EXHAUSTIVE_FOLDING_MAX_CONSTANTS:
            folded_expression = _find_first_exhaustive_fold(
                expression, cas_constants, insertion_points_map
            )
        else:
            folded_expression = _find_best_local_fold(
                expression, cas_constants, insertion_points_map
            )

        if folded_expression is None:
            return expression
        expression = folded_expression


def _find_first_exhaustive_fold(expression, constants, insertion_points_map):
    for const_subset in _subsets(list(constants)):
        insertion_points = _filter_insertion_points(
            expression, const_subset, insertion_points_map
        )
        replacements = _generate_replacement_instructions(
            const_subset,
            constants,
            insertion_points,
        )
        if replacements:
            return _perform_constant_folding(expression, replacements)
    return None


def _find_best_local_fold(expression, constants, insertion_points_map):
    best_fold = None
    best_score = (0, 0)
    constant_order = list(constants)

    for node in _postorder(expression):
        if node.operator in _TERMINAL_OPS:
            continue

        const_subset = {
            constant for constant in constant_order if constant in node.depends_on
        }
        if not const_subset:
            continue

        insertion_points = _filter_insertion_points(
            node, const_subset, insertion_points_map
        )
        replacements = _generate_replacement_instructions(
            const_subset,
            constants,
            insertion_points,
        )
        if not replacements:
            continue

        folded_node = _perform_constant_folding(node, replacements)
        score = _fold_score(node, folded_node)
        if score > best_score:
            best_fold = (node, folded_node)
            best_score = score

    if best_fold is None:
        return None
    return _replace_identity(expression, *best_fold)


def _postorder(expression):
    if expression.operator not in _TERMINAL_OPS:
        for operand in expression.operands:
            yield from _postorder(operand)
    yield expression


def _fold_score(before, after):
    before_leaves, before_distinct = _constant_counts(before)
    after_leaves, after_distinct = _constant_counts(after)
    return before_distinct - after_distinct, before_leaves - after_leaves


def _constant_counts(expression):
    distinct_constants = set()

    def _count(node):
        if node.operator == CONSTANT:
            distinct_constants.add(node.operands[0])
            return 1
        if node.operator in _TERMINAL_OPS:
            return 0
        return sum(_count(operand) for operand in node.operands)

    return _count(expression), len(distinct_constants)


def _replace_identity(expression, target, replacement):
    if expression is target:
        return replacement
    if expression.operator in _TERMINAL_OPS:
        return expression

    new_operands = [
        _replace_identity(operand, target, replacement)
        for operand in expression.operands
    ]
    if all(new is old for new, old in zip(new_operands, expression.operands)):
        return expression
    return CASExpression(expression.operator, new_operands)


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


def _generate_replacement_instructions(
    const_subset, constants, insertion_points
):
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
        Mapping ``{node: (depends_on, operands)}`` for every non-terminal
        node, where *depends_on* is the node's ``depends_on`` set and
        *operands* is its operand list.
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

    _, operands = ip_map[expression]

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
