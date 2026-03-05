"""Translate between command-array stacks and CAS expression trees.

Adapted for the ``expressions`` package where CONSTANT and INTEGER
values are stored in external tuples (not embedded in the command
array).
"""

import numpy as np

from ..operators import IS_TERMINAL_MAP, IS_ARITY_2_MAP, CONSTANT, INTEGER, VARIABLE
from .cas_expression import CASExpression, get_interned_integer, get_interned_variable


def build_cas_expression(stack, constants, integers):
    """Translate a command array into a CAS expression tree.

    Memoizes by row index so that shared sub-expressions in the DAG
    are built only once.

    Parameters
    ----------
    stack : Nx3 numpy array of int
        The command array encoding the equation.
    constants : tuple of float
        Constant values indexed by CONSTANT node params.
    integers : tuple of int
        Integer values indexed by INTEGER node params.

    Returns
    -------
    CASExpression
    """
    memo = {}
    return _build_expression_recursive(stack, constants, integers, len(stack) - 1, memo)


def _build_expression_recursive(stack, constants, integers, location, memo):
    if location in memo:
        return memo[location]

    operator = int(stack[location, 0])
    param_1 = int(stack[location, 1])
    param_2 = int(stack[location, 2])

    if IS_TERMINAL_MAP[operator]:
        if operator == CONSTANT:
            # Store an opaque constant index — constant folding uses
            # this to track which constants can be merged.
            result = CASExpression(CONSTANT, [param_1])
        elif operator == INTEGER:
            # Store the actual integer *value* for integer arithmetic
            # during automatic simplification.
            value = integers[param_1] if param_1 < len(integers) else 0
            result = get_interned_integer(value)
        else:
            # VARIABLE — store the column index
            result = get_interned_variable(param_1)
        memo[location] = result
        return result

    operands = [_build_expression_recursive(stack, constants, integers, param_1, memo)]
    if IS_ARITY_2_MAP[operator]:
        operands.append(
            _build_expression_recursive(stack, constants, integers, param_2, memo)
        )
    result = CASExpression(operator, operands)
    memo[location] = result
    return result


def build_simplified_cas_expression(stack, constants, integers):
    """Build a CAS tree and simplify each node as it is constructed.

    This fuses the bottom-up tree build with
    :func:`automatic_simplify`, eliminating a separate recursive
    simplification pass.  The memo stores *already-simplified* nodes so
    shared sub-expressions are only simplified once.

    Parameters
    ----------
    stack : Nx3 numpy array of int
        The command array encoding the equation.
    constants : tuple of float
        Constant values indexed by CONSTANT node params.
    integers : tuple of int
        Integer values indexed by INTEGER node params.

    Returns
    -------
    CASExpression
    """
    from .automatic_simplification import SIMPLIFICATION_FUNCTIONS

    memo = {}
    return _build_simplified_recursive(
        stack,
        constants,
        integers,
        len(stack) - 1,
        memo,
        SIMPLIFICATION_FUNCTIONS,
    )


def _build_simplified_recursive(stack, constants, integers, location, memo, simp_funcs):
    if location in memo:
        return memo[location]

    operator = int(stack[location, 0])
    param_1 = int(stack[location, 1])
    param_2 = int(stack[location, 2])

    if IS_TERMINAL_MAP[operator]:
        if operator == CONSTANT:
            result = CASExpression(CONSTANT, [param_1])
        elif operator == INTEGER:
            value = integers[param_1] if param_1 < len(integers) else 0
            result = get_interned_integer(value)
        else:
            result = get_interned_variable(param_1)
        memo[location] = result
        return result

    # Operands are already simplified (bottom-up + memo).
    operands = [
        _build_simplified_recursive(
            stack, constants, integers, param_1, memo, simp_funcs
        )
    ]
    if IS_ARITY_2_MAP[operator]:
        operands.append(
            _build_simplified_recursive(
                stack, constants, integers, param_2, memo, simp_funcs
            )
        )
    node = CASExpression(operator, operands)
    # Dispatch to the operator-specific simplifier.
    result = simp_funcs[operator](node)
    memo[location] = result
    return result


def build_agraph_stack(expression, original_constants):
    """Translate a CAS expression tree back into a command array.

    Parameters
    ----------
    expression : CASExpression
        The simplified CAS expression.
    original_constants : tuple of float
        The original constant values.  Each CAS ``CONSTANT`` node
        references an index into this tuple; unknown indices default
        to ``1.0``.

    Returns
    -------
    tuple
        ``(command_array, constants, integers, const_idx_map)`` — a
        uint8 command array, the corresponding constant/integer tuples,
        and a dict mapping each original CAS constant index to the new
        sequential index in the output constants tuple.
    """
    stack_dict = {}
    const_list = []
    int_list = []
    const_idx_map = {}  # old_idx -> new_idx (dedup same original index)
    int_val_map = {}  # value -> new_idx (dedup same integer value)
    _build_stack_recursive(
        expression,
        stack_dict,
        original_constants,
        const_list,
        int_list,
        const_idx_map,
        int_val_map,
    )

    stack = np.empty((len(stack_dict), 3), dtype=np.uint8)
    for command, loc in stack_dict.items():
        stack[loc] = command
    return stack, tuple(const_list), tuple(int_list), const_idx_map


def _build_stack_recursive(
    expression,
    stack_dict,
    original_constants,
    const_list,
    int_list,
    const_idx_map,
    int_val_map,
):
    operator = expression.operator

    if operator == CONSTANT:
        old_idx = expression.operands[0]
        if old_idx in const_idx_map:
            new_idx = const_idx_map[old_idx]
        else:
            new_idx = len(const_list)
            if old_idx < len(original_constants):
                value = float(original_constants[old_idx])
            else:
                value = 1.0
            const_list.append(value)
            const_idx_map[old_idx] = new_idx
        command = (operator, new_idx, new_idx)
        return _add_command_to_stack_dict(command, stack_dict)

    if operator == INTEGER:
        value = expression.operands[0]
        if value in int_val_map:
            new_idx = int_val_map[value]
        else:
            new_idx = len(int_list)
            int_list.append(value)
            int_val_map[value] = new_idx
        command = (operator, new_idx, new_idx)
        return _add_command_to_stack_dict(command, stack_dict)

    if operator == VARIABLE:
        col = expression.operands[0]
        command = (operator, col, col)
        return _add_command_to_stack_dict(command, stack_dict)

    # Non-terminal
    operand_locations = [
        _build_stack_recursive(
            operand,
            stack_dict,
            original_constants,
            const_list,
            int_list,
            const_idx_map,
            int_val_map,
        )
        for operand in expression.operands
    ]

    if len(operand_locations) == 1:
        command = (operator, operand_locations[0], operand_locations[0])
        return _add_command_to_stack_dict(command, stack_dict)

    if len(operand_locations) == 2:
        command = (operator, operand_locations[0], operand_locations[1])
        return _add_command_to_stack_dict(command, stack_dict)

    # Associative operators with >2 operands (flattened sums/products)
    if not expression.is_constant_valued and expression.operands[0].is_constant_valued:
        loc = _add_associative_operators_to_stack(
            operator, operand_locations[1:], stack_dict
        )
        command = (operator, operand_locations[0], loc)
        return _add_command_to_stack_dict(command, stack_dict)

    return _add_associative_operators_to_stack(operator, operand_locations, stack_dict)


def _add_command_to_stack_dict(command, stack_dict):
    if command in stack_dict:
        return stack_dict[command]
    loc = len(stack_dict)
    stack_dict[command] = loc
    return loc


def _add_associative_operators_to_stack(operator, operand_locs, stack_dict):
    if len(operand_locs) == 1:
        return operand_locs[0]
    operand_div = len(operand_locs) // 2
    loc_1 = _add_associative_operators_to_stack(
        operator, operand_locs[:operand_div], stack_dict
    )
    loc_2 = _add_associative_operators_to_stack(
        operator, operand_locs[operand_div:], stack_dict
    )
    command = (operator, loc_1, loc_2)
    return _add_command_to_stack_dict(command, stack_dict)
