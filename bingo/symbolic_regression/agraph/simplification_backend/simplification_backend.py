"""
This module represents the python backend associated with `AGraph` equation
simplification.  This backend is used to perform the the simplification of an
equation represented by an `AGraph`.  It can be performed based on a simple
reduction or a more involved algebraic simplification.
"""

import numpy as np

from ..operator_definitions import IS_ARITY_2_MAP, IS_TERMINAL_MAP
from .simplify import simplify as cas_simplify

ENGINE = "Python"


def get_utilized_commands(stack):
    """Find which commands are utilized

    Find the commands (rows) of the stack upon which the last command of the
    stack depends. This is inclusive of the last command.

    Parameters
    ----------
    stack : Nx3 numpy array of int.
        The command stack associated with an equation. N is the number of
        commands in the stack.

    Returns
    -------
    bytearray of length N
        Non-zero values for whether each command is utilized.
    """
    n = stack.shape[0]
    util = bytearray(n)
    util[-1] = 1
    nodes = stack[:, 0]
    p1s = stack[:, 1]
    p2s = stack[:, 2]
    for i in range(n - 1, -1, -1):
        if not util[i]:
            continue
        node = int(nodes[i])
        if not IS_TERMINAL_MAP[node]:
            util[int(p1s[i])] = 1
            if IS_ARITY_2_MAP[node]:
                util[int(p2s[i])] = 1
    return util


def simplify_stack(stack):
    """Simplifies a stack based on computational algebra

    An acyclic graph is given in stack form.  The stack is algebraically
    simplified and put in a canonical form.

    Parameters
    ----------
    stack : Nx3 numpy array of int.
        The command stack associated with an equation. N is the number of
        commands in the stack.

    Returns
    -------
    Mx3 numpy array of int. :
        a simplified stack representing the original equation
    """
    return cas_simplify(stack)


def reduce_stack(stack):
    """Reduces a stack

    An acyclic graph is given in stack form.  The stack is simplified to
    consist only of the commands used by the last command.

    Parameters
    ----------
    stack : Nx3 numpy array of int.
        The command stack associated with an equation. N is the number of
        commands in the stack.

    Returns
    -------
    Mx3 numpy array of int. :
        a simplified stack where M is the number of  used commands
    """
    used_commands = get_utilized_commands(stack)
    num_commands = np.count_nonzero(used_commands)
    new_stack = np.empty((num_commands, 3), dtype=stack.dtype)
    reduced_map = np.cumsum(used_commands) - 1
    j = 0
    for i in range(stack.shape[0]):
        if not used_commands[i]:
            continue
        node = stack[i, 0]
        new_stack[j, 0] = node
        if IS_TERMINAL_MAP[node]:
            new_stack[j, 1] = stack[i, 1]
            new_stack[j, 2] = stack[i, 2]
        else:
            new_stack[j, 1] = reduced_map[stack[i, 1]]
            if IS_ARITY_2_MAP[node]:
                new_stack[j, 2] = reduced_map[stack[i, 2]]
            else:
                new_stack[j, 2] = new_stack[j, 1]
        j += 1
    return new_stack
