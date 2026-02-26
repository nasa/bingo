"""Stack simplification for AGraph expressions.

Provides stack reduction — removing unused commands and remapping
parameter references.
"""

import numpy as np

from .operators import IS_ARITY_2_MAP, IS_TERMINAL_MAP, CONSTANT, INTEGER


def get_utilized_commands(stack):
    """Find which commands are utilized by the final output.

    Parameters
    ----------
    stack : Nx3 numpy array of int
        The command stack.

    Returns
    -------
    bytearray of length N
        Non-zero values indicate the command is utilized.
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


def reduce_stack(stack):
    """Reduce a stack by removing unused commands.

    Unused commands are removed and parameter references are remapped.
    This does *not* renumber constant/integer indices — that is handled
    by the expression class after reduction.

    Parameters
    ----------
    stack : Nx3 numpy array of int
        The command stack.

    Returns
    -------
    Mx3 numpy array (same dtype as input)
        Reduced stack with only used commands.
    """
    used_commands = get_utilized_commands(stack)
    num_commands = sum(used_commands)
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
