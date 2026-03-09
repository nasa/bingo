"""Stack reduction for AGraph expressions.

Provides stack reduction — removing unused commands and remapping
parameter references.  This is the "cheap" simplification used during
GA evaluation; for full algebraic simplification see :mod:`._simplify`.
"""

import numpy as np

from ..operators import ARITY_2_IDS, TERMINAL_IDS, CONSTANT, INTEGER


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
        if node not in TERMINAL_IDS:
            util[int(p1s[i])] = 1
            if node in ARITY_2_IDS:
                util[int(p2s[i])] = 1
    return util


def reduce(raw_command_array, raw_constants, raw_integers):
    """Reduce the raw stack and derive simplified constants and integers.

    Performs reduction and terminal renumbering in a single pass over the
    raw command array.  Unused rows are dropped, operator row-references
    are remapped, and CONSTANT / INTEGER indices are compacted to
    sequential positions with only the referenced values retained.

    Parameters
    ----------
    raw_command_array : Nx3 numpy array
        The GA-facing command stack.
    raw_constants : tuple of float
        Constant values indexed by CONSTANT node params.
    raw_integers : tuple of int
        Integer values indexed by INTEGER node params.

    Returns
    -------
    tuple
        ``(command_array, constants, integers, constant_mapping)``.
        ``command_array`` is the reduced Mx3 stack with renumbered
        CONSTANT / INTEGER params.  ``constants`` and ``integers`` are
        tuples containing only the values actually referenced.
        ``constant_mapping`` is a tuple where
        ``constant_mapping[reduced_idx] == raw_idx``.
    """
    if raw_command_array.shape[0] == 0:
        return (
            np.empty((0, 3), dtype=raw_command_array.dtype),
            (),
            (),
            (),
        )

    used_commands = get_utilized_commands(raw_command_array)
    num_commands = sum(used_commands)
    stack = np.empty((num_commands, 3), dtype=raw_command_array.dtype)
    reduced_map = np.cumsum(used_commands) - 1

    new_constants = []
    new_integers = []
    reduced_to_raw = []
    j = 0
    for i in range(raw_command_array.shape[0]):
        if not used_commands[i]:
            continue
        node = int(raw_command_array[i, 0])
        stack[j, 0] = node
        if node in TERMINAL_IDS:
            if node == CONSTANT:
                old_idx = int(raw_command_array[i, 1])
                new_idx = len(new_constants)
                value = raw_constants[old_idx] if old_idx < len(raw_constants) else 1.0
                new_constants.append(value)
                reduced_to_raw.append(old_idx)
                stack[j, 1] = new_idx
                stack[j, 2] = new_idx
            elif node == INTEGER:
                old_idx = int(raw_command_array[i, 1])
                new_idx = len(new_integers)
                value = raw_integers[old_idx] if old_idx < len(raw_integers) else 0
                new_integers.append(value)
                stack[j, 1] = new_idx
                stack[j, 2] = new_idx
            else:
                stack[j, 1] = raw_command_array[i, 1]
                stack[j, 2] = raw_command_array[i, 2]
        else:
            stack[j, 1] = reduced_map[int(raw_command_array[i, 1])]
            if node in ARITY_2_IDS:
                stack[j, 2] = reduced_map[int(raw_command_array[i, 2])]
            else:
                stack[j, 2] = stack[j, 1]
        j += 1

    return stack, tuple(new_constants), tuple(new_integers), tuple(reduced_to_raw)
