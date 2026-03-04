"""Top-level CAS simplification pipeline.

Orchestrates the full algebraic simplification:
``command_array → reduce (dead-code elimination) → CAS tree →
auto-simplify → fold constants → optional modifications →
command_array``.
"""

from ._reduce import reduce as stack_reduce
from .interpreter import build_simplified_cas_expression, build_agraph_stack
from .constant_folding import fold_constants
from .optional_modifications import optional_modifications


def simplify(raw_command_array, raw_constants, raw_integers):
    """Simplify via the full CAS pipeline.

    A cheap :func:`reduce` pass runs first to eliminate dead code and
    compact terminals so that the CAS tree is built from the smallest
    possible stack.

    Parameters
    ----------
    raw_command_array : Nx3 numpy array of uint8
    raw_constants : tuple of float
    raw_integers : tuple of int

    Returns
    -------
    tuple
        ``(command_array, constants, integers)`` — a simplified uint8
        command array with corresponding constant and integer tuples.
    """
    # Eliminate dead code first — the CAS pipeline then operates on the
    # smaller, compacted stack.
    reduced_stack, reduced_constants, reduced_integers = stack_reduce(
        raw_command_array, raw_constants, raw_integers
    )

    # Build the CAS tree with automatic simplification fused in
    # (each node is simplified as it is constructed bottom-up).
    cas_expr = build_simplified_cas_expression(
        reduced_stack, reduced_constants, reduced_integers
    )

    # Fold constants — merges multiple constant-valued sub-expressions.
    cas_expr = fold_constants(cas_expr)

    cas_expr = optional_modifications(cas_expr)
    return build_agraph_stack(cas_expr, reduced_constants)
