"""Top-level CAS simplification pipeline.

Orchestrates the full algebraic simplification:
``command_array → CAS tree → auto-simplify → fold constants →
optional modifications → command_array``.
"""

from .interpreter import build_cas_expression, build_agraph_stack
from .automatic_simplification import automatic_simplify
from .constant_folding import fold_constants
from .optional_modifications import optional_modifications


def simplify(raw_command_array, raw_constants, raw_integers):
    """Simplify via the full CAS pipeline.

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
    cas_expr = build_cas_expression(
        raw_command_array, raw_constants, raw_integers
    )
    cas_expr = automatic_simplify(cas_expr)
    cas_expr = fold_constants(cas_expr)
    cas_expr = optional_modifications(cas_expr)
    return build_agraph_stack(cas_expr, raw_constants)
