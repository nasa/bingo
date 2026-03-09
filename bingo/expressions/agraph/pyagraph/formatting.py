"""String formatting for AGraph expressions.

Generates string representations in three formats:

* **console** — human-readable infix notation (used by ``__str__``).
* **sympy** — sympy ``srepr``-style functional notation that can be
  round-tripped through ``sympy.sympify``.
* **latex** — LaTeX math notation.

Variables are named ``X0``, ``X1``, … (no underscore).
Constants are named ``C0``, ``C1``, … (no underscore).

Parenthesization in console and LaTeX formats is precedence-aware:
terminals and function calls are never wrapped, and nested operations
at the same precedence level omit unnecessary parentheses (e.g.
``X0*X1*X2`` rather than ``(X0*X1)*X2``).

Attributes
----------
LATEX_FORMAT_MAP : dict {int: str}
CONSOLE_FORMAT_MAP : dict {int: str}
SYMPY_SREPR_FORMAT_MAP : dict {int: str}
"""

from .operators import (
    VARIABLE,
    CONSTANT,
    INTEGER,
    ADDITION,
    SUBTRACTION,
    MULTIPLICATION,
    DIVISION,
    POWER,
    SAFE_POWER,
    SQUARE,
    CUBE,
    SQRT,
    ABS,
    EXPONENTIAL,
    LOGARITHM,
    SIN,
    COS,
    TAN,
    ARCSIN,
    ARCCOS,
    ARCTAN,
    SINH,
    COSH,
    TANH,
    IS_ARITY_2_MAP,
)

# ------------------------------------------------------------------ #
#  Operator precedence (binary arithmetic / power only)               #
# ------------------------------------------------------------------ #

_BINARY_PRECEDENCE = {
    ADDITION: 0,
    SUBTRACTION: 0,
    MULTIPLICATION: 1,
    DIVISION: 1,
    POWER: 2,
    SAFE_POWER: 2,
}

# ------------------------------------------------------------------ #
#  Console format (human-readable infix, bare templates)              #
# ------------------------------------------------------------------ #

CONSOLE_FORMAT_MAP = {
    ADDITION: "{} + {}",
    SUBTRACTION: "{} - {}",
    MULTIPLICATION: "{}*{}",
    DIVISION: "{}/{}",
    POWER: "{}**{}",
    SAFE_POWER: "|{}|**{}",
    SQUARE: "{}**2",
    CUBE: "{}**3",
    SQRT: "sqrt({})",
    ABS: "|{}|",
    EXPONENTIAL: "exp({})",
    LOGARITHM: "log({})",
    SIN: "sin({})",
    COS: "cos({})",
    TAN: "tan({})",
    ARCSIN: "asin({})",
    ARCCOS: "acos({})",
    ARCTAN: "atan({})",
    SINH: "sinh({})",
    COSH: "cosh({})",
    TANH: "tanh({})",
}

# ------------------------------------------------------------------ #
#  Sympy srepr format                                                 #
# ------------------------------------------------------------------ #

SYMPY_SREPR_FORMAT_MAP = {
    ADDITION: "Add({}, {})",
    SUBTRACTION: "Add({}, Mul(Integer(-1), {}))",
    MULTIPLICATION: "Mul({}, {})",
    DIVISION: "Mul({}, Pow({}, Integer(-1)))",
    POWER: "Pow({}, {})",
    SAFE_POWER: "Pow(Abs({}), {})",
    SQUARE: "Pow({}, Integer(2))",
    CUBE: "Pow({}, Integer(3))",
    SQRT: "Pow({}, Rational(1, 2))",
    ABS: "Abs({})",
    EXPONENTIAL: "exp({})",
    LOGARITHM: "log({})",
    SIN: "sin({})",
    COS: "cos({})",
    TAN: "tan({})",
    ARCSIN: "asin({})",
    ARCCOS: "acos({})",
    ARCTAN: "atan({})",
    SINH: "sinh({})",
    COSH: "cosh({})",
    TANH: "tanh({})",
}

# ------------------------------------------------------------------ #
#  LaTeX format (bare templates — wrapping added dynamically)         #
# ------------------------------------------------------------------ #

LATEX_FORMAT_MAP = {
    ADDITION: "{} + {}",
    SUBTRACTION: "{} - {}",
    MULTIPLICATION: "{} \\cdot {}",
    DIVISION: "\\frac{{ {} }}{{ {} }}",
    POWER: "{{{}}}^{{ {} }}",
    SAFE_POWER: "{{|{}|}}^{{ {} }}",
    SQUARE: "{{{}}}^2",
    CUBE: "{{{}}}^3",
    SQRT: "\\sqrt{{ {} }}",
    ABS: "|{}|",
    EXPONENTIAL: "exp{{ {} }}",
    LOGARITHM: "log{{ {} }}",
    SIN: "sin{{ {} }}",
    COS: "cos{{ {} }}",
    TAN: "tan {{ {} }}",
    ARCSIN: "asin {{ {} }}",
    ARCCOS: "acos {{ {} }}",
    ARCTAN: "atan {{ {} }}",
    SINH: "sinh{{ {} }}",
    COSH: "cosh{{ {} }}",
    TANH: "tanh {{ {} }}",
}


_FORMAT_DICTS = {
    "console": CONSOLE_FORMAT_MAP,
    "sympy": SYMPY_SREPR_FORMAT_MAP,
    "latex": LATEX_FORMAT_MAP,
}


# ------------------------------------------------------------------ #
#  Precedence-aware parenthesization                                  #
# ------------------------------------------------------------------ #


def _needs_parens(child_op, parent_op, position, eq_format):
    """Decide whether a child sub-expression needs ``(…)`` wrapping.

    Parameters
    ----------
    child_op : int
        Operator ID of the child node.
    parent_op : int
        Operator ID of the parent node.
    position : str
        ``"left"``, ``"right"``, or ``"base"`` (for SQUARE / CUBE).
    eq_format : str
        ``"console"`` or ``"latex"``.
    """
    # Terminals and function-style ops never need wrapping.
    if child_op not in _BINARY_PRECEDENCE:
        return False

    child_prec = _BINARY_PRECEDENCE[child_op]
    parent_prec = _BINARY_PRECEDENCE.get(parent_op, 99)

    if position == "left":
        # LaTeX \frac provides structural grouping for the numerator.
        if eq_format == "latex" and parent_op == DIVISION:
            return False
        # |…| in SAFE_POWER already groups the left operand.
        if parent_op == SAFE_POWER:
            return False
        if child_prec < parent_prec:
            return True
        # Right-associative operators: left child at same prec needs
        # parens, e.g. (a**b)**c  →  must not become a**b**c.
        if child_prec == parent_prec and parent_op in (POWER, SAFE_POWER):
            return True
        return False

    if position == "right":
        # LaTeX structural grouping: \frac{…}{…} and ^{…} enclose the
        # right operand so no extra parentheses are needed.
        if eq_format == "latex" and parent_op in (DIVISION, POWER, SAFE_POWER):
            return False
        if child_prec < parent_prec:
            return True
        if child_prec == parent_prec:
            # Non-commutative operators at the same level need parens on
            # the right to preserve meaning, e.g. a - (b + c), a / (b * c).
            if parent_op in (SUBTRACTION, DIVISION):
                return True
            # a * (b / c)  ≠  a * b / c
            if parent_op == MULTIPLICATION and child_op == DIVISION:
                return True
        return False

    # position == "base" — SQUARE / CUBE operand
    # Any binary-op child needs wrapping so that the exponent applies
    # to the whole sub-expression, e.g. (X0+X1)**2 not X0+X1**2.
    return True


def _wrap(child_str, child_op, parent_op, position, eq_format):
    """Return *child_str* wrapped in ``(…)`` if needed."""
    if _needs_parens(child_op, parent_op, position, eq_format):
        return f"({child_str})"
    return child_str


# ------------------------------------------------------------------ #
#  Public API                                                         #
# ------------------------------------------------------------------ #


def get_formatted_string(eq_format, command_array, constants, integers):
    """Build a formatted string from a command array.

    Parameters
    ----------
    eq_format : str
        One of ``"console"``, ``"sympy"``, or ``"latex"``.
    command_array : Nx3 array of int
        The command stack.
    constants : tuple of numeric
        Numeric constants in the equation.
    integers : tuple of int
        Integer values in the equation.

    Returns
    -------
    str
        Equation in the requested format.
    """
    format_dict = _FORMAT_DICTS.get(eq_format, CONSOLE_FORMAT_MAP)
    is_srepr = eq_format == "sympy"
    str_list = []
    op_list = []
    for stack_element in command_array:
        tmp_str = _get_formatted_element_string(
            stack_element,
            str_list,
            op_list,
            format_dict,
            constants,
            integers,
            is_srepr,
            eq_format,
        )
        str_list.append(tmp_str)
        op_list.append(int(stack_element[0]))
    return str_list[-1]


def _get_formatted_element_string(
    stack_element,
    str_list,
    op_list,
    format_dict,
    constants,
    integers,
    is_srepr=False,
    eq_format="console",
):
    node, param1, param2 = stack_element

    # ---- terminals ----
    if node == VARIABLE:
        if is_srepr:
            return f"Symbol('X{param1}')"
        return f"X{param1}"
    if node == CONSTANT:
        if param1 >= len(constants):
            if is_srepr:
                return "Symbol('C')"
            return "?"
        if is_srepr:
            return f"Float({constants[param1]})"
        return str(constants[param1])
    if node == INTEGER:
        if param1 < len(integers):
            if is_srepr:
                return f"Integer({int(integers[param1])})"
            return str(int(integers[param1]))
        if is_srepr:
            return "Symbol('?')"
        return "?"

    # ---- sympy srepr — functional notation, no smart wrapping ----
    if is_srepr:
        return format_dict[node].format(str_list[param1], str_list[param2])

    # ---- console / latex — precedence-aware wrapping ----
    child1_str = str_list[param1]
    child1_op = op_list[param1]

    if IS_ARITY_2_MAP.get(node, False):
        child2_str = str_list[param2]
        child2_op = op_list[param2]
        child1_str = _wrap(child1_str, child1_op, node, "left", eq_format)
        child2_str = _wrap(child2_str, child2_op, node, "right", eq_format)
        return format_dict[node].format(child1_str, child2_str)

    if node in (SQUARE, CUBE):
        child1_str = _wrap(child1_str, child1_op, node, "base", eq_format)
        return format_dict[node].format(child1_str)

    # Unary functions (sin, cos, sqrt, abs, exp, log, …)
    return format_dict[node].format(child1_str)
