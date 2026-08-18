"""Equation string parsing for AGraph expressions.

Converts equation strings (or sympy expressions) into command arrays,
constants tuples, and integers tuples. Integer literals in parsed
expressions are routed to the integers tuple with the INTEGER operator.
"""

import re
import numpy as np

from .operators import (
    VARIABLE,
    CONSTANT,
    INTEGER,
    ADDITION,
    SUBTRACTION,
    MULTIPLICATION,
    DIVISION,
    POWER,
    SIN,
    COS,
    TAN,
    SINH,
    COSH,
    TANH,
    EXPONENTIAL,
    LOGARITHM,
    ABS,
    SQRT,
    ARCSIN,
    ARCCOS,
    ARCTAN,
    SQUARE,
    CUBE,
)


_OPERATORS = {"+", "-", "*", "/", "^"}
_FUNCTIONS = {
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "exp",
    "log",
    "abs",
    "sqrt",
    "arcsin",
    "arccos",
    "arctan",
    "asin",
    "acos",
    "atan",
    "sq",
    "square",
    "cb",
    "cube",
}
_PRECEDENCE = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}

_OPERATOR_MAP = {
    "+": ADDITION,
    "-": SUBTRACTION,
    "*": MULTIPLICATION,
    "/": DIVISION,
    "^": POWER,
    "X": VARIABLE,
    "x": VARIABLE,
    "C": CONSTANT,
    "c": CONSTANT,
    "sin": SIN,
    "cos": COS,
    "tan": TAN,
    "sinh": SINH,
    "cosh": COSH,
    "tanh": TANH,
    "exp": EXPONENTIAL,
    "log": LOGARITHM,
    "abs": ABS,
    "sqrt": SQRT,
    "arcsin": ARCSIN,
    "arccos": ARCCOS,
    "arctan": ARCTAN,
    "asin": ARCSIN,
    "acos": ARCCOS,
    "atan": ARCTAN,
    "sq": SQUARE,
    "square": SQUARE,
    "cb": CUBE,
    "cube": CUBE,
}

_VAR_OR_CONST_PATTERN = re.compile(r"([XC])_?(\d+)", re.IGNORECASE)
_INT_PATTERN = re.compile(r"\d+")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_]\w*")
_NUMBER_PATTERN = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_TOKEN_PATTERN = re.compile(
    r"\s*(\*\*|[+\-*/^()]|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[A-Za-z_]\w*)"
)
_NEGATIVE_PATTERN = re.compile(r"-([^\s\d])")
_LEGACY_SYMBOL_PATTERN = re.compile(r"[XC]_?\d+", re.IGNORECASE)


def validate_variable_names(variables):
    """Validate and normalize persistent variable names.

    Parameters
    ----------
    variables : iterable of str or None

    Returns
    -------
    tuple of str or None
    """
    return _validate_symbol_names(variables, "variables")


def validate_parameter_names(parameters, *, variables=None):
    """Validate and normalize parse-only parameter names."""
    parameter_names = _validate_symbol_names(parameters, "parameters")
    if parameter_names is None:
        return None
    variable_names = () if variables is None else tuple(variables)
    overlap = sorted(set(variable_names) & set(parameter_names))
    if overlap:
        raise ValueError(
            "variables and parameters must not overlap; "
            f"shared names: {', '.join(overlap)}"
        )
    return parameter_names


def get_required_variable_dimension(command_array):
    """Return the smallest variable-schema length that fits *command_array*."""
    if command_array is None or len(command_array) == 0:
        return 0
    variable_rows = command_array[np.asarray(command_array)[:, 0] == VARIABLE]
    if len(variable_rows) == 0:
        return 0
    return int(np.max(variable_rows[:, 1])) + 1


def validate_variable_schema_length(variables, required_dimension):
    """Enforce exact-length variable schemas when a dimension is known."""
    variable_names = validate_variable_names(variables)
    if variable_names is None:
        return None
    if len(variable_names) != required_dimension:
        raise ValueError(
            "variables must have exact length matching the required input "
            f"dimension ({required_dimension}), got {len(variable_names)}"
        )
    return variable_names


def _validate_symbol_names(names, label):
    if names is None:
        return None
    try:
        normalized = tuple(str(name) for name in names)
    except TypeError as err:
        raise TypeError(f"{label} must be an iterable of strings") from err
    seen = set()
    for name in normalized:
        if not _IDENTIFIER_PATTERN.fullmatch(name):
            raise ValueError(
                f"{label} entries must be identifier-style names, got {name!r}"
            )
        if name in _FUNCTIONS:
            raise ValueError(f"{label} entries must not use reserved function name {name!r}")
        if name in seen:
            raise ValueError(f"{label} entries must be unique; duplicate {name!r}")
        seen.add(name)
    return normalized


def eq_string_to_command_array_and_constants(eq_string, *, variables=None, parameters=None):
    """Convert an equation string to command array, constants, and integers.

    Parameters
    ----------
    eq_string : str or sympy.Expr
        An equation string or sympy expression.
    variables : tuple of str, optional
        Persistent variable schema used for named parsing.
    parameters : tuple of str, optional
        Parse-only parameter symbols that become optimizable constants.

    Returns
    -------
    tuple of (Nx3 numpy array of uint8, tuple of float, tuple of int)
        ``(command_array, constants, integers)``
    """
    eq_string = str(eq_string)
    variable_names = validate_variable_names(variables)
    parameter_names = validate_parameter_names(parameters, variables=variable_names)
    named_mode = variable_names is not None or parameter_names is not None
    infix_tokens = _eq_string_to_infix_tokens(eq_string)
    postfix_tokens = _infix_to_postfix(infix_tokens)
    return _postfix_to_command_array_and_constants(
        postfix_tokens,
        variable_names=variable_names,
        parameter_names=parameter_names,
        named_mode=named_mode,
    )


def _eq_string_to_infix_tokens(eq_string):
    if any(bad in eq_string for bad in ["zoo", "I", "oo", "nan"]):
        raise RuntimeError("Cannot parse inf/complex")
    eq_string = eq_string.replace(")(", ")*(")
    eq_string = _NEGATIVE_PATTERN.sub(r"-1 * \1", eq_string)
    tokens = []
    position = 0
    while position < len(eq_string):
        match = _TOKEN_PATTERN.match(eq_string, position)
        if match is None:
            if eq_string[position:].strip() == "":
                break
            raise RuntimeError(f"Unknown token {eq_string[position:]}")
        token = match.group(1)
        tokens.append("^" if token == "**" else token)
        position = match.end()
    return _normalize_unary_minus(tokens)


def _normalize_unary_minus(tokens):
    normalized = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        is_unary = token == "-" and (
            i == 0 or tokens[i - 1] in _OPERATORS or tokens[i - 1] == "("
        )
        if is_unary:
            if i + 1 >= len(tokens):
                raise RuntimeError("Dangling unary minus")
            next_token = tokens[i + 1]
            if _NUMBER_PATTERN.fullmatch(next_token):
                normalized.append(f"-{next_token}")
                i += 2
                continue
            normalized.extend(["-1", "*"])
        else:
            normalized.append(token)
        i += 1
    return normalized


def _infix_to_postfix(infix_tokens):
    stack = []
    output = []
    for token in infix_tokens:
        if token in _OPERATORS:
            while (
                len(stack) > 0
                and stack[-1] in _OPERATORS
                and (
                    _PRECEDENCE[stack[-1]] > _PRECEDENCE[token]
                    or (_PRECEDENCE[stack[-1]] == _PRECEDENCE[token] and token != "^")
                )
            ):
                output.append(stack.pop())
            stack.append(token)
        elif token == "(" or token in _FUNCTIONS:
            stack.append(token)
        elif token == ")":
            while len(stack) > 0 and stack[-1] != "(":
                output.append(stack.pop())
            if len(stack) == 0 or stack.pop() != "(":
                raise RuntimeError("Mismatched parenthesis")
            if len(stack) > 0 and stack[-1] in _FUNCTIONS:
                output.append(stack.pop())
        else:
            output.append(token)

    while len(stack) > 0:
        token = stack.pop()
        if token == "(":
            raise RuntimeError("Mismatched parenthesis")
        output.append(token)

    return output


def _postfix_to_command_array_and_constants(
    postfix_tokens,
    *,
    variable_names=None,
    parameter_names=None,
    named_mode=False,
):
    stack = []
    command_array = []
    command_to_i = {}
    constants = []
    integers = []
    parameter_to_i = {}
    n_constants = 0
    n_integers = 0
    i = 0
    variable_lookup = (
        {} if variable_names is None else {name: index for index, name in enumerate(variable_names)}
    )
    parameter_lookup = (
        {} if parameter_names is None else {name: index for index, name in enumerate(parameter_names)}
    )

    for token in postfix_tokens:
        if token in _OPERATORS:
            operands = stack.pop(), stack.pop()
            command = [_OPERATOR_MAP[token], operands[1], operands[0]]
        elif token in _FUNCTIONS or token.lower() in _FUNCTIONS:
            if token not in _FUNCTIONS:
                if named_mode:
                    raise RuntimeError(f"Unknown token {token}")
                token = token.lower()
            operand = stack.pop()
            command = [_OPERATOR_MAP[token], operand, operand]
        else:
            var_or_const = None if named_mode else _VAR_OR_CONST_PATTERN.fullmatch(token)
            integer = _INT_PATTERN.fullmatch(token)
            if named_mode and token in variable_lookup:
                idx = variable_lookup[token]
                command = [VARIABLE, idx, idx]
            elif named_mode and token in parameter_lookup:
                if token not in parameter_to_i:
                    parameter_to_i[token] = n_constants
                    constants.append(1.0)
                    n_constants += 1
                idx = parameter_to_i[token]
                command = [CONSTANT, idx, idx]
            elif var_or_const:
                groups = var_or_const.groups()
                op = _OPERATOR_MAP[groups[0]]
                idx = int(groups[1])
                command = [op, idx, idx]
            elif integer:
                value = int(token)
                command = [INTEGER, n_integers, n_integers]
                integers.append(value)
                n_integers += 1
            else:
                try:
                    constant = float(token)
                    command = [CONSTANT, n_constants, n_constants]
                    constants.append(constant)
                    n_constants += 1
                except ValueError as err:
                    if named_mode and _LEGACY_SYMBOL_PATTERN.fullmatch(token):
                        raise RuntimeError(f"Unknown token {token}") from err
                    raise RuntimeError(f"Unknown token {token}") from err

        key = tuple(command)
        if key in command_to_i:
            stack.append(command_to_i[key])
        else:
            command_to_i[key] = i
            command_array.append(command)
            stack.append(i)
            i += 1

    if len(stack) > 1:
        raise RuntimeError("Error evaluating postfix expression")

    return (
        np.array(command_array, dtype=np.uint8),
        tuple(constants),
        tuple(integers),
    )
