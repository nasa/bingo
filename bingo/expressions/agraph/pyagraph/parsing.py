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
_NON_UNARY_OP_PATTERN = re.compile(r"([*/^()])")
_NEGATIVE_PATTERN = re.compile(r"-([^\s\d])")


def eq_string_to_command_array_and_constants(eq_string):
    """Convert an equation string to command array, constants, and integers.

    Parameters
    ----------
    eq_string : str or sympy.Expr
        An equation string or sympy expression.

    Returns
    -------
    tuple of (Nx3 numpy array of uint8, tuple of float, tuple of int)
        ``(command_array, constants, integers)``
    """
    eq_string = str(eq_string)
    infix_tokens = _eq_string_to_infix_tokens(eq_string)
    postfix_tokens = _infix_to_postfix(infix_tokens)
    return _postfix_to_command_array_and_constants(postfix_tokens)


def _eq_string_to_infix_tokens(eq_string):
    if any(bad in eq_string for bad in ["zoo", "I", "oo", "nan"]):
        raise RuntimeError("Cannot parse inf/complex")
    eq_string = eq_string.replace(")(", ")*(").replace("**", "^")
    eq_string = _NEGATIVE_PATTERN.sub(r"-1 * \1", eq_string)
    tokens = _NON_UNARY_OP_PATTERN.sub(r" \1 ", eq_string).split(" ")
    tokens = [x.lower() for x in tokens if x != ""]
    return tokens


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


def _postfix_to_command_array_and_constants(postfix_tokens):
    stack = []
    command_array = []
    command_to_i = {}
    constants = []
    integers = []
    n_constants = 0
    n_integers = 0
    i = 0

    for token in postfix_tokens:
        if token in _OPERATORS:
            operands = stack.pop(), stack.pop()
            command = [_OPERATOR_MAP[token], operands[1], operands[0]]
        elif token in _FUNCTIONS:
            operand = stack.pop()
            command = [_OPERATOR_MAP[token], operand, operand]
        else:
            var_or_const = _VAR_OR_CONST_PATTERN.fullmatch(token)
            integer = _INT_PATTERN.fullmatch(token)
            if var_or_const:
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
