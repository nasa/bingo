"""Operator definitions for the AGraph expression.

All operator IDs are non-negative and fit within uint8 (0–255).
They are grouped logically: terminals, arithmetic, power/root,
miscellaneous, exponential/logarithmic, trigonometric, hyperbolic.

Attributes
----------
IS_ARITY_2_MAP : dict {int: bool}
    Whether the operator has arity 2 (binary).
IS_TERMINAL_MAP : dict {int: bool}
    Whether the operator is a terminal node.
OPERATOR_NAMES : dict {int: list of str}
    Common names for each operator (used in parsing).
"""

# --- Terminals (0–2) ---
VARIABLE = 0
CONSTANT = 1
INTEGER = 2

# --- Arithmetic (3–6) ---
ADDITION = 3
SUBTRACTION = 4
MULTIPLICATION = 5
DIVISION = 6

# --- Power / Root (7–11) ---
POWER = 7
SAFE_POWER = 8
SQUARE = 9
CUBE = 10
SQRT = 11

# --- Miscellaneous (12) ---
ABS = 12

# --- Exponential / Logarithmic (13–14) ---
EXPONENTIAL = 13
LOGARITHM = 14

# --- Trigonometric (15–20) ---
SIN = 15
COS = 16
TAN = 17
ARCSIN = 18
ARCCOS = 19
ARCTAN = 20

# --- Hyperbolic (21–23) ---
SINH = 21
COSH = 22
TANH = 23

# ---- Maps ----

IS_TERMINAL_MAP = {
    VARIABLE: True,
    CONSTANT: True,
    INTEGER: True,
    ADDITION: False,
    SUBTRACTION: False,
    MULTIPLICATION: False,
    DIVISION: False,
    POWER: False,
    SAFE_POWER: False,
    SQUARE: False,
    CUBE: False,
    SQRT: False,
    ABS: False,
    EXPONENTIAL: False,
    LOGARITHM: False,
    SIN: False,
    COS: False,
    TAN: False,
    ARCSIN: False,
    ARCCOS: False,
    ARCTAN: False,
    SINH: False,
    COSH: False,
    TANH: False,
}

IS_ARITY_2_MAP = {
    VARIABLE: False,
    CONSTANT: False,
    INTEGER: False,
    ADDITION: True,
    SUBTRACTION: True,
    MULTIPLICATION: True,
    DIVISION: True,
    POWER: True,
    SAFE_POWER: True,
    SQUARE: False,
    CUBE: False,
    SQRT: False,
    ABS: False,
    EXPONENTIAL: False,
    LOGARITHM: False,
    SIN: False,
    COS: False,
    TAN: False,
    ARCSIN: False,
    ARCCOS: False,
    ARCTAN: False,
    SINH: False,
    COSH: False,
    TANH: False,
}

OPERATOR_NAMES = {
    VARIABLE: ["load", "x"],
    CONSTANT: ["constant", "c"],
    INTEGER: ["integer"],
    ADDITION: ["add", "addition", "+"],
    SUBTRACTION: ["subtract", "subtraction", "-"],
    MULTIPLICATION: ["multiply", "multiplication", "*"],
    DIVISION: ["divide", "division", "/"],
    POWER: ["power", "pow", "^"],
    SAFE_POWER: ["safe power", "safe pow"],
    SQUARE: ["square", "sq"],
    CUBE: ["cube", "cb"],
    SQRT: ["square root", "sqrt"],
    ABS: ["absolute value", "||", "|"],
    EXPONENTIAL: ["exponential", "exp", "e"],
    LOGARITHM: ["logarithm", "log"],
    SIN: ["sine", "sin"],
    COS: ["cosine", "cos"],
    TAN: ["tangent", "tan"],
    ARCSIN: ["arcsin", "asin"],
    ARCCOS: ["arccos", "acos"],
    ARCTAN: ["arctan", "atan"],
    SINH: ["sineh", "sinh"],
    COSH: ["cosineh", "cosh"],
    TANH: ["tangenth", "tanh"],
}
