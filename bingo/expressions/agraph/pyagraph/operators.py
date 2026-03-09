"""Operator definitions for the AGraph expression.

All operator IDs are non-negative and fit within uint8 (0–255).
They are grouped logically: terminals, arithmetic, power/root,
miscellaneous, exponential/logarithmic, trigonometric, hyperbolic.

Attributes
----------
TERMINAL_IDS : frozenset of int
    Operator IDs that are terminal nodes.
ARITY_2_IDS : frozenset of int
    Operator IDs that are binary (arity-2) operators.
IS_TERMINAL_ARRAY : numpy.ndarray of bool
    Boolean lookup array indexed by operator ID.
IS_ARITY_2_ARRAY : numpy.ndarray of bool
    Boolean lookup array indexed by operator ID.
OPERATOR_NAMES : dict {int: list of str}
    Common names for each operator (used in parsing).
"""

import numpy as np

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

# ---- Operator property sets (source of truth) ----

TERMINAL_IDS = frozenset({VARIABLE, CONSTANT, INTEGER})

ARITY_2_IDS = frozenset(
    {
        ADDITION,
        SUBTRACTION,
        MULTIPLICATION,
        DIVISION,
        POWER,
        SAFE_POWER,
    }
)

# ---- NumPy boolean lookup arrays (index by operator ID) ----

_ALL_IDS = (
    TERMINAL_IDS
    | ARITY_2_IDS
    | {SQUARE, CUBE, SQRT, ABS, EXPONENTIAL, LOGARITHM}
    | {SIN, COS, TAN, ARCSIN, ARCCOS, ARCTAN}
    | {SINH, COSH, TANH}
)
_n = max(_ALL_IDS) + 1

IS_TERMINAL_ARRAY = np.zeros(_n, dtype=bool)
IS_TERMINAL_ARRAY[list(TERMINAL_IDS)] = True

IS_ARITY_2_ARRAY = np.zeros(_n, dtype=bool)
IS_ARITY_2_ARRAY[list(ARITY_2_IDS)] = True

del _n, _ALL_IDS

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
