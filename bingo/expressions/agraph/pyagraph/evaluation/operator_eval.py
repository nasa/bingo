"""Per-operator forward and reverse evaluation functions.

Each operator has a ``_<name>_forward_eval`` function that computes its
value during the forward pass, and a ``_<name>_reverse_eval`` function
that propagates adjoints during the reverse (derivative) pass.

Forward eval signature::

    f(param1, param2, x, constants, integers, forward_eval) -> value

Reverse eval signature::

    f(reverse_index, param1, param2, forward_eval, reverse_eval) -> None

Attributes
----------
FORWARD_EVAL_MAP : dict {int: callable}
    Map of operator ID to forward evaluation function.
REVERSE_EVAL_MAP : dict {int: callable}
    Map of operator ID to reverse evaluation function.
"""

import numpy as np

from ..operators import (
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
)

np.seterr(divide="ignore", invalid="ignore", over="ignore")


# ---- Terminals ----


def _variable_forward_eval(param1, _p2, x, _constants, _integers, _fwd):
    return x[:, param1].reshape((-1, 1))


def _variable_reverse_eval(_ri, _p1, _p2, _fwd, _rev):
    pass


def _constant_forward_eval(param1, _p2, _x, constants, _integers, _fwd):
    return constants[param1]


def _constant_reverse_eval(_ri, _p1, _p2, _fwd, _rev):
    pass


def _integer_forward_eval(param1, _p2, _x, _constants, integers, _fwd):
    return float(integers[param1])


def _integer_reverse_eval(_ri, _p1, _p2, _fwd, _rev):
    pass


# ---- Arithmetic ----


def _add_forward_eval(param1, param2, _x, _c, _i, fwd):
    return fwd[param1] + fwd[param2]


def _add_reverse_eval(ri, param1, param2, _fwd, rev):
    rev[param1] += rev[ri]
    rev[param2] += rev[ri]


def _subtract_forward_eval(param1, param2, _x, _c, _i, fwd):
    return fwd[param1] - fwd[param2]


def _subtract_reverse_eval(ri, param1, param2, _fwd, rev):
    rev[param1] += rev[ri]
    rev[param2] -= rev[ri]


def _multiply_forward_eval(param1, param2, _x, _c, _i, fwd):
    return fwd[param1] * fwd[param2]


def _multiply_reverse_eval(ri, param1, param2, fwd, rev):
    rev[param1] += rev[ri] * fwd[param2]
    rev[param2] += rev[ri] * fwd[param1]


def _divide_forward_eval(param1, param2, _x, _c, _i, fwd):
    return fwd[param1] / fwd[param2]


def _divide_reverse_eval(ri, param1, param2, fwd, rev):
    rev[param1] += rev[ri] / fwd[param2]
    rev[param2] -= rev[ri] * fwd[ri] / fwd[param2]


# ---- Power / Root ----


def _pow_forward_eval(param1, param2, _x, _c, _i, fwd):
    return np.power(fwd[param1], fwd[param2])


def _pow_reverse_eval(ri, param1, param2, fwd, rev):
    rev[param1] += rev[ri] * fwd[ri] * fwd[param2] / fwd[param1]
    rev[param2] += rev[ri] * fwd[ri] * np.log(fwd[param1])


def _safe_pow_forward_eval(param1, param2, _x, _c, _i, fwd):
    return np.power(np.abs(fwd[param1]), fwd[param2])


def _safe_pow_reverse_eval(ri, param1, param2, fwd, rev):
    rev[param1] += rev[ri] * fwd[ri] * fwd[param2] / fwd[param1]
    rev[param2] += rev[ri] * fwd[ri] * np.log(np.abs(fwd[param1]))


def _square_forward_eval(param1, _p2, _x, _c, _i, fwd):
    return fwd[param1] ** 2


def _square_reverse_eval(ri, param1, _p2, fwd, rev):
    rev[param1] += rev[ri] * 2 * fwd[param1]


def _cube_forward_eval(param1, _p2, _x, _c, _i, fwd):
    return fwd[param1] ** 3


def _cube_reverse_eval(ri, param1, _p2, fwd, rev):
    rev[param1] += rev[ri] * 3 * (fwd[param1] ** 2)


def _sqrt_forward_eval(param1, _p2, _x, _c, _i, fwd):
    return np.sqrt(np.abs(fwd[param1]))


def _sqrt_reverse_eval(ri, param1, _p2, fwd, rev):
    rev[param1] += 0.5 * rev[ri] / fwd[ri] * np.sign(fwd[param1])


# ---- Miscellaneous ----


def _abs_forward_eval(param1, _p2, _x, _c, _i, fwd):
    return np.abs(fwd[param1])


def _abs_reverse_eval(ri, param1, _p2, fwd, rev):
    rev[param1] += rev[ri] * np.sign(fwd[param1])


# ---- Exponential / Logarithmic ----


def _exp_forward_eval(param1, _p2, _x, _c, _i, fwd):
    return np.exp(fwd[param1])


def _exp_reverse_eval(ri, param1, _p2, fwd, rev):
    rev[param1] += rev[ri] * fwd[ri]


def _log_forward_eval(param1, _p2, _x, _c, _i, fwd):
    return np.log(np.abs(fwd[param1]))


def _log_reverse_eval(ri, param1, _p2, fwd, rev):
    rev[param1] += rev[ri] / fwd[param1]


# ---- Trigonometric ----


def _sin_forward_eval(param1, _p2, _x, _c, _i, fwd):
    return np.sin(fwd[param1])


def _sin_reverse_eval(ri, param1, _p2, fwd, rev):
    rev[param1] += rev[ri] * np.cos(fwd[param1])


def _cos_forward_eval(param1, _p2, _x, _c, _i, fwd):
    return np.cos(fwd[param1])


def _cos_reverse_eval(ri, param1, _p2, fwd, rev):
    rev[param1] -= rev[ri] * np.sin(fwd[param1])


def _tan_forward_eval(param1, _p2, _x, _c, _i, fwd):
    return np.tan(fwd[param1])


def _tan_reverse_eval(ri, param1, _p2, fwd, rev):
    rev[param1] += rev[ri] / (np.cos(fwd[param1]) ** 2)


def _arcsin_forward_eval(param1, _p2, _x, _c, _i, fwd):
    return np.arcsin(fwd[param1])


def _arcsin_reverse_eval(ri, param1, _p2, fwd, rev):
    xsqr = fwd[param1] ** 2
    rev[param1] += rev[ri] / np.sqrt(np.ones_like(xsqr) - xsqr)


def _arccos_forward_eval(param1, _p2, _x, _c, _i, fwd):
    return np.arccos(fwd[param1])


def _arccos_reverse_eval(ri, param1, _p2, fwd, rev):
    xsqr = fwd[param1] ** 2
    rev[param1] -= rev[ri] / np.sqrt(np.ones_like(xsqr) - xsqr)


def _arctan_forward_eval(param1, _p2, _x, _c, _i, fwd):
    return np.arctan(fwd[param1])


def _arctan_reverse_eval(ri, param1, _p2, fwd, rev):
    xsqr = fwd[param1] ** 2
    rev[param1] += rev[ri] / (np.ones_like(xsqr) + xsqr)


# ---- Hyperbolic ----


def _sinh_forward_eval(param1, _p2, _x, _c, _i, fwd):
    return np.sinh(fwd[param1])


def _sinh_reverse_eval(ri, param1, _p2, fwd, rev):
    rev[param1] += rev[ri] * np.cosh(fwd[param1])


def _cosh_forward_eval(param1, _p2, _x, _c, _i, fwd):
    return np.cosh(fwd[param1])


def _cosh_reverse_eval(ri, param1, _p2, fwd, rev):
    rev[param1] += rev[ri] * np.sinh(fwd[param1])


def _tanh_forward_eval(param1, _p2, _x, _c, _i, fwd):
    return np.tanh(fwd[param1])


def _tanh_reverse_eval(ri, param1, _p2, fwd, rev):
    rev[param1] += rev[ri] / (np.cosh(fwd[param1]) ** 2)


# ---- Dispatch functions ----


def forward_eval_function(node, param1, param2, x, constants, integers, forward_eval):
    """Evaluate one row of the command stack (forward pass)."""
    return FORWARD_EVAL_MAP[node](param1, param2, x, constants, integers, forward_eval)


def reverse_eval_function(
    node, reverse_index, param1, param2, forward_eval, reverse_eval
):
    """Evaluate one row of the command stack (reverse/derivative pass)."""
    REVERSE_EVAL_MAP[node](reverse_index, param1, param2, forward_eval, reverse_eval)


# ---- Maps ----

FORWARD_EVAL_MAP = {
    VARIABLE: _variable_forward_eval,
    CONSTANT: _constant_forward_eval,
    INTEGER: _integer_forward_eval,
    ADDITION: _add_forward_eval,
    SUBTRACTION: _subtract_forward_eval,
    MULTIPLICATION: _multiply_forward_eval,
    DIVISION: _divide_forward_eval,
    POWER: _pow_forward_eval,
    SAFE_POWER: _safe_pow_forward_eval,
    SQUARE: _square_forward_eval,
    CUBE: _cube_forward_eval,
    SQRT: _sqrt_forward_eval,
    ABS: _abs_forward_eval,
    EXPONENTIAL: _exp_forward_eval,
    LOGARITHM: _log_forward_eval,
    SIN: _sin_forward_eval,
    COS: _cos_forward_eval,
    TAN: _tan_forward_eval,
    ARCSIN: _arcsin_forward_eval,
    ARCCOS: _arccos_forward_eval,
    ARCTAN: _arctan_forward_eval,
    SINH: _sinh_forward_eval,
    COSH: _cosh_forward_eval,
    TANH: _tanh_forward_eval,
}

REVERSE_EVAL_MAP = {
    VARIABLE: _variable_reverse_eval,
    CONSTANT: _constant_reverse_eval,
    INTEGER: _integer_reverse_eval,
    ADDITION: _add_reverse_eval,
    SUBTRACTION: _subtract_reverse_eval,
    MULTIPLICATION: _multiply_reverse_eval,
    DIVISION: _divide_reverse_eval,
    POWER: _pow_reverse_eval,
    SAFE_POWER: _safe_pow_reverse_eval,
    SQUARE: _square_reverse_eval,
    CUBE: _cube_reverse_eval,
    SQRT: _sqrt_reverse_eval,
    ABS: _abs_reverse_eval,
    EXPONENTIAL: _exp_reverse_eval,
    LOGARITHM: _log_reverse_eval,
    SIN: _sin_reverse_eval,
    COS: _cos_reverse_eval,
    TAN: _tan_reverse_eval,
    ARCSIN: _arcsin_reverse_eval,
    ARCCOS: _arccos_reverse_eval,
    ARCTAN: _arctan_reverse_eval,
    SINH: _sinh_reverse_eval,
    COSH: _cosh_reverse_eval,
    TANH: _tanh_reverse_eval,
}
