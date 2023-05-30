import torch

from bingo.symbolic_regression.agraph.operator_definitions import *


# Integer value
def _integer_forward_eval(param1, _param2, x, _constants, _forwardeval):
    return torch.ones(x.size(1), 1) * int(param1)


# Load x column
def _loadx_forward_eval(param1, _param2, x, _constants, _forwardeval):
    return x[param1].view(x.size(1), 1)


# Load constant
def _loadc_forward_eval(param1, _param2, x, constants, _forwardeval):
    return constants[param1]


# Addition
def _add_forward_eval(param1, param2, _x, _constants, forward_eval):
    return forward_eval[param1] + forward_eval[param2]


# Subtraction
def _subtract_forward_eval(param1, param2, _x, _constants, forward_eval):
    return forward_eval[param1] - forward_eval[param2]


# Multiplication
def _multiply_forward_eval(param1, param2, _x, _constants, forward_eval):
    return forward_eval[param1] * forward_eval[param2]


# Division
def _divide_forward_eval(param1, param2, _x, _constants, forward_eval):
    return forward_eval[param1] / forward_eval[param2]


# Sine
def _sin_forward_eval(param1, _param2, _x, _constants, forward_eval):
    return torch.sin(forward_eval[param1])


# Cosine
def _cos_forward_eval(param1, _param2, _x, _constants, forward_eval):
    return torch.cos(forward_eval[param1])


# Hyperbolic Sine
def _sinh_forward_eval(param1, _param2, _x, _constants, forward_eval):
    return torch.sinh(forward_eval[param1])


# Hyperbolic Cosine
def _cosh_forward_eval(param1, _param2, _x, _constants, forward_eval):
    return torch.cosh(forward_eval[param1])


# Exponential
def _exp_forward_eval(param1, _param2, _x, _constants, forward_eval):
    return torch.exp(forward_eval[param1])


# Natural logarithm
def _log_forward_eval(param1, _param2, _x, _constants, forward_eval):
    return torch.log(torch.abs(forward_eval[param1]))


# Power
def _pow_forward_eval(param1, param2, _x, _constants, forward_eval):
    return torch.pow(forward_eval[param1], forward_eval[param2])


# Safe Power
def _safe_pow_forward_eval(param1, param2, _x, _constants, forward_eval):
    return torch.pow(torch.abs(forward_eval[param1]), forward_eval[param2])


# Absolute value
def _abs_forward_eval(param1, _param2, _x, _constants, forward_eval):
    return torch.abs(forward_eval[param1])


# Square root
def _sqrt_forward_eval(param1, _param2, _x, _constants, forward_eval):
    return torch.sqrt(torch.abs(forward_eval[param1]))


def forward_eval_function(node, param1, param2, x, constants, forward_eval):
    """Performs calculation of one line of stack"""
    # IMPORTANT: Assumes x is column-major
    return FORWARD_EVAL_MAP[node](param1, param2, x, constants, forward_eval)


# Node maps
FORWARD_EVAL_MAP = {INTEGER: _integer_forward_eval,
                    VARIABLE: _loadx_forward_eval,
                    CONSTANT: _loadc_forward_eval,
                    ADDITION: _add_forward_eval,
                    SUBTRACTION: _subtract_forward_eval,
                    MULTIPLICATION: _multiply_forward_eval,
                    DIVISION: _divide_forward_eval,
                    SIN: _sin_forward_eval,
                    COS: _cos_forward_eval,
                    SINH: _sinh_forward_eval,
                    COSH: _cosh_forward_eval,
                    EXPONENTIAL: _exp_forward_eval,
                    LOGARITHM: _log_forward_eval,
                    POWER: _pow_forward_eval,
                    ABS: _abs_forward_eval,
                    SQRT: _sqrt_forward_eval,
                    SAFE_POWER: _safe_pow_forward_eval}
