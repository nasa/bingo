"""
This module provides the python implementation of the functions for each
mathematical nodes used in Agraph

Attributes
----------
FORWARD_EVAL_MAP : dictionary {int: function}
                   A map of node number to evaluation function
REVERSE_EVAL_MAP : dictionary {int: function}
                   A map of node number to derivative evaluation function
"""

import numpy as np
np.seterr(divide='ignore', invalid='ignore')


# Load x column
def _loadx_forward_eval(param1, _param2, x, _constants, _forwardeval):
    return x[:, param1]


def _loadx_reverse_eval(_reverseindex, _param1, _param2, _forwardeval,
                        _reverseeval):
    pass


# Load constant
def _loadc_forward_eval(param1, _param2, _x, constants, _forwardeval):
    return constants[param1]


def _loadc_reverse_eval(_reverseindex, _param1, _param2, _forwardeval,
                        _reverseeval):
    pass


# Addition
def _add_forward_eval(param1, param2, _x, _constants, forward_eval):
    return forward_eval[param1] + forward_eval[param2]


def _add_reverse_eval(reverse_index, param1, param2, _forwardeval,
                      reverse_eval):
    reverse_eval[param1] += reverse_eval[reverse_index]
    reverse_eval[param2] += reverse_eval[reverse_index]


# Subtraction
def _subtract_forward_eval(param1, param2, _x, _constants, forward_eval):
    return forward_eval[param1] - forward_eval[param2]


def _subtract_reverse_eval(reverse_index, param1, param2, _forwardeval,
                           reverse_eval):
    reverse_eval[param1] += reverse_eval[reverse_index]
    reverse_eval[param2] -= reverse_eval[reverse_index]


# Multiplication
def _multiply_forward_eval(param1, param2, _x, _constants, forward_eval):
    return forward_eval[param1] * forward_eval[param2]


def _multiply_reverse_eval(reverse_index, param1, param2, forward_eval,
                           reverse_eval):
    reverse_eval[param1] += reverse_eval[reverse_index]*forward_eval[param2]
    reverse_eval[param2] += reverse_eval[reverse_index]*forward_eval[param1]


# Division
def _divide_forward_eval(param1, param2, _x, _constants, forward_eval):
    return forward_eval[param1] / forward_eval[param2]


def _divide_reverse_eval(reverse_index, param1, param2, forward_eval,
                         reverse_eval):
    reverse_eval[param1] += reverse_eval[reverse_index] / forward_eval[param2]
    reverse_eval[param2] -= reverse_eval[reverse_index] *\
                            forward_eval[reverse_index] / forward_eval[param2]


# Sine
def _sin_forward_eval(param1, _param2, _x, _constants, forward_eval):
    return np.sin(forward_eval[param1])


def _sin_reverse_eval(reverse_index, param1, _param2, forward_eval,
                      reverse_eval):
    reverse_eval[param1] += \
        reverse_eval[reverse_index] * np.cos(forward_eval[param1])


# Cosine
def _cos_forward_eval(param1, _param2, _x, _constants, forward_eval):
    return np.cos(forward_eval[param1])


def _cos_reverse_eval(reverse_index, param1, _param2, forward_eval,
                      reverse_eval):
    reverse_eval[param1] -= \
        reverse_eval[reverse_index] * np.sin(forward_eval[param1])


# Exponential
def _exp_forward_eval(param1, _param2, _x, _constants, forward_eval):
    return np.exp(forward_eval[param1])


def _exp_reverse_eval(reverse_index, param1, _param2, forward_eval,
                      reverse_eval):
    reverse_eval[param1] += reverse_eval[reverse_index] *\
                            forward_eval[reverse_index]


# Natural logarithm
def _log_forward_eval(param1, _param2, _x, _constants, forward_eval):
    return np.log(np.abs(forward_eval[param1]))


def _log_reverse_eval(reverse_index, param1, _param2, forward_eval,
                      reverse_eval):
    reverse_eval[param1] += reverse_eval[reverse_index] /\
                            forward_eval[param1]


# Power
def _pow_forward_eval(param1, param2, _x, _constants, forward_eval):
    return np.power(np.abs(forward_eval[param1]), forward_eval[param2])


def _pow_reverse_eval(reverse_index, param1, param2, forward_eval,
                      reverse_eval):
    reverse_eval[param1] += reverse_eval[reverse_index] *\
                            forward_eval[reverse_index] *\
                            forward_eval[param2] / forward_eval[param1]
    reverse_eval[param2] += reverse_eval[reverse_index] *\
                            forward_eval[reverse_index] *\
                            np.log(np.abs(forward_eval[param1]))


# Absolute value
def _abs_forward_eval(param1, _param2, _x, _constants, forward_eval):
    return np.abs(forward_eval[param1])


def _abs_reverse_eval(reverse_index, param1, _param2, forward_eval,
                      reverse_eval):
    reverse_eval[param1] += reverse_eval[reverse_index] *\
                            np.sign(forward_eval[param1])


# Square root
def _sqrt_forward_eval(param1, _param2, _x, _constants, forward_eval):
    return np.sqrt(np.abs(forward_eval[param1]))


def _sqrt_reverse_eval(reverse_index, param1, _param2, forward_eval,
                       reverse_eval):
    reverse_eval[param1] += 0.5*reverse_eval[reverse_index] /\
                            forward_eval[reverse_index] *\
                            np.sign(forward_eval[param1])


def forward_eval_function(node, param1, param2, x, constants, forward_eval):
    return FORWARD_EVAL_MAP[node](param1, param2, x, constants, forward_eval)


def reverse_eval_function(node, reverse_index, param1, param2, forward_eval,
                          reverse_eval):
    REVERSE_EVAL_MAP[node](reverse_index, param1, param2, forward_eval,
                           reverse_eval)


# Node maps
FORWARD_EVAL_MAP = {0: _loadx_forward_eval,
                    1: _loadc_forward_eval,
                    2: _add_forward_eval,
                    3: _subtract_forward_eval,
                    4: _multiply_forward_eval,
                    5: _divide_forward_eval,
                    6: _sin_forward_eval,
                    7: _cos_forward_eval,
                    8: _exp_forward_eval,
                    9: _log_forward_eval,
                    10: _pow_forward_eval,
                    11: _abs_forward_eval,
                    12: _sqrt_forward_eval}

REVERSE_EVAL_MAP = {0: _loadx_reverse_eval,
                    1: _loadc_reverse_eval,
                    2: _add_reverse_eval,
                    3: _subtract_reverse_eval,
                    4: _multiply_reverse_eval,
                    5: _divide_reverse_eval,
                    6: _sin_reverse_eval,
                    7: _cos_reverse_eval,
                    8: _exp_reverse_eval,
                    9: _log_reverse_eval,
                    10: _pow_reverse_eval,
                    11: _abs_reverse_eval,
                    12: _sqrt_reverse_eval}
