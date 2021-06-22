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
from numba import jit, prange, cuda
from math import prod, ceil

np.seterr(divide='ignore', invalid='ignore')

USE_GPU_FLAG = False
GPU_THREADS_PER_BLOCK = 256

from bingo.symbolic_regression.agraph.operator_definitions import *


# Integer value
def _integer_forward_eval(param1, _param2, _x, _constants, _forwardeval):
    return float(param1)


def _integer_reverse_eval(_reverseindex, _param1, _param2, _forwardeval,
                         _reverseeval):
    pass


# Load x column
def _loadx_forward_eval(param1, _param2, x, _constants, _forwardeval):
    return x[:, param1].reshape((-1, 1))


def _loadx_reverse_eval(_reverseindex, _param1, _param2, _forwardeval,
                        _reverseeval):
    pass


# Load constant
def _loadc_forward_eval(param1, _param2, _x, constants, _forwardeval):
    return constants[param1]


def _loadc_reverse_eval(_reverseindex, _param1, _param2, _forwardeval,
                        _reverseeval):
    pass

@cuda.jit
def _add_gpu_kernel(elem1, elem2, result):
    index = cuda.grid(1)

    if index < result.shape[0]:
        result[index] = elem1[index] + elem2[index]

@cuda.jit
def _sub_gpu_kernel(elem1, elem2, result):
    index = cuda.grid(1)

    if index < result.shape[0]:
        result[index] = elem1[index] - elem2[index]

# Addition
def _add_forward_eval(param1, param2, _x, _constants, forward_eval):
    elem1 = forward_eval[param1]
    elem2 = forward_eval[param2]

    if not USE_GPU_FLAG:
        print("not using gpu")
        return elem1 + elem2

    else:
        print("using gpu")
        result = np.zeros(prod(elem1.shape))
        blockspergrid = ceil(prod(result.shape) / GPU_THREADS_PER_BLOCK)
        _add_gpu_kernel[blockspergrid, GPU_THREADS_PER_BLOCK](elem1.flatten(), elem2.flatten(), result)
        return result.reshape(elem1.shape)

    """
    if isinstance(elem1, np.ndarray) and isinstance(elem2, np.ndarray):
        if len(elem1.shape) < len(elem2.shape):
            elem1 = np.resize(elem1, elem2.shape)
        elif len(elem2.shape) < len(elem1.shape):
            elem2 = np.resize(elem2, elem1.shape)
            print("resized")
        if not elem1.shape == elem2.shape:
            raise RuntimeError("Error: matrix dimensions {} and {} are not compatible with addition".format(elem1.shape, elem2.shape))

        result = np.zeros(elem1.shape)
        _elementwise_op_gpu_helper(elem1, elem2)
        return result
    elif isinstance(elem1, np.ndarray):
        result = np.zeros(elem1.shape)
        _elementwise_op_gpu_helper(elem1, elem2)
        return result
    elif isinstance(elem2, np.ndarray):
        result = np.zeros(elem2.shape)
        _elementwise_op_gpu_helper(elem2, elem1, lambda a, b, i : a[i] + b, result)
        return result
    else:
        return elem1 + elem2
        
    """


def _add_reverse_eval(reverse_index, param1, param2, _forwardeval,
                      reverse_eval):
    reverse_eval[param1] += reverse_eval[reverse_index]
    reverse_eval[param2] += reverse_eval[reverse_index]


# Subtraction
def _subtract_forward_eval(param1, param2, _x, _constants, forward_eval):
    elem1 = forward_eval[param1]
    elem2 = forward_eval[param2]

    if not USE_GPU_FLAG:
        return elem1 - elem2

    else:
        result = np.zeros(prod(elem1.shape))
        blockspergrid = ceil(prod(result.shape) / GPU_THREADS_PER_BLOCK)
        _sub_gpu_kernel[blockspergrid, GPU_THREADS_PER_BLOCK](elem1.flatten(), elem2.flatten(), result)
        return result.reshape(elem1.shape)


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


# Hyperbolic Sine
def _sinh_forward_eval(param1, _param2, _x, _constants, forward_eval):
    return np.sinh(forward_eval[param1])


def _sinh_reverse_eval(reverse_index, param1, _param2, forward_eval,
                      reverse_eval):
    reverse_eval[param1] += \
        reverse_eval[reverse_index] * np.cosh(forward_eval[param1])


# Hyperbolic Cosine
def _cosh_forward_eval(param1, _param2, _x, _constants, forward_eval):
    return np.cosh(forward_eval[param1])


def _cosh_reverse_eval(reverse_index, param1, _param2, forward_eval,
                      reverse_eval):
    reverse_eval[param1] += \
        reverse_eval[reverse_index] * np.sinh(forward_eval[param1])

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
    return np.power(forward_eval[param1], forward_eval[param2])


def _pow_reverse_eval(reverse_index, param1, param2, forward_eval,
                      reverse_eval):
    reverse_eval[param1] += reverse_eval[reverse_index] *\
                            forward_eval[reverse_index] *\
                            forward_eval[param2] / forward_eval[param1]
    reverse_eval[param2] += reverse_eval[reverse_index] *\
                            forward_eval[reverse_index] *\
                            np.log(forward_eval[param1])


# Safe Power
def _safe_pow_forward_eval(param1, param2, _x, _constants, forward_eval):
    return np.power(np.abs(forward_eval[param1]), forward_eval[param2])


def _safe_pow_reverse_eval(reverse_index, param1, param2, forward_eval,
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
    """Performs calculation of one line of stack"""
    return FORWARD_EVAL_MAP[node](param1, param2, x, constants, forward_eval)


def reverse_eval_function(node, reverse_index, param1, param2, forward_eval,
                          reverse_eval):
    """Performs calculation of one line of stack for derivative calculation"""
    REVERSE_EVAL_MAP[node](reverse_index, param1, param2, forward_eval,
                           reverse_eval)


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

REVERSE_EVAL_MAP = {INTEGER: _integer_reverse_eval,
                    VARIABLE: _loadx_reverse_eval,
                    CONSTANT: _loadc_reverse_eval,
                    ADDITION: _add_reverse_eval,
                    SUBTRACTION: _subtract_reverse_eval,
                    MULTIPLICATION: _multiply_reverse_eval,
                    DIVISION: _divide_reverse_eval,
                    SIN: _sin_reverse_eval,
                    COS: _cos_reverse_eval,
                    SINH: _sinh_reverse_eval,
                    COSH: _cosh_reverse_eval,
                    EXPONENTIAL: _exp_reverse_eval,
                    LOGARITHM: _log_reverse_eval,
                    POWER: _pow_reverse_eval,
                    ABS: _abs_reverse_eval,
                    SQRT: _sqrt_reverse_eval,
                    SAFE_POWER: _safe_pow_reverse_eval}
