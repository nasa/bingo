"""
This module represents the python backend associated with the Agraph equation
evaluation.  This backend is used to perform the the evaluation of the equation
represented by an `AGraph`.  It can also perform derivatives.
"""

import numpy as np
import math
from cupyx import jit
import cupy as cp

from .operator_eval import forward_eval_function, reverse_eval_function
import bingo.util.global_imports as gi
import bingo.symbolic_regression.agraph.operator_definitions as defs


ENGINE = "Python"

def evaluate(stack, x, constants):
    """Evaluate an equation

    Evaluate the equation associated with an Agraph, at the values x.

    Parameters
    ----------
    stack : Nx3 numpy array of int.
        The command stack associated with an equation. N is the number of
        commands in the stack.
    x : MxD array of numeric.
        Values at which to evaluate the equations. D is the number of
        dimensions in x and M is the number of data points in x.
    constants : list-like of numeric.
        numeric constants that are used in the equation

    Returns
    -------
    Mx1 array of numeric
        :math`f(x)`
    """

    if gi.USING_GPU:
        num_particles = 1
        if hasattr(constants, 'shape'):
            num_particles = constants.shape[1]
        elif isinstance(constants, tuple):
            if len(constants) == 0:
                constants = cp.asarray([[]])
            else:
                constants = cp.stack(constants, axis=0)
                num_particles = constants.shape[1]
        forward_eval = cp.ones((len(stack), x.shape[0], num_particles), dtype=np.double) * np.inf
        blockspergrid = math.ceil(x.shape[0] * num_particles / gi.GPU_THREADS_PER_BLOCK)
        _f_eval_gpu_kernel[blockspergrid, gi.GPU_THREADS_PER_BLOCK](stack, x, constants, num_particles, x.shape[0], stack.shape[0], forward_eval)
        output = forward_eval[-1]
    else:
        forward_eval = _forward_eval(stack, x, constants)
        output = forward_eval[-1]
    return _reshape_output(output, constants, x)

def _reshape_output(output, constants, x):
    x_dim = len(x)
    c_dim = 1
    if len(constants) > 0:
        if isinstance(constants[0], np.ndarray):
            c_dim = len(constants[0])
    if isinstance(output, gi.num_lib.ndarray) and \
            output.shape == (x_dim, c_dim):
        return output

    return gi.num_lib.ones((x_dim, c_dim)) * output


def evaluate_with_derivative(stack, x, constants, wrt_param_x_or_c):
    """Evaluate equation and take derivative

    Evaluate the derivatives of the equation associated with an Agraph, at the
    values x.

    Parameters
    ----------
    stack : Nx3 numpy array of int.
        The command stack associated with an equation. N is the number of
        commands in the stack.
    x : MxD array of numeric.
        Values at which to evaluate the equations. D is the number of
        dimensions in x and M is the number of data points in x.
    constants : list-like of numeric.
        numeric constants that are used in the equation
    wrt_param_x_or_c : boolean
        Take derivative with respect to x or constants. True signifies
        derivatives are wrt x. False signifies derivatives are wrt constants.

    Returns
    -------
    MxD array of numeric or MxL array of numeric:
        Derivatives of all dimensions of x/constants at location x.
    """
    return _evaluate_with_derivative(stack, x, constants, wrt_param_x_or_c)


def _forward_eval(stack, x, constants):
    forward_eval = [None]*stack.shape[0] # np.empty((stack.shape[0], x.shape[0]))
    for i, (node, param1, param2) in enumerate(stack):
        forward_eval[i] = forward_eval_function(node, param1, param2, x,
                                                constants, forward_eval)

    return forward_eval


@jit.rawkernel()
def _f_eval_gpu_kernel(stack, x, constants, num_particles, data_size, stack_size, f_eval_arr):
    index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x

    if index < data_size * num_particles:
        data_index = index // num_particles
        constant_index = index % num_particles

        for i in range(stack_size):
            node = stack[i, 0]
            param1 = stack[i, 1]
            param2 = stack[i, 2]

            if node == defs.INTEGER:
                f_eval_arr[i, data_index, constant_index] = float(param1)
            elif node == defs.VARIABLE:
                f_eval_arr[i, data_index, constant_index] = x[data_index, param1]
            elif node == defs.CONSTANT:
                #if num_particles < 2: # case doesn't work for some reason
                #    f_eval_arr[i, data_index, constant_index] = constants[int(param1)]
                f_eval_arr[i, data_index, constant_index] = constants[int(param1), constant_index]
            elif node == defs.ADDITION:
                f_eval_arr[i, data_index, constant_index] = f_eval_arr[int(param1), data_index, constant_index] + \
                                                            f_eval_arr[int(param2), data_index, constant_index]
            elif node == defs.SUBTRACTION:
                f_eval_arr[i, data_index, constant_index] = f_eval_arr[int(param1), data_index, constant_index] - \
                                                            f_eval_arr[int(param2), data_index, constant_index]
            elif node == defs.MULTIPLICATION:
                f_eval_arr[i, data_index, constant_index] = f_eval_arr[int(param1), data_index, constant_index] * \
                                                            f_eval_arr[int(param2), data_index, constant_index]
            elif node == defs.DIVISION:
                f_eval_arr[i, data_index, constant_index] = f_eval_arr[int(param1), data_index, constant_index] / \
                                                            f_eval_arr[int(param2), data_index, constant_index]
            elif node == defs.SIN:
                f_eval_arr[i, data_index, constant_index] = cp.sin(f_eval_arr[int(param1), data_index, constant_index])
            elif node == defs.COS:
                f_eval_arr[i, data_index, constant_index] = cp.cos(f_eval_arr[int(param1), data_index, constant_index])
            elif node == defs.EXPONENTIAL:
                f_eval_arr[i, data_index, constant_index] = cp.exp(f_eval_arr[int(param1), data_index, constant_index])
            elif node == defs.LOGARITHM:
                f_eval_arr[i, data_index, constant_index] = cp.log(cp.abs(f_eval_arr[int(param1), data_index, constant_index]))
            elif node == defs.POWER:
                f_eval_arr[i, data_index, constant_index] = cp.power(f_eval_arr[int(param1), data_index, constant_index],
                                                                     f_eval_arr[int(param2), data_index, constant_index])
            elif node == defs.ABS:
                f_eval_arr[i, data_index, constant_index] = cp.abs(f_eval_arr[int(param1), data_index, constant_index])
            elif node == defs.SQRT:
                f_eval_arr[i, data_index, constant_index] = cp.sqrt(f_eval_arr[int(param1), data_index, constant_index])


def _evaluate_with_derivative(stack, x, constants, wrt_param_x_or_c):

    forward_eval = _forward_eval(stack, x, constants)

    if wrt_param_x_or_c:  # x
        deriv_shape = x.shape
        deriv_wrt_node = 0
    else:  # c
        deriv_shape = (x.shape[0], len(constants))
        deriv_wrt_node = 1

    derivative = _reverse_eval(deriv_shape, deriv_wrt_node, forward_eval,
                               stack)

    return _reshape_output(forward_eval[-1], constants, x), derivative


def _reverse_eval(deriv_shape, deriv_wrt_node, forward_eval, stack):
    derivative = np.zeros(deriv_shape)
    reverse_eval = [0] * stack.shape[0]
    reverse_eval[-1] = 1.0
    for i in range(stack.shape[0] - 1, -1, -1):
        node, param1, param2 = stack[i]
        if node == deriv_wrt_node:
            derivative[:, param1] += _reshape_reverse_eval(reverse_eval[i],
                                                           deriv_shape[0])
        else:
            reverse_eval_function(node, i, param1, param2, forward_eval,
                                  reverse_eval)
    return derivative


def _reshape_reverse_eval(r_eval, new_size):
    if isinstance(r_eval, np.ndarray):
        return r_eval.reshape((new_size, ))
    return r_eval
