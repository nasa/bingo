"""Evaluation backend for AGraph expressions.

Pure-Python implementation of forward evaluation and reverse-mode
automatic differentiation for acyclic graph command stacks.
"""

import numpy as np

from .operator_eval import forward_eval_function, reverse_eval_function
from .operators import VARIABLE, CONSTANT


def evaluate(stack, x, constants, integers):
    """Evaluate an equation represented by a command stack.

    Parameters
    ----------
    stack : Nx3 numpy array of uint8
        The command stack. N is the number of commands.
    x : MxD array of numeric
        Input data. M data points, D dimensions.
    constants : tuple of numeric
        Numeric constants used in the equation.
    integers : tuple of int
        Integer values used in the equation.

    Returns
    -------
    Mx1 array of numeric
        f(x)
    """
    forward_eval = _forward_eval(stack, x, constants, integers)
    return _reshape_output(forward_eval[-1], constants, x)


def evaluate_with_derivative(stack, x, constants, integers, wrt_param_x_or_c):
    """Evaluate equation and compute derivative via reverse-mode autodiff.

    Parameters
    ----------
    stack : Nx3 numpy array of uint8
        The command stack.
    x : MxD array of numeric
        Input data.
    constants : tuple of numeric
        Numeric constants used in the equation.
    integers : tuple of int
        Integer values used in the equation.
    wrt_param_x_or_c : bool
        True for derivatives w.r.t. x, False for w.r.t. constants.

    Returns
    -------
    tuple of (Mx1 array, MxD or MxL array)
        (f(x), df/dx) or (f(x), df/dc)
    """
    forward_eval = _forward_eval(stack, x, constants, integers)

    if wrt_param_x_or_c:  # w.r.t. x
        deriv_shape = x.shape
        deriv_wrt_node = VARIABLE
    else:  # w.r.t. constants
        deriv_shape = (x.shape[0], len(constants))
        deriv_wrt_node = CONSTANT

    derivative = _reverse_eval(deriv_shape, deriv_wrt_node, forward_eval, stack)

    return _reshape_output(forward_eval[-1], constants, x), derivative


def _forward_eval(stack, x, constants, integers):
    forward_eval = [None] * stack.shape[0]
    for i, (node, param1, param2) in enumerate(stack):
        forward_eval[i] = forward_eval_function(
            node, param1, param2, x, constants, integers, forward_eval
        )
    return forward_eval


def _reverse_eval(deriv_shape, deriv_wrt_node, forward_eval, stack):
    derivative = np.zeros(deriv_shape)
    reverse_eval = [0] * stack.shape[0]
    reverse_eval[-1] = 1.0
    for i in range(stack.shape[0] - 1, -1, -1):
        node, param1, param2 = stack[i]
        if node == deriv_wrt_node:
            derivative[:, param1] += _reshape_reverse_eval(
                reverse_eval[i], deriv_shape[0]
            )
        else:
            reverse_eval_function(node, i, param1, param2, forward_eval, reverse_eval)
    return derivative


def _reshape_reverse_eval(r_eval, new_size):
    if isinstance(r_eval, np.ndarray):
        return r_eval.reshape((new_size,))
    return r_eval


def _reshape_output(output, constants, x):
    x_dim = len(x)
    c_dim = 1
    if len(constants) > 0:
        if isinstance(constants[0], np.ndarray):
            c_dim = len(constants[0])
    if isinstance(output, np.ndarray) and output.shape == (x_dim, c_dim):
        return output
    return np.ones((x_dim, c_dim)) * output
