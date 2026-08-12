"""Evaluation backend for AGraph expressions.

Pure-Python implementation of forward evaluation and reverse-mode
automatic differentiation for acyclic graph command stacks.
"""

import numpy as np

from .operator_eval import (
    forward_eval_function,
    get_operator_partials,
    reverse_eval_function,
)
from ..operators import VARIABLE, CONSTANT


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


def evaluate_with_const_hessian(stack, x, constants, integers):
    """Evaluate an equation with constant gradient and Hessian.

    The gradient and Hessian are calculated by differentiating the existing
    reverse-mode constant-gradient evaluation in forward mode. This computes
    all Hessian rows simultaneously while preserving the derivative
    conventions used by :func:`evaluate_with_derivative`.

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

    Returns
    -------
    tuple of (Mx1 array, MxL array, MxLxL array)
        ``(f(x), df/dc, d2f/dc2)`` where ``L = len(constants)``.
    """
    forward_eval = _forward_eval(stack, x, constants, integers)
    num_samples = x.shape[0]
    num_constants = len(constants)
    normalized_forward = [
        _reshape_node_value(value, num_samples) for value in forward_eval
    ]

    forward_tangent, constant_dependencies = _forward_tangent_eval(
        stack, normalized_forward, num_samples, num_constants
    )
    gradient, hessian = _forward_over_reverse_eval(
        stack,
        normalized_forward,
        forward_tangent,
        constant_dependencies,
        num_samples,
        num_constants,
    )
    return _reshape_output(forward_eval[-1], constants, x), gradient, hessian


def _forward_eval(stack, x, constants, integers):
    forward_eval = [None] * stack.shape[0]
    for i, (node, param1, param2) in enumerate(stack):
        forward_eval[i] = forward_eval_function(
            node, param1, param2, x, constants, integers, forward_eval
        )
    return forward_eval


def _forward_tangent_eval(stack, forward_eval, num_samples, num_constants):
    forward_tangent = [
        np.zeros((num_samples, num_constants)) for _ in range(stack.shape[0])
    ]
    constant_dependencies = np.zeros(stack.shape[0], dtype=bool)
    for i, (node, param1, param2) in enumerate(stack):
        if node == CONSTANT:
            forward_tangent[i][:, param1] = 1.0
            constant_dependencies[i] = True
            continue

        first_partial, second_partial, _, _, _ = get_operator_partials(
            node, i, param1, param2, forward_eval
        )
        if first_partial is not None and constant_dependencies[param1]:
            forward_tangent[i] = first_partial * forward_tangent[param1]
        if second_partial is not None and constant_dependencies[param2]:
            forward_tangent[i] += second_partial * forward_tangent[param2]
        constant_dependencies[i] = (
            first_partial is not None
            and constant_dependencies[param1]
        ) or (
            second_partial is not None
            and constant_dependencies[param2]
        )
    return forward_tangent, constant_dependencies


def _forward_over_reverse_eval(
    stack,
    forward_eval,
    forward_tangent,
    constant_dependencies,
    num_samples,
    num_constants,
):
    reverse_eval = [np.ones((num_samples, 1)) * 0.0 for _ in range(stack.shape[0])]
    reverse_tangent = [
        np.zeros((num_samples, num_constants)) for _ in range(stack.shape[0])
    ]
    reverse_eval[-1] = np.ones((num_samples, 1))

    gradient = np.zeros((num_samples, num_constants))
    hessian = np.zeros((num_samples, num_constants, num_constants))

    for i in range(stack.shape[0] - 1, -1, -1):
        node, param1, param2 = stack[i]
        if node == CONSTANT:
            gradient[:, param1] += reverse_eval[i][:, 0]
            hessian[:, param1, :] += reverse_tangent[i]
            continue

        partials = get_operator_partials(node, i, param1, param2, forward_eval)
        if partials[0] is not None and constant_dependencies[param1]:
            reverse_eval[param1] += reverse_eval[i] * partials[0]
            reverse_tangent[param1] += (
                reverse_tangent[i] * partials[0]
                + reverse_eval[i]
                * (
                    partials[2] * forward_tangent[param1]
                    + (
                        partials[3] * forward_tangent[param2]
                        if partials[3] is not None
                        and constant_dependencies[param2]
                        else 0.0
                    )
                )
            )
        if partials[1] is not None and constant_dependencies[param2]:
            reverse_eval[param2] += reverse_eval[i] * partials[1]
            reverse_tangent[param2] += (
                reverse_tangent[i] * partials[1]
                + reverse_eval[i]
                * (
                    (
                        partials[3] * forward_tangent[param1]
                        if constant_dependencies[param1]
                        else 0.0
                    )
                    + partials[4] * forward_tangent[param2]
                )
            )

    return gradient, hessian


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


def _reshape_node_value(value, num_samples):
    if isinstance(value, np.ndarray) and value.shape == (num_samples, 1):
        return value
    return np.ones((num_samples, 1)) * value


def _reshape_output(output, constants, x):
    x_dim = len(x)
    c_dim = 1
    if len(constants) > 0:
        if isinstance(constants[0], np.ndarray):
            c_dim = len(constants[0])
    if isinstance(output, np.ndarray) and output.shape == (x_dim, c_dim):
        return output
    return np.ones((x_dim, c_dim)) * output
