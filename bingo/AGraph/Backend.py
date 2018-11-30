"""
This module represents the python backend associated with the Agraph equation
representation.  The backend is used to perform the most computationally
demanding functions required by the Agraph.
"""

import numpy as np

from bingo.AGraph import BackendOperators as Operators


def simplify_and_evaluate(stack, x, constants):
    """Evaluate an equation after simplification.

    Evauluate the equation associated with an Agraph, at the values x.
    Simplification ensures that only the commands utilized in the result are
    considered.

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
    used_commands_mask = get_utilized_commands(stack)
    forward_eval = _forward_eval_with_mask(stack, x, constants,
                                           used_commands_mask)
    return forward_eval[-1].reshape((-1, 1))


def simplify_and_evaluate_with_derivative(stack, x, constants,
                                          wrt_param_x_or_c):
    """Evaluate equation and take derivative

    Evauluate the derivatives of the equation associated with an Agraph, at the
    values x.  Simplification ensures that only the commands utilized in the
    result are considered.

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
                       Take derivative with respect to x or constants. True
                       signifies derivatives are wrt x. False signifies
                       derivatives are wrt constants.

    Returns
    -------
    MxD array of numeric or MxL array of numeric.
        Derivatives of all dimensions of x/constants at location x.
    """
    used_commands_mask = get_utilized_commands(stack)
    return _evaluate_with_derivative_and_mask(stack, x, constants,
                                              used_commands_mask,
                                              wrt_param_x_or_c)


def get_utilized_commands(stack):
    """Find which commands are utilized.

    Find the commands (rows) of the stack upon which the last command of the
    stack depends. This is inclusive of the last command.

    Parameters
    ----------
    stack : Nx3 numpy array of int.
            The command stack associated with an equation. N is the number of
            commands in the stack.

    Returns
    -------
    list of bool of length N
        Boolean values for whether each command is utilized.
    """
    util = [False]*stack.shape[0]
    util[-1] = True
    for i in range(1, stack.shape[0]):
        operator, param1, param2 = stack[-i]
        if util[-i] and operator > 1:
            util[param1] = True
            if Operators.IS_ARITY_2_MAP[operator]:
                util[param2] = True
    return util


def _forward_eval_with_mask(stack, x, constants, used_commands_mask):
    forward_eval = np.empty((stack.shape[0], x.shape[0]))
    for i, command_is_used in enumerate(used_commands_mask):
        if command_is_used:
            operator, param1, param2 = stack[i]
            forward_eval[i] = Operators.FORWARD_EVAL_MAP[operator](param1,
                                                                   param2,
                                                                   x,
                                                                   constants,
                                                                   forward_eval)
    return forward_eval


def _evaluate_with_derivative_and_mask(stack, x, constants, used_commands_mask,
                                       wrt_param_x_or_c):

    forward_eval = _forward_eval_with_mask(stack, x, constants,
                                           used_commands_mask)

    if wrt_param_x_or_c:  # x
        deriv_shape = x.shape
        deriv_wrt_operator = 0
    else:  # c
        deriv_shape = (x.shape[0], len(constants))
        deriv_wrt_operator = 1

    derivative = _reverse_eval_with_mask(deriv_shape, deriv_wrt_operator,
                                         forward_eval, stack,
                                         used_commands_mask)

    return forward_eval[-1].reshape((-1, 1)), derivative


def _reverse_eval_with_mask(deriv_shape, deriv_wrt_operator, forward_eval,
                            stack, used_commands_mask):
    derivative = np.zeros(deriv_shape)
    reverse_eval = [0] * stack.shape[0]
    reverse_eval[-1] = 1.0
    for i in range(stack.shape[0] - 1, -1, -1):
        if used_commands_mask[i]:
            operator, param1, param2 = stack[i]
            if operator == deriv_wrt_operator:
                derivative[:, param1] += reverse_eval[i]
            else:
                Operators.REVERSE_EVAL_MAP[operator](i, param1, param2,
                                                     forward_eval, reverse_eval)
    return derivative
