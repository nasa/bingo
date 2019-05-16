"""
This module represents the python backend associated with the Agraph equation
representation.  The backend is used to perform the most computationally
demanding functions required by the Agraph.
"""

import numpy as np

from . import AGraph
from . import BackendNodes as Nodes


def is_cpp():
    """Identify whether the backend is C++

    Returns
    -------
    bool :
        false, the backend is not c++
    """
    return False


def evaluate(stack, x, constants):
    """Evaluate an equation

    Evauluate the equation associated with an Agraph, at the values x.

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
    forward_eval = _forward_eval(stack, x, constants)
    return forward_eval[-1].reshape((-1, 1))


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

    Evaluate the derivatives of the equation associated with an Agraph, at the
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
                       Take derivative with respect to x or constants. True
                       signifies derivatives are wrt x. False signifies
                       derivatives are wrt constants.

    Returns
    -------
    MxD array of numeric or MxL array of numeric.
        Derivatives of all dimensions of x/constants at location x.
    """
    return _evaluate_with_derivative(stack, x, constants, wrt_param_x_or_c)


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
        node, param1, param2 = stack[-i]
        if util[-i] and node > 1:
            util[param1] = True
            if AGraph.IS_ARITY_2_MAP[node]:
                util[param2] = True
    return util


def simplify_stack(stack):
    """Simplifies a stack.

    An acyclic graph is given in stack form.  The stack is first simplified to
    consist only of the commands used by the last command.

    Parameters
    ----------
    stack : Nx3 numpy array of int.
            The command stack associated with an equation. N is the number of
            commands in the stack.

    Returns
    -------
    Mx3 numpy array of int. :
        a simplified stack where M is the number of  used commands
    """
    used_commands = get_utilized_commands(stack)
    reduced_param_map = {}
    num_commands = np.sum(used_commands)
    new_stack = np.empty((num_commands, 3), int)

    j = 0
    for i, (node, param1, param2) in enumerate(stack):
        if used_commands[i]:
            new_stack[j, 0] = node
            if AGraph.IS_TERMINAL_MAP[node]:
                new_stack[j, 1] = param1
                new_stack[j, 2] = param2
            else:
                new_stack[j, 1] = reduced_param_map[param1]
                if AGraph.IS_ARITY_2_MAP[node]:
                    new_stack[j, 2] = reduced_param_map[param2]
                else:
                    new_stack[j, 2] = new_stack[j, 1]
            reduced_param_map[i] = j
            j += 1
    return new_stack


def _forward_eval(stack, x, constants):
    forward_eval = np.empty((stack.shape[0], x.shape[0]))
    for i, (node, param1, param2) in enumerate(stack):
        forward_eval[i] = Nodes.forward_eval_function(node,
                                                      param1,
                                                      param2,
                                                      x,
                                                      constants,
                                                      forward_eval)
    return forward_eval


def _forward_eval_with_mask(stack, x, constants, used_commands_mask):
    forward_eval = np.empty((stack.shape[0], x.shape[0]))
    for i, command_is_used in enumerate(used_commands_mask):
        if command_is_used:
            node, param1, param2 = stack[i]
            forward_eval[i] = Nodes.forward_eval_function(node,
                                                          param1,
                                                          param2,
                                                          x,
                                                          constants,
                                                          forward_eval)
    return forward_eval


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

    return forward_eval[-1].reshape((-1, 1)), derivative


def _evaluate_with_derivative_and_mask(stack, x, constants, used_commands_mask,
                                       wrt_param_x_or_c):

    forward_eval = _forward_eval_with_mask(stack, x, constants,
                                           used_commands_mask)

    if wrt_param_x_or_c:  # x
        deriv_shape = x.shape
        deriv_wrt_node = 0
    else:  # c
        deriv_shape = (x.shape[0], len(constants))
        deriv_wrt_node = 1

    derivative = _reverse_eval_with_mask(deriv_shape, deriv_wrt_node,
                                         forward_eval, stack,
                                         used_commands_mask)

    return forward_eval[-1].reshape((-1, 1)), derivative


def _reverse_eval(deriv_shape, deriv_wrt_node, forward_eval, stack):
    derivative = np.zeros(deriv_shape)
    reverse_eval = [0] * stack.shape[0]
    reverse_eval[-1] = 1.0
    for i in range(stack.shape[0] - 1, -1, -1):
        node, param1, param2 = stack[i]
        if node == deriv_wrt_node:
            derivative[:, param1] += reverse_eval[i]
        else:
            Nodes.reverse_eval_function(node, i, param1, param2,
                                        forward_eval, reverse_eval)
    return derivative


def _reverse_eval_with_mask(deriv_shape, deriv_wrt_node, forward_eval,
                            stack, used_commands_mask):
    derivative = np.zeros(deriv_shape)
    reverse_eval = [0] * stack.shape[0]
    reverse_eval[-1] = 1.0
    for i in range(stack.shape[0] - 1, -1, -1):
        if used_commands_mask[i]:
            node, param1, param2 = stack[i]
            if node == deriv_wrt_node:
                derivative[:, param1] += reverse_eval[i]
            else:
                Nodes.reverse_eval_function(node, i, param1, param2,
                                            forward_eval, reverse_eval)
    return derivative
