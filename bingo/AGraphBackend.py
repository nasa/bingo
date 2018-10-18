import numpy as np

from . import AGraphBackendOperators as Operators


def simplify_and_evaluate(stack, x, constants):
    used_commands_mask = get_utilized_commands(stack)
    forward_eval = _forward_eval_with_mask(stack, x, constants,
                                           used_commands_mask)
    return forward_eval[-1].reshape((-1, 1))


def simplify_and_evaluate_with_derivative(stack, x, constants,
                                          wrt_param_x_or_c):
    used_commands_mask = get_utilized_commands(stack)
    return _evaluate_with_derivative_and_mask(stack, x, constants,
                                              used_commands_mask,
                                              wrt_param_x_or_c)


def get_utilized_commands(stack):
    util = [False]*stack.shape[0]
    util[-1] = True
    for i in range(1, stack.shape[0]):
        operator, param1, param2 = stack[-i]
        if util[-i] and operator > 1:
            util[param1] = True
            if Operators.is_arity_2_map[operator]:
                util[param2] = True
    return util


def _forward_eval_with_mask(stack, x, constants, used_commands_mask):
    forward_eval = np.empty((stack.shape[0], x.shape[0]))
    for i, command_is_used in enumerate(used_commands_mask):
        if command_is_used:
            operator, param1, param2 = stack[i]
            forward_eval[i] = Operators.forward_eval_map[operator](param1,
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
                Operators.reverse_eval_map[operator](i, param1, param2,
                                                     forward_eval, reverse_eval)
    return derivative

