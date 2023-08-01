"""
This module represents the python backend associated with the Agraph equation
evaluation.  This backend is used to perform the the evaluation of the equation
represented by an `AGraph`.  It can also perform derivatives.
"""
import numpy as np

import torch
from torch.func import hessian as get_hessian
from torch_eval import evaluate as cpp_evaluate
from torch_eval import evaluate_with_deriv as cpp_evaluate_with_deriv

ENGINE = "pytorch_cpp"


def _get_torch_const(constants, data_len):
    with torch.no_grad():
        constants = torch.from_numpy(np.array(constants)).double()
        if len(constants) > 0:
            if constants.ndim == 1:
                constants = constants.unsqueeze(1)
            constants = constants.unsqueeze(2).expand(-1, -1, data_len).mT
        return constants


def _reshape_output(output, constants, x):
    x_dim = x.size(1)
    c_dim = 1
    if len(constants) > 0:
        if isinstance(constants[0], np.ndarray):
            c_dim = len(constants[0])
    if isinstance(output, np.ndarray):
        if output.shape == (x_dim, c_dim):
            return output
        elif output.shape == (x_dim,):
            return output.reshape((x_dim, 1))
    return np.ones((x_dim, c_dim)) * output


def evaluate(cmd_arr, x, constants):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.T).double()
    torch_constants = torch.from_numpy(np.array(constants)).double()
    evaluation = cpp_evaluate(cmd_arr, x, torch_constants).detach().numpy()
    return _reshape_output(evaluation, constants, x)


def evaluate_with_derivative(cmd_arr, x, constants, wrt_param_x_or_c):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.T).double()
    torch_constants = _get_torch_const(constants, x.size(1))
    eval, deriv = cpp_evaluate_with_deriv(cmd_arr, x, torch_constants, wrt_param_x_or_c)
    if deriv.ndim == 3:
        deriv = deriv[:, :, 0]
    return _reshape_output(eval.detach().numpy(), constants, x), deriv.T.detach().numpy()


def evaluate_with_hessian(cmd_arr, x, constants, wrt_param_x_or_c):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.T).double()
    evaluation = evaluate(cmd_arr, x, constants)

    torch_constants = torch.from_numpy(np.array(constants)).double()

    deriv_argnum = 2  # wrt c
    if wrt_param_x_or_c:  # wrt x
        deriv_argnum = 1
    hessian = get_hessian(cpp_evaluate, argnums=deriv_argnum)(cmd_arr, x,
                                                              torch_constants)

    return evaluation, hessian
