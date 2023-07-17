/*
#include <torch/torch.h>
#include <chrono>
*/
#include <torch/extension.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include <iostream>
#include <vector>

typedef torch::Tensor (
  *torch_op_fn)(
    int, int, const torch::Tensor &,
    const torch::Tensor &, std::vector<torch::Tensor> &
);
torch::Tensor torch_loadx (int param1, int param2,
        const torch::Tensor &X, const torch::Tensor &constants,  std::vector<torch::Tensor> &stack) {
    return X[param1].view({X.size(1), 1});
}

torch::Tensor torch_loadc (int param1, int param2,
        const torch::Tensor &X, const torch::Tensor &constants,  std::vector<torch::Tensor> &stack) {
    return constants[param1];
}

torch::Tensor torch_add (int param1, int param2,
        const torch::Tensor &X, const torch::Tensor &constants,  std::vector<torch::Tensor> &stack) {
    return stack[param1] + stack[param2];
}

torch::Tensor torch_subtract (int param1, int param2,
        const torch::Tensor &X, const torch::Tensor &constants,  std::vector<torch::Tensor> &stack) {
    return stack[param1] - stack[param2];
}

torch::Tensor torch_multiply (int param1, int param2,
        const torch::Tensor &X, const torch::Tensor &constants,  std::vector<torch::Tensor> &stack) {
    return stack[param1] * stack[param2];
}

torch::Tensor torch_divide (int param1, int param2,
        const torch::Tensor &X, const torch::Tensor &constants,  std::vector<torch::Tensor> &stack) {
    return stack[param1] / stack[param2];
}

torch::Tensor torch_sin (int param1, int param2,
        const torch::Tensor &X, const torch::Tensor &constants,  std::vector<torch::Tensor> &stack) {
    return torch::sin(stack[param1]);
}

torch::Tensor torch_cos (int param1, int param2,
        const torch::Tensor &X, const torch::Tensor &constants,  std::vector<torch::Tensor> &stack) {
    return torch::cos(stack[param1]);
}

torch::Tensor torch_exp (int param1, int param2,
        const torch::Tensor &X, const torch::Tensor &constants,  std::vector<torch::Tensor> &stack) {
    return torch::exp(stack[param1]);
}

torch::Tensor torch_log (int param1, int param2,
        const torch::Tensor &X, const torch::Tensor &constants,  std::vector<torch::Tensor> &stack) {
    return torch::log(torch::abs(stack[param1]));
}

torch::Tensor torch_pow (int param1, int param2,
        const torch::Tensor &X, const torch::Tensor &constants,  std::vector<torch::Tensor> &stack) {
    return torch::pow(stack[param1], stack[param2]);  // NOTE (David): bingocpp has stack[param1] in abs but bingo doesn't
}

torch::Tensor torch_abs (int param1, int param2,
        const torch::Tensor &X, const torch::Tensor &constants,  std::vector<torch::Tensor> &stack) {
    return torch::abs(stack[param1]);
}

torch::Tensor torch_sqrt (int param1, int param2,
        const torch::Tensor &X, const torch::Tensor &constants,  std::vector<torch::Tensor> &stack) {
    return torch::sqrt(torch::abs(stack[param1]));
}

const torch_op_fn operator_fns[] = {
  torch_loadx,  // 0
  torch_loadc,  // 1
  torch_add,  // 2
  torch_subtract,  // 3
  torch_multiply,  // 4
  torch_divide,  // 5
  torch_sin,  // 6
  torch_cos,  // 7
  torch_exp,  // 8
  torch_log,  // 9
  torch_pow,  // 10
  torch_abs,  // 11
  torch_sqrt  // 12
};


torch::Tensor get_forward_eval(
    py::array_t<int> stack,
    torch::Tensor &X,
    torch::Tensor &constants) {
    auto stack_info = stack.unchecked<2>();
    int stack_size = stack_info.shape(0);

    std::vector<torch::Tensor> forward_eval(stack_size);
    for (int i = 0; i < stack_size; i++) {
        forward_eval[i] = (*operator_fns[stack_info(i, 0)])(stack_info(i, 1), stack_info(i, 2), X, constants, forward_eval);
    }

    return forward_eval[stack_size - 1];
}

torch::autograd::tensor_list eval_with_deriv(
    py::array_t<int> stack,
    torch::Tensor &X,
    torch::Tensor &constants,
    bool wrt_x_or_c) {
    pybind11::gil_scoped_release no_gil;

    torch::Tensor eval_constants = constants;

    torch::autograd::tensor_list inputs = {X};
    if (!wrt_x_or_c) {
//        eval_constants = eval_constants.view({eval_constants.size(0), 1}) * torch::ones_like(X);
        inputs = {eval_constants};
    }
    inputs[0].set_requires_grad(true);

    torch::Tensor eval = get_forward_eval(stack, X, eval_constants);

    torch::Tensor deriv;
    if (eval.requires_grad()) {
        deriv = torch::autograd::grad({eval}, inputs, {torch::ones_like(eval)}, c10::nullopt, false, false)[0];
    }
    else {
        deriv = torch::zeros_like(eval);
    }
    inputs[0].set_requires_grad(false);

    return {eval, deriv};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("evaluate", &get_forward_eval, "pytorch stack evaluation");
    m.def("evaluate_with_deriv", &eval_with_deriv, "pytorch stack and deriv evaluation");
}

int main() {
  return 0;
}

