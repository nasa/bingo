/**
 * @file evaluation.cpp
 * @brief Forward evaluation and reverse-mode autodiff implementation.
 *
 * Mirrors bingo.expressions.agraph.pyagraph.evaluation.evaluation.
 */

#include "cppagraph/evaluation.h"
#include "cppagraph/operator_eval.h"
#include "cppagraph/operators.h"

namespace cppagraph {

// -----------------------------------------------------------------
//  reshape_output  (matches pyagraph._reshape_output)
// -----------------------------------------------------------------

RowMatrixXd reshape_output(
        const RowMatrixXd& output,
        const std::vector<double>& /*constants*/,
        Eigen::Index m) {
    // c_dim is always 1 in the current scalar-constant path.
    Eigen::Index c_dim = 1;
    if (output.rows() == m && output.cols() == c_dim)
        return output;
    // Broadcast scalar to Mx1
    return RowMatrixXd::Constant(m, c_dim, output(0, 0));
}

// -----------------------------------------------------------------
//  forward_pass
// -----------------------------------------------------------------

ForwardBuffer forward_pass(
        const StackMatrix& stack,
        const RowMatrixXd& x,
        const std::vector<double>& constants,
        const std::vector<int>& integers) {
    const Eigen::Index n = stack.rows();
    ForwardBuffer fwd(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        uint8_t node   = stack(i, 0);
        uint8_t param1 = stack(i, 1);
        uint8_t param2 = stack(i, 2);
        fwd[i] = forward_eval_one(node, param1, param2,
                                   x, constants, integers, fwd);
    }
    return fwd;
}

// -----------------------------------------------------------------
//  reverse_pass
// -----------------------------------------------------------------

RowMatrixXd reverse_pass(
        const StackMatrix& stack,
        const ForwardBuffer& forward,
        Eigen::Index deriv_rows,
        Eigen::Index deriv_cols,
        uint8_t deriv_wrt_node) {
    const Eigen::Index n = stack.rows();

    RowMatrixXd derivative = RowMatrixXd::Zero(deriv_rows, deriv_cols);

    // Reverse buffer: each entry may be scalar (1x1) or (M x C).
    // Initialise to 0 (1x1) for every row except the last, which is 1.
    ReverseBuf rev(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        rev[i] = RowMatrixXd::Zero(1, 1);
    }
    rev[n - 1](0, 0) = 1.0;

    for (Eigen::Index i = n - 1; i >= 0; --i) {
        uint8_t node   = stack(i, 0);
        uint8_t param1 = stack(i, 1);
        uint8_t param2 = stack(i, 2);

        if (node == deriv_wrt_node) {
            // Accumulate into the derivative matrix.
            const auto& r = rev[i];
            if (r.rows() == deriv_rows) {
                // r is already (M, C); collapse to (M,) and add to col.
                for (Eigen::Index row = 0; row < deriv_rows; ++row)
                    derivative(row, param1) += r(row, 0);
            } else {
                // r is scalar (1x1): add to every row of col param1.
                double v = r(0, 0);
                for (Eigen::Index row = 0; row < deriv_rows; ++row)
                    derivative(row, param1) += v;
            }
        } else if (!IS_TERMINAL[node]) {
            reverse_eval_one(node, static_cast<int>(i),
                             param1, param2, forward, rev);
        }
    }
    return derivative;
}

// -----------------------------------------------------------------
//  Public API
// -----------------------------------------------------------------

RowMatrixXd evaluate(
        const StackMatrix& stack,
        const RowMatrixXd& x,
        const std::vector<double>& constants,
        const std::vector<int>& integers) {
    auto fwd = forward_pass(stack, x, constants, integers);
    return reshape_output(fwd.back(), constants, x.rows());
}

std::pair<RowMatrixXd, RowMatrixXd> evaluate_with_derivative(
        const StackMatrix& stack,
        const RowMatrixXd& x,
        const std::vector<double>& constants,
        const std::vector<int>& integers,
        bool wrt_x) {
    auto fwd = forward_pass(stack, x, constants, integers);

    Eigen::Index deriv_rows = x.rows();
    Eigen::Index deriv_cols;
    uint8_t deriv_wrt_node;
    if (wrt_x) {
        deriv_cols = x.cols();
        deriv_wrt_node = static_cast<uint8_t>(Op::VARIABLE);
    } else {
        deriv_cols = static_cast<Eigen::Index>(constants.size());
        deriv_wrt_node = static_cast<uint8_t>(Op::CONSTANT);
    }

    auto deriv = reverse_pass(stack, fwd, deriv_rows, deriv_cols,
                              deriv_wrt_node);

    return {reshape_output(fwd.back(), constants, x.rows()), deriv};
}

}  // namespace cppagraph
