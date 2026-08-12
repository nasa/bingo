/**
 * @file evaluation.cpp
 * @brief Forward evaluation and reverse-mode autodiff implementation.
 *
 * Mirrors bingo.expressions.agraph.pyagraph.evaluation.evaluation.
 */

#include "cppagraph/evaluation.h"
#include "cppagraph/operator_eval.h"
#include "cppagraph/operators.h"

#include <cmath>
#include <stdexcept>

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

RowMatrixXd evaluate(
        const StackMatrix& stack,
        const RowMatrixXd& x,
        Eigen::Ref<const RowMatrixXd> constants,
        const std::vector<int>& integers) {
    const Eigen::Index batch_size = constants.cols();
    if (batch_size == 0)
        return RowMatrixXd(x.rows(), 0);

    const Eigen::Index n = stack.rows();
    ForwardBuffer forward(n);
    for (Eigen::Index row = 0; row < n; ++row) {
        forward[row] = forward_eval_one_batched(
            stack(row, 0), stack(row, 1), stack(row, 2),
            x, constants, integers, forward);
    }

    const auto& output = forward.back();
    if (output.rows() == x.rows() && output.cols() == batch_size)
        return output;
    if ((output.rows() != 1 && output.rows() != x.rows()) ||
        (output.cols() != 1 && output.cols() != batch_size)) {
        throw std::invalid_argument("incompatible evaluation output shape");
    }
    const Eigen::Index row_factor = output.rows() == x.rows() ? 1 : x.rows();
    const Eigen::Index col_factor = output.cols() == batch_size ? 1 : batch_size;
    return output.replicate(row_factor, col_factor);
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

ConstHessianResult evaluate_with_const_hessian(
        const StackMatrix& stack,
        const RowMatrixXd& x,
        const std::vector<double>& constants,
        const std::vector<int>& integers) {
    const Eigen::Index m = x.rows();
    const Eigen::Index l = static_cast<Eigen::Index>(constants.size());
    auto forward = forward_pass(stack, x, constants, integers);
    for (Eigen::Index i = 0; i < stack.rows(); ++i) {
        if (stack(i, 0) == static_cast<uint8_t>(Op::DIVISION)
            && forward[stack(i, 1)].size() == 1
            && forward[stack(i, 2)].size() == 1
            && forward[stack(i, 2)](0, 0) == 0.0) {
            throw std::domain_error("float division by zero");
        }
    }
    std::vector<RowMatrixXd> gradients(stack.rows(), RowMatrixXd::Zero(m, l));
    std::vector<RowMatrixXd> hessians(
        stack.rows(), RowMatrixXd::Zero(m, l * l));
    std::vector<bool> constant_dependencies(stack.rows(), false);

    auto value_at = [m](const RowMatrixXd& value, Eigen::Index row) {
        return value.rows() == 1 ? value(0, 0) : value(row, 0);
    };

    for (Eigen::Index i = 0; i < stack.rows(); ++i) {
        const uint8_t node = stack(i, 0);
        const uint8_t p1 = stack(i, 1);
        const uint8_t p2 = stack(i, 2);
        if (node == static_cast<uint8_t>(Op::CONSTANT)) {
            for (Eigen::Index row = 0; row < m; ++row)
                gradients[i](row, p1) = 1.0;
            constant_dependencies[i] = true;
            continue;
        }
        if (IS_TERMINAL[node]) continue;

        const bool depends_on_first = constant_dependencies[p1];
        const bool is_binary = IS_ARITY_2[node];
        const bool depends_on_second =
            is_binary && constant_dependencies[p2];
        constant_dependencies[i] = depends_on_first || depends_on_second;
        if (!constant_dependencies[i]) continue;

        for (Eigen::Index row = 0; row < m; ++row) {
            const double a = value_at(forward[p1], row);
            const double b = is_binary ? value_at(forward[p2], row) : 0.0;
            const double y = value_at(forward[i], row);
            const double sign_a = static_cast<double>((a > 0.0) - (a < 0.0));
            double fa = 0.0, fb = 0.0, faa = 0.0, fab = 0.0, fbb = 0.0;
            switch (static_cast<Op>(node)) {
                case Op::ADDITION: fa = fb = 1.0; break;
                case Op::SUBTRACTION: fa = 1.0; fb = -1.0; break;
                case Op::MULTIPLICATION: fa = b; fb = a; fab = 1.0; break;
                case Op::DIVISION:
                    fa = 1.0 / b; fb = -a / (b * b); fab = -1.0 / (b * b);
                    fbb = 2.0 * a / (b * b * b); break;
                case Op::POWER:
                    fa = y * b / a; fb = y * std::log(a);
                    faa = y * b * (b - 1.0) / (a * a);
                    fab = y * (b * std::log(a) + 1.0) / a;
                    fbb = y * std::log(a) * std::log(a); break;
                case Op::SAFE_POWER:
                    fa = y * b / a; fb = y * std::log(std::abs(a));
                    faa = y * b * (b - 1.0) / (a * a);
                    fab = y * (b * std::log(std::abs(a)) + 1.0) / a;
                    fbb = y * std::log(std::abs(a)) * std::log(std::abs(a)); break;
                case Op::SQUARE: fa = 2.0 * a; faa = 2.0; break;
                case Op::CUBE: fa = 3.0 * a * a; faa = 6.0 * a; break;
                case Op::SQRT: fa = 0.5 * sign_a / y;
                    faa = -0.25 / (std::abs(a) * y); break;
                case Op::ABS: fa = sign_a; break;
                case Op::EXPONENTIAL: fa = faa = y; break;
                case Op::LOGARITHM: fa = 1.0 / a; faa = -1.0 / (a * a); break;
                case Op::SIN: fa = std::cos(a); faa = -std::sin(a); break;
                case Op::COS: fa = -std::sin(a); faa = -std::cos(a); break;
                case Op::TAN: fa = 1.0 / (std::cos(a) * std::cos(a));
                    faa = 2.0 * fa * std::tan(a); break;
                case Op::ARCSIN: fa = 1.0 / std::sqrt(1.0 - a * a);
                    faa = a / std::pow(1.0 - a * a, 1.5); break;
                case Op::ARCCOS: fa = -1.0 / std::sqrt(1.0 - a * a);
                    faa = -a / std::pow(1.0 - a * a, 1.5); break;
                case Op::ARCTAN: fa = 1.0 / (1.0 + a * a);
                    faa = -2.0 * a / std::pow(1.0 + a * a, 2.0); break;
                case Op::SINH: fa = std::cosh(a); faa = std::sinh(a); break;
                case Op::COSH: fa = std::sinh(a); faa = std::cosh(a); break;
                case Op::TANH: fa = 1.0 / (std::cosh(a) * std::cosh(a));
                    faa = -2.0 * std::tanh(a) * fa; break;
                default: break;
            }
            for (Eigen::Index j = 0; j < l; ++j) {
                gradients[i](row, j) =
                    (depends_on_first ? fa * gradients[p1](row, j) : 0.0)
                    + (depends_on_second ? fb * gradients[p2](row, j) : 0.0);
                for (Eigen::Index k = 0; k < l; ++k) {
                    hessians[i](row, j * l + k) =
                        (depends_on_first
                             ? fa * hessians[p1](row, j * l + k)
                             + faa * gradients[p1](row, j) * gradients[p1](row, k)
                             : 0.0)
                        + (depends_on_second
                               ? fb * hessians[p2](row, j * l + k)
                               + fbb * gradients[p2](row, j) * gradients[p2](row, k)
                               : 0.0)
                        + (depends_on_first && depends_on_second
                               ? fab * (gradients[p1](row, j) * gradients[p2](row, k)
                                      + gradients[p2](row, j) * gradients[p1](row, k))
                               : 0.0);
                }
            }
        }
    }
    return {reshape_output(forward.back(), constants, m), gradients.back(),
            hessians.back()};
}

}  // namespace cppagraph
