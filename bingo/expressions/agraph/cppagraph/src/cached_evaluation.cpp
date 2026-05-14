/**
 * @file cached_evaluation.cpp
 * @brief CachedEvaluator implementation.
 *
 * Mirrors bingo.expressions.agraph.pyagraph.evaluation.cached_evaluation.
 */

#include "cppagraph/cached_evaluation.h"
#include "cppagraph/operator_eval.h"
#include "cppagraph/operators.h"

#include <algorithm>
#include <cstring>

namespace cppagraph {

// ================================================================
//  Dependency mask
// ================================================================

std::vector<bool> build_dependency_mask(const StackMatrix& stack) {
    const Eigen::Index n = stack.rows();
    std::vector<bool> depends(n, false);

    for (Eigen::Index i = 0; i < n; ++i) {
        uint8_t node = stack(i, 0);
        if (node == static_cast<uint8_t>(Op::CONSTANT)) {
            depends[i] = true;
        } else if (!IS_TERMINAL[node]) {
            uint8_t p1 = stack(i, 1);
            if (depends[p1]) {
                depends[i] = true;
            } else if (IS_ARITY_2[node]) {
                uint8_t p2 = stack(i, 2);
                if (depends[p2]) {
                    depends[i] = true;
                }
            }
        }
    }
    return depends;
}

// ================================================================
//  Reverse variant mask
// ================================================================

std::pair<std::vector<bool>, std::vector<bool>>
build_reverse_variant_mask(const StackMatrix& stack,
                           const std::vector<bool>& forward_depends) {
    const Eigen::Index n = stack.rows();
    std::vector<bool> reverse_variant(n, false);
    std::vector<bool> row_variant(n, false);

    // The last row always has adjoint = 1.0, which is invariant,
    // but if it's non-terminal and forward-depends, it's variant.
    // The loop handles this naturally.

    for (Eigen::Index i = n - 1; i >= 0; --i) {
        uint8_t node = stack(i, 0);
        if (IS_TERMINAL[node]) continue;

        bool is_add_sub = (node == static_cast<uint8_t>(Op::ADDITION) ||
                           node == static_cast<uint8_t>(Op::SUBTRACTION));
        bool is_var;
        if (is_add_sub) {
            // ADD/SUB reverse reads forward values only for shape (not value)
            // determination, and shapes are constant-independent, so variant
            // status depends only on whether the adjoint at this node varies.
            is_var = reverse_variant[i];
        } else {
            is_var = reverse_variant[i] || forward_depends[i];
        }
        row_variant[i] = is_var;

        if (is_var) {
            uint8_t p1 = stack(i, 1);
            reverse_variant[p1] = true;
            if (IS_ARITY_2[node]) {
                uint8_t p2 = stack(i, 2);
                reverse_variant[p2] = true;
            }
        }
    }
    return {reverse_variant, row_variant};
}

// ================================================================
//  CachedEvaluator
// ================================================================

CachedEvaluator::CachedEvaluator(StackMatrix stack, RowMatrixXd x,
                                   std::vector<int> integers)
    : stack_(std::move(stack)),
      x_(std::move(x)),
      integers_(std::move(integers)),
      n_(stack_.rows())
{
    depends_on_constant_ = build_dependency_mask(stack_);
    auto [rv, rowv] = build_reverse_variant_mask(stack_, depends_on_constant_);
    reverse_variant_ = std::move(rv);
    row_variant_ = std::move(rowv);
}

// ---- public API ---- //

RowMatrixXd CachedEvaluator::forward_eval(
        const std::vector<double>& constants) {
    auto forward = compute_forward(constants);

    // Cache for potential jacobian reuse.
    last_params_ = constants;
    last_forward_ = forward;
    has_last_ = true;

    return reshape_output(forward.back(), constants, x_.rows());
}

std::pair<RowMatrixXd, RowMatrixXd>
CachedEvaluator::forward_eval_with_const_derivative(
        const std::vector<double>& constants) {
    ForwardBuffer forward;
    if (has_last_ && params_equal(last_params_, constants)) {
        forward = last_forward_;
    } else {
        forward = compute_forward(constants);
        last_params_ = constants;
        last_forward_ = forward;
        has_last_ = true;
    }

    Eigen::Index deriv_rows = x_.rows();
    Eigen::Index deriv_cols = static_cast<Eigen::Index>(constants.size());
    auto derivative = reverse_eval_constants(forward, deriv_rows, deriv_cols);

    return {reshape_output(forward.back(), constants, x_.rows()), derivative};
}

// ---- internals ---- //

ForwardBuffer CachedEvaluator::compute_forward(
        const std::vector<double>& constants) {
    if (!static_forward_populated_) {
        // First call: evaluate everything and cache constant-independent rows.
        ForwardBuffer forward(n_);
        static_forward_.resize(n_);
        for (Eigen::Index i = 0; i < n_; ++i) {
            uint8_t node   = stack_(i, 0);
            uint8_t param1 = stack_(i, 1);
            uint8_t param2 = stack_(i, 2);
            forward[i] = forward_eval_one(
                node, param1, param2, x_, constants, integers_, forward);
            if (!depends_on_constant_[i]) {
                static_forward_[i] = forward[i];
            }
        }
        static_forward_populated_ = true;
        return forward;
    }

    // Subsequent calls: copy static rows, recompute dependent ones.
    ForwardBuffer forward(static_forward_);
    for (Eigen::Index i = 0; i < n_; ++i) {
        if (depends_on_constant_[i]) {
            uint8_t node   = stack_(i, 0);
            uint8_t param1 = stack_(i, 1);
            uint8_t param2 = stack_(i, 2);
            forward[i] = forward_eval_one(
                node, param1, param2, x_, constants, integers_, forward);
        }
    }
    return forward;
}

RowMatrixXd CachedEvaluator::reverse_eval_constants(
        const ForwardBuffer& forward,
        Eigen::Index deriv_rows, Eigen::Index deriv_cols) {
    if (!static_reverse_populated_) {
        return reverse_eval_first(forward, deriv_rows, deriv_cols);
    }
    return reverse_eval_cached(forward, deriv_rows, deriv_cols);
}

RowMatrixXd CachedEvaluator::reverse_eval_first(
        const ForwardBuffer& forward,
        Eigen::Index deriv_rows, Eigen::Index deriv_cols) {
    constexpr uint8_t CONST_ID = static_cast<uint8_t>(Op::CONSTANT);

    // --- Full reverse pass (correct result) ---
    RowMatrixXd derivative = RowMatrixXd::Zero(deriv_rows, deriv_cols);
    ReverseBuf reverse(n_);
    for (Eigen::Index i = 0; i < n_; ++i)
        reverse[i] = RowMatrixXd::Zero(1, 1);
    reverse[n_ - 1](0, 0) = 1.0;

    for (Eigen::Index i = n_ - 1; i >= 0; --i) {
        uint8_t node   = stack_(i, 0);
        uint8_t param1 = stack_(i, 1);
        uint8_t param2 = stack_(i, 2);
        if (node == CONST_ID) {
            // Accumulate adjoint into derivative.
            const auto& r = reverse[i];
            if (r.rows() == deriv_rows) {
                derivative.col(param1) += r.col(0);
            } else {
                derivative.col(param1).array() += r(0, 0);
            }
        } else if (!IS_TERMINAL[node]) {
            reverse_eval_one(node, static_cast<int>(i),
                             param1, param2, forward, reverse);
        }
    }

    // --- Static reverse pass (invariant rows only) ---
    ReverseBuf static_rev(n_);
    for (Eigen::Index i = 0; i < n_; ++i)
        static_rev[i] = RowMatrixXd::Zero(1, 1);
    static_rev[n_ - 1](0, 0) = 1.0;

    RowMatrixXd static_deriv = RowMatrixXd::Zero(deriv_rows, deriv_cols);

    for (Eigen::Index i = n_ - 1; i >= 0; --i) {
        uint8_t node   = stack_(i, 0);
        uint8_t param1 = stack_(i, 1);
        uint8_t param2 = stack_(i, 2);
        if (row_variant_[i]) continue;

        if (node == CONST_ID) {
            if (!reverse_variant_[i]) {
                const auto& r = static_rev[i];
                if (r.rows() == deriv_rows) {
                    static_deriv.col(param1) += r.col(0);
                } else {
                    static_deriv.col(param1).array() += r(0, 0);
                }
            }
        } else if (!IS_TERMINAL[node]) {
            reverse_eval_one(node, static_cast<int>(i),
                             param1, param2, forward, static_rev);
        }
    }

    static_reverse_ = std::move(static_rev);
    static_derivative_ = std::move(static_deriv);
    static_reverse_populated_ = true;

    return derivative;
}

RowMatrixXd CachedEvaluator::reverse_eval_cached(
        const ForwardBuffer& forward,
        Eigen::Index deriv_rows, Eigen::Index deriv_cols) {
    constexpr uint8_t CONST_ID = static_cast<uint8_t>(Op::CONSTANT);

    // Deep-copy cached buffers.
    ReverseBuf reverse(n_);
    for (Eigen::Index i = 0; i < n_; ++i) {
        reverse[i] = static_reverse_[i];  // Eigen copy
    }
    RowMatrixXd derivative = static_derivative_;

    for (Eigen::Index i = n_ - 1; i >= 0; --i) {
        uint8_t node   = stack_(i, 0);
        uint8_t param1 = stack_(i, 1);
        uint8_t param2 = stack_(i, 2);
        if (node == CONST_ID) {
            if (reverse_variant_[i]) {
                const auto& r = reverse[i];
                if (r.rows() == deriv_rows) {
                    derivative.col(param1) += r.col(0);
                } else {
                    derivative.col(param1).array() += r(0, 0);
                }
            }
        } else if (row_variant_[i]) {
            reverse_eval_one(node, static_cast<int>(i),
                             param1, param2, forward, reverse);
        }
    }
    return derivative;
}

bool CachedEvaluator::params_equal(const std::vector<double>& a,
                                    const std::vector<double>& b) {
    if (a.size() != b.size()) return false;
    return std::equal(a.begin(), a.end(), b.begin());
}

}  // namespace cppagraph
