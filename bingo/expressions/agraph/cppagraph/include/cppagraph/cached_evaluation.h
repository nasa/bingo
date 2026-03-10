/**
 * @file cached_evaluation.h
 * @brief CachedEvaluator for constant-optimisation (fitting).
 *
 * Mirrors bingo.expressions.agraph.pyagraph.evaluation.cached_evaluation.
 *
 * Three optimisations over the naive evaluate() path:
 *
 * 1. Fused residual / jacobian — cache the forward buffer from the
 *    residual eval and reuse it for the immediately following jacobian
 *    call (same constants), eliminating one forward pass per LM step.
 *
 * 2. Partial forward re-evaluation — only constant-dependent rows are
 *    recomputed; constant-independent rows use a persistent cache.
 *
 * 3. Partial reverse re-evaluation — invariant reverse contributions
 *    are cached and reused; only variant rows are re-executed.
 */

#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "cppagraph/data_container.h"   // RowMatrixXd
#include "cppagraph/evaluation.h"       // StackMatrix, ForwardBuffer
#include "cppagraph/operator_eval.h"    // ReverseBuf

namespace cppagraph {

/**
 * Build a boolean mask indicating constant-dependent stack rows.
 *
 * A row depends on constants if it is a CONSTANT node or any of its
 * operands point to a constant-dependent row.
 */
std::vector<bool> build_dependency_mask(const StackMatrix& stack);

/**
 * Build masks identifying variant reverse-pass rows.
 *
 * @param stack           Nx3 command stack.
 * @param forward_depends Dependency mask from build_dependency_mask().
 * @return (reverse_variant, row_variant) pair.
 */
std::pair<std::vector<bool>, std::vector<bool>>
build_reverse_variant_mask(const StackMatrix& stack,
                           const std::vector<bool>& forward_depends);

/**
 * @class CachedEvaluator
 * @brief Evaluation context with caching for repeated constant-only changes.
 *
 * Create at the start of fit(), discard when fitting is complete.
 */
class CachedEvaluator {
public:
    /**
     * @param stack    Simplified command stack (Nx3 uint8).
     * @param x        Training input data (MxD).
     * @param integers Integer lookup.
     */
    CachedEvaluator(StackMatrix stack, RowMatrixXd x,
                    std::vector<int> integers);

    /**
     * Evaluate f(x) with the given constants.
     */
    RowMatrixXd forward_eval(const std::vector<double>& constants);

    /**
     * Evaluate f(x) and df/dc with the given constants.
     *
     * If the cached forward buffer was computed with the same constants
     * (from a preceding forward_eval call), the forward pass is skipped.
     */
    std::pair<RowMatrixXd, RowMatrixXd>
    forward_eval_with_const_derivative(const std::vector<double>& constants);

private:
    StackMatrix stack_;
    RowMatrixXd x_;
    std::vector<int> integers_;
    Eigen::Index n_;

    // Dependency mask (computed once in ctor)
    std::vector<bool> depends_on_constant_;

    // Reverse variant masks (computed once in ctor)
    std::vector<bool> reverse_variant_;
    std::vector<bool> row_variant_;

    // Partial-forward cache
    ForwardBuffer static_forward_;   // constant-independent rows
    bool static_forward_populated_ = false;

    // Partial-reverse cache
    ReverseBuf static_reverse_;
    RowMatrixXd static_derivative_;
    bool static_reverse_populated_ = false;

    // Fused residual/jacobian state
    std::vector<double> last_params_;
    ForwardBuffer last_forward_;
    bool has_last_ = false;

    // Internal helpers
    ForwardBuffer compute_forward(const std::vector<double>& constants);
    RowMatrixXd reverse_eval_constants(const ForwardBuffer& forward,
                                       Eigen::Index deriv_rows,
                                       Eigen::Index deriv_cols);
    RowMatrixXd reverse_eval_first(const ForwardBuffer& forward,
                                    Eigen::Index deriv_rows,
                                    Eigen::Index deriv_cols);
    RowMatrixXd reverse_eval_cached(const ForwardBuffer& forward,
                                     Eigen::Index deriv_rows,
                                     Eigen::Index deriv_cols);

    static bool params_equal(const std::vector<double>& a,
                             const std::vector<double>& b);
};

}  // namespace cppagraph
