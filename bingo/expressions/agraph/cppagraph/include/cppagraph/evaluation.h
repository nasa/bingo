/**
 * @file evaluation.h
 * @brief Forward evaluation and reverse-mode autodiff for AGraph stacks.
 *
 * Mirrors bingo.expressions.agraph.pyagraph.evaluation.evaluation,
 * operating on Eigen RowMajor matrices for zero-copy NumPy interop.
 *
 * Key types used throughout:
 *   - Stack:     Nx3 uint8 RowMajor matrix  (node, param1, param2)
 *   - X:         MxD double RowMajor matrix  (data points)
 *   - Constants: std::vector<double>          (fitted constants)
 *   - Integers:  std::vector<int>             (integer lookup)
 *   - ForwardBuffer: std::vector<RowMatrixXd> (one MxC matrix per row)
 */

#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

#include <Eigen/Core>

#include "cppagraph/data_container.h"   // RowMatrixXd typedef

namespace cppagraph {

/// Nx3 uint8 RowMajor matrix — the command stack.
using StackMatrix =
    Eigen::Matrix<uint8_t, Eigen::Dynamic, 3, Eigen::RowMajor>;

/// Per-row forward-evaluation buffer.
using ForwardBuffer = std::vector<RowMatrixXd>;

/** Result of constant-gradient and Hessian evaluation.
 *
 * hessian stores each per-sample LxL matrix flattened row-major into an
 * Mx(L*L) matrix, where L is the number of constants.
 */
struct ConstHessianResult {
    RowMatrixXd value;
    RowMatrixXd gradient;
    RowMatrixXd hessian;
};

// -----------------------------------------------------------------
//  Public API  (mirrors pyagraph.evaluation.evaluate / evaluate_with_derivative)
// -----------------------------------------------------------------

/**
 * Evaluate an equation represented by a command stack.
 *
 * @param stack     Nx3 uint8 matrix (node, param1, param2).
 * @param x         MxD input data.
 * @param constants Fitted constants.
 * @param integers  Integer lookup.
 * @return Mx1 (or MxC) result matrix f(x).
 */
RowMatrixXd evaluate(
    const StackMatrix& stack,
    const RowMatrixXd& x,
    const std::vector<double>& constants,
    const std::vector<int>& integers);

/** Evaluate with constant-major LxB temporary constants, returning MxB. */
RowMatrixXd evaluate(
    const StackMatrix& stack,
    const RowMatrixXd& x,
    Eigen::Ref<const RowMatrixXd> constants,
    const std::vector<int>& integers);

/**
 * Evaluate and compute derivative via reverse-mode autodiff.
 *
 * @param stack     Nx3 uint8 matrix.
 * @param x         MxD input data.
 * @param constants Fitted constants.
 * @param integers  Integer lookup.
 * @param wrt_x     true → derivative w.r.t. x (MxD);
 *                  false → derivative w.r.t. constants (MxL).
 * @return (f(x), derivative) pair.
 */
std::pair<RowMatrixXd, RowMatrixXd> evaluate_with_derivative(
    const StackMatrix& stack,
    const RowMatrixXd& x,
    const std::vector<double>& constants,
    const std::vector<int>& integers,
    bool wrt_x);

/** Evaluate f(x), df/dc, and d2f/dc2. */
ConstHessianResult evaluate_with_const_hessian(
    const StackMatrix& stack,
    const RowMatrixXd& x,
    const std::vector<double>& constants,
    const std::vector<int>& integers);

// -----------------------------------------------------------------
//  Internal helpers (exposed for CachedEvaluator & testing)
// -----------------------------------------------------------------

/**
 * Run the forward pass over the entire stack.
 *
 * @return vector of length N where element i is the Mx? result of row i.
 */
ForwardBuffer forward_pass(
    const StackMatrix& stack,
    const RowMatrixXd& x,
    const std::vector<double>& constants,
    const std::vector<int>& integers);

/**
 * Run the reverse pass to compute derivatives.
 *
 * @param stack          Nx3 command stack.
 * @param forward        Forward buffer from forward_pass().
 * @param deriv_rows     M — number of data rows.
 * @param deriv_cols     D (wrt x) or L (wrt constants).
 * @param deriv_wrt_node Op::VARIABLE or Op::CONSTANT.
 * @return MxD or MxL derivative matrix.
 */
RowMatrixXd reverse_pass(
    const StackMatrix& stack,
    const ForwardBuffer& forward,
    Eigen::Index deriv_rows,
    Eigen::Index deriv_cols,
    uint8_t deriv_wrt_node);

/**
 * Reshape / broadcast the final forward-buffer entry to (M, C).
 *
 * C is 1 unless constants are array-valued (rare; kept for parity).
 */
RowMatrixXd reshape_output(
    const RowMatrixXd& output,
    const std::vector<double>& constants,
    Eigen::Index m);

}  // namespace cppagraph
