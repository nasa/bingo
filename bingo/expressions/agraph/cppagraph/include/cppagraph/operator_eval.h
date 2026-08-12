/**
 * @file operator_eval.h
 * @brief Per-operator forward and reverse evaluation dispatch.
 *
 * Mirrors bingo.expressions.agraph.pyagraph.evaluation.operator_eval.
 *
 * Forward signature:
 *   forward_eval_one(node, param1, param2, x, constants, integers, fwd) -> MatrixXd
 *
 * Reverse signature:
 *   reverse_eval_one(node, ri, param1, param2, fwd, rev) -> void  (mutates rev)
 */

#pragma once

#include <cstdint>
#include <vector>

#include "cppagraph/data_container.h"  // RowMatrixXd

namespace cppagraph {

// Forward-buffer type is std::vector<RowMatrixXd> (same as evaluation.h).
// We redeclare here to avoid a circular include.
using ForwardBuf  = std::vector<RowMatrixXd>;
using ReverseBuf  = std::vector<RowMatrixXd>;

/**
 * Evaluate one row of the command stack (forward pass).
 *
 * Dispatches to the correct operator function based on @p node.
 */
RowMatrixXd forward_eval_one(
    uint8_t node,
    uint8_t param1,
    uint8_t param2,
    const RowMatrixXd& x,
    const std::vector<double>& constants,
    const std::vector<int>& integers,
    const ForwardBuf& fwd);

/** Evaluate one row with constant-major LxB temporary constants. */
RowMatrixXd forward_eval_one_batched(
    uint8_t node,
    uint8_t param1,
    uint8_t param2,
    const RowMatrixXd& x,
    Eigen::Ref<const RowMatrixXd> constants,
    const std::vector<int>& integers,
    const ForwardBuf& fwd);

/**
 * Evaluate one row of the command stack (reverse / derivative pass).
 *
 * Dispatches to the correct operator reverse function.
 * Mutates @p rev in-place.
 */
void reverse_eval_one(
    uint8_t node,
    int ri,
    uint8_t param1,
    uint8_t param2,
    const ForwardBuf& fwd,
    ReverseBuf& rev);

}  // namespace cppagraph
