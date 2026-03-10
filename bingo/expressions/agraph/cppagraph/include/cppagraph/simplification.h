/**
 * @file simplification.h
 * @brief Stack reduction for AGraph command arrays.
 *
 * Mirrors bingo.expressions.agraph.pyagraph.simplification._reduce.
 *
 * Two functions are provided:
 *
 * - get_utilized_commands(): backward scan that marks which rows of
 *   the stack are reachable from the final output.
 *
 * - reduce(): dead-code elimination with terminal-index renumbering.
 *   Unused rows are dropped, operator row-references are remapped,
 *   and CONSTANT / INTEGER indices are compacted to sequential
 *   positions retaining only the referenced values.
 */

#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

#include "cppagraph/evaluation.h"   // StackMatrix

namespace cppagraph {

/**
 * Find which commands are utilized by the final output.
 *
 * Starting from the last row (the output), walks backwards and marks
 * any row referenced by a utilized non-terminal.
 *
 * @param stack  Nx3 uint8 command array.
 * @return       Boolean vector of length N (true = utilized).
 */
std::vector<bool> get_utilized_commands(const StackMatrix& stack);

/**
 * Result of the reduce() function.
 */
struct ReduceResult {
    StackMatrix stack;                      ///< Reduced Mx3 command array.
    std::vector<double> constants;          ///< Compacted constant values.
    std::vector<int> integers;              ///< Compacted integer values.
    std::vector<int> constant_mapping;      ///< constant_mapping[new] = raw_idx.
};

/**
 * Reduce the raw stack and derive simplified constants and integers.
 *
 * Performs dead-code elimination and terminal renumbering in a single
 * pass over the raw command array.  Unused rows are dropped, operator
 * row-references are remapped, and CONSTANT / INTEGER indices are
 * compacted to sequential positions with only the referenced values
 * retained.
 *
 * @param raw_stack      Nx3 uint8 raw command array.
 * @param raw_constants  Constant values indexed by CONSTANT node params.
 * @param raw_integers   Integer values indexed by INTEGER node params.
 * @return ReduceResult  The reduced stack, compacted constants/integers,
 *                       and the constant_mapping (new_idx → raw_idx).
 */
ReduceResult reduce(
    const StackMatrix& raw_stack,
    const std::vector<double>& raw_constants,
    const std::vector<int>& raw_integers);

}  // namespace cppagraph
