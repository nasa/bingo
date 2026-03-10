/**
 * @file interpreter.h
 * @brief Translate between command-array stacks and CAS expression trees.
 *
 * Port of pyagraph/simplification/interpreter.py.
 */

#pragma once

#include <cstdint>
#include <map>
#include <tuple>
#include <vector>

#include "cppagraph/cas_expression.h"
#include "cppagraph/evaluation.h"  // StackMatrix

namespace cppagraph {

/**
 * Translate a command array into a CAS expression tree.
 */
CASExprPtr build_cas_expression(
    const StackMatrix& stack,
    const std::vector<double>& constants,
    const std::vector<int>& integers);

/**
 * Build a CAS tree and simplify each node as it is constructed.
 *
 * Fuses the bottom-up tree build with automatic_simplify.
 */
CASExprPtr build_simplified_cas_expression(
    const StackMatrix& stack,
    const std::vector<double>& constants,
    const std::vector<int>& integers);

/**
 * Result of build_agraph_stack().
 */
struct BuildStackResult {
    StackMatrix stack;
    std::vector<double> constants;
    std::vector<int> integers;
    std::map<int, int> const_idx_map;  ///< old CONSTANT index → new index.
};

/**
 * Translate a CAS expression tree back into a command array.
 */
BuildStackResult build_agraph_stack(
    const CASExprPtr& expression,
    const std::vector<double>& original_constants);

}  // namespace cppagraph
