/**
 * @file cas_simplify.h
 * @brief Top-level CAS simplification pipeline.
 *
 * Port of pyagraph/simplification/_simplify.py.
 * Orchestrates: reduce → build+simplify CAS tree → fold constants →
 * optional modifications → back to command array.
 */

#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

#include "cppagraph/evaluation.h"  // StackMatrix

namespace cppagraph {

/**
 * Result of the full CAS simplification pipeline.
 */
struct SimplifyResult {
    StackMatrix stack;
    std::vector<double> constants;
    std::vector<int> integers;
    std::vector<int> constant_mapping;  ///< constant_mapping[new] = raw_idx.
};

/**
 * Simplify via the full CAS pipeline.
 *
 * @param raw_stack      Nx3 uint8 raw command array.
 * @param raw_constants  Constant values indexed by CONSTANT node params.
 * @param raw_integers   Integer values indexed by INTEGER node params.
 * @return SimplifyResult  The simplified stack with mappings.
 */
SimplifyResult cas_simplify(
    const StackMatrix& raw_stack,
    const std::vector<double>& raw_constants,
    const std::vector<int>& raw_integers);

}  // namespace cppagraph
