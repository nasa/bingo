/**
 * @file cas_simplify.cpp
 * @brief Top-level CAS simplification pipeline — port of _simplify.py.
 */

#include "cppagraph/cas_simplify.h"
#include "cppagraph/simplification.h"        // reduce()
#include "cppagraph/interpreter.h"           // build_simplified_cas_expression, build_agraph_stack
#include "cppagraph/constant_folding.h"      // fold_constants
#include "cppagraph/optional_modifications.h"  // optional_modifications

#include <algorithm>

namespace cppagraph {

SimplifyResult cas_simplify(
    const StackMatrix& raw_stack,
    const std::vector<double>& raw_constants,
    const std::vector<int>& raw_integers)
{
    // 1. Eliminate dead code.
    auto reduced = reduce(raw_stack, raw_constants, raw_integers);

    // 2. Build the CAS tree with automatic simplification fused in.
    auto cas_expr = build_simplified_cas_expression(
        reduced.stack, reduced.constants, reduced.integers);

    // 3. Fold constants.
    cas_expr = fold_constants(cas_expr);

    // 4. Optional modifications (default flags).
    cas_expr = optional_modifications(cas_expr);

    // 5. Convert back to command-array form.
    auto built = build_agraph_stack(cas_expr, reduced.constants);

    // 6. Compose: final_idx → reduced_idx → raw_idx.
    // built.const_idx_map maps reduced_idx → final_idx.
    // We need constant_mapping[final_idx] = raw_idx.
    //
    // Sort by final_idx (the value in const_idx_map).
    std::vector<std::pair<int, int>> sorted_pairs(
        built.const_idx_map.begin(), built.const_idx_map.end());
    std::sort(sorted_pairs.begin(), sorted_pairs.end(),
              [](auto& a, auto& b) { return a.second < b.second; });

    std::vector<int> constant_mapping;
    constant_mapping.reserve(sorted_pairs.size());
    for (auto& [reduced_idx, _final_idx] : sorted_pairs) {
        if (reduced_idx < static_cast<int>(reduced.constant_mapping.size()))
            constant_mapping.push_back(reduced.constant_mapping[reduced_idx]);
        else
            constant_mapping.push_back(reduced_idx);
    }

    return {
        std::move(built.stack),
        std::move(built.constants),
        std::move(built.integers),
        std::move(constant_mapping)
    };
}

}  // namespace cppagraph
