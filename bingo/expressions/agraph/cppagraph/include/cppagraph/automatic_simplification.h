/**
 * @file automatic_simplification.h
 * @brief CAS automatic simplification rules.
 *
 * Port of pyagraph/simplification/automatic_simplification.py.
 * Based on the algorithm in Chapter 3 of Joel Cohen's book.
 */

#pragma once

#include <cstdint>
#include <functional>
#include <unordered_map>

#include "cppagraph/cas_expression.h"

namespace cppagraph {

/// Simplification function signature.
using SimplifyFn = std::function<CASExprPtr(const CASExprPtr&)>;

/// Dispatch table mapping operator → simplification function.
const std::unordered_map<uint8_t, SimplifyFn>& simplification_functions();

/// Recursively simplify a CAS expression.
CASExprPtr automatic_simplify(const CASExprPtr& expression);

// Individual simplification functions (also called directly by the
// interpreter for the fused build-simplify path).
CASExprPtr simplify_power(const CASExprPtr& expression);
CASExprPtr simplify_product(const CASExprPtr& expression);
CASExprPtr simplify_sum(const CASExprPtr& expression);
CASExprPtr simplify_quotient(const CASExprPtr& expression);
CASExprPtr simplify_difference(const CASExprPtr& expression);

}  // namespace cppagraph
