/**
 * @file constant_folding.h
 * @brief Constant folding for CAS expressions.
 *
 * Port of pyagraph/simplification/constant_folding.py.
 * Reduces the number of distinct constants by grouping and merging
 * constant-valued sub-expressions.
 */

#pragma once

#include "cppagraph/cas_expression.h"

namespace cppagraph {

/**
 * Fold constant-valued sub-expressions together.
 *
 * @param expression  The CAS expression to fold.
 * @return            The folded expression.
 */
CASExprPtr fold_constants(const CASExprPtr& expression);

}  // namespace cppagraph
