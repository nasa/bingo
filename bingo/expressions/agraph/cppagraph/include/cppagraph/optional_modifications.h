/**
 * @file optional_modifications.h
 * @brief Optional post-simplification modifications.
 *
 * Port of pyagraph/simplification/optional_modifications.py.
 * These produce a canonical form better suited to the AGraph stack.
 */

#pragma once

#include "cppagraph/cas_expression.h"

namespace cppagraph {

/// Control flags (match module-level Python globals).
struct OptionalModFlags {
    bool insert_subtraction          = true;
    bool insert_division             = true;
    bool insert_square_cube          = true;
    bool replace_integer_powers      = false;
    bool replace_integers_with_constants = false;
};

/**
 * Apply optional post-simplification modifications.
 *
 * @param expression  The CAS expression.
 * @param flags       Which modifications to apply.
 * @return            The modified expression.
 */
CASExprPtr optional_modifications(
    const CASExprPtr& expression,
    const OptionalModFlags& flags = OptionalModFlags{});

}  // namespace cppagraph
