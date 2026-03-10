/**
 * @file simplification.cpp
 * @brief Stack reduction implementation.
 *
 * Mirrors bingo.expressions.agraph.pyagraph.simplification._reduce.
 */

#include "cppagraph/simplification.h"
#include "cppagraph/operators.h"

#include <numeric>   // std::partial_sum
#include <vector>

namespace cppagraph {

// -----------------------------------------------------------------
//  get_utilized_commands
// -----------------------------------------------------------------

std::vector<bool> get_utilized_commands(const StackMatrix& stack) {
    const Eigen::Index n = stack.rows();
    std::vector<bool> util(n, false);
    if (n == 0) return util;

    util[n - 1] = true;

    for (Eigen::Index i = n - 1; i >= 0; --i) {
        if (!util[i]) continue;

        uint8_t node = stack(i, 0);
        if (IS_TERMINAL[node]) continue;

        // Mark param1
        util[stack(i, 1)] = true;

        // Mark param2 if arity-2
        if (IS_ARITY_2[node]) {
            util[stack(i, 2)] = true;
        }
    }
    return util;
}

// -----------------------------------------------------------------
//  reduce
// -----------------------------------------------------------------

ReduceResult reduce(
        const StackMatrix& raw_stack,
        const std::vector<double>& raw_constants,
        const std::vector<int>& raw_integers) {
    const Eigen::Index n = raw_stack.rows();

    // Handle empty stack
    if (n == 0) {
        return {
            StackMatrix(0, 3),
            {},
            {},
            {},
        };
    }

    // Step 1: find utilized commands
    auto used = get_utilized_commands(raw_stack);

    // Step 2: count utilized and build reduced_map (old index → new index)
    // reduced_map[i] = cumsum(used) - 1
    std::vector<int> reduced_map(n);
    int count = 0;
    for (Eigen::Index i = 0; i < n; ++i) {
        if (used[i]) ++count;
        reduced_map[i] = count - 1;
    }

    // Step 3: build the reduced stack
    StackMatrix stack(count, 3);
    std::vector<double> new_constants;
    std::vector<int> new_integers;
    std::vector<int> constant_mapping;  // new_idx → raw_idx

    int j = 0;
    for (Eigen::Index i = 0; i < n; ++i) {
        if (!used[i]) continue;

        uint8_t node = raw_stack(i, 0);
        stack(j, 0) = node;

        if (IS_TERMINAL[node]) {
            if (node == static_cast<uint8_t>(Op::CONSTANT)) {
                int old_idx = static_cast<int>(raw_stack(i, 1));
                int new_idx = static_cast<int>(new_constants.size());
                double value = (old_idx < static_cast<int>(raw_constants.size()))
                                   ? raw_constants[old_idx]
                                   : 1.0;
                new_constants.push_back(value);
                constant_mapping.push_back(old_idx);
                stack(j, 1) = static_cast<uint8_t>(new_idx);
                stack(j, 2) = static_cast<uint8_t>(new_idx);
            } else if (node == static_cast<uint8_t>(Op::INTEGER)) {
                int old_idx = static_cast<int>(raw_stack(i, 1));
                int new_idx = static_cast<int>(new_integers.size());
                int value = (old_idx < static_cast<int>(raw_integers.size()))
                                ? raw_integers[old_idx]
                                : 0;
                new_integers.push_back(value);
                stack(j, 1) = static_cast<uint8_t>(new_idx);
                stack(j, 2) = static_cast<uint8_t>(new_idx);
            } else {
                // VARIABLE — pass through param unchanged
                stack(j, 1) = raw_stack(i, 1);
                stack(j, 2) = raw_stack(i, 2);
            }
        } else {
            // Non-terminal: remap row references
            stack(j, 1) = static_cast<uint8_t>(
                reduced_map[static_cast<int>(raw_stack(i, 1))]);

            if (IS_ARITY_2[node]) {
                stack(j, 2) = static_cast<uint8_t>(
                    reduced_map[static_cast<int>(raw_stack(i, 2))]);
            } else {
                // Unary: param2 = param1
                stack(j, 2) = stack(j, 1);
            }
        }
        ++j;
    }

    return {std::move(stack), std::move(new_constants),
            std::move(new_integers), std::move(constant_mapping)};
}

}  // namespace cppagraph
