/**
 * @file operators.h
 * @brief Operator ID enumeration and compile-time property tables.
 *
 * Mirrors bingo.expressions.agraph.pyagraph.operators exactly.
 * Every operator ID, property set, and name table entry matches
 * the Python reference implementation.
 */

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cppagraph {

// ---- Operator IDs (uint8_t, matching pyagraph exactly) ----

enum class Op : uint8_t {
    // Terminals (0–2)
    VARIABLE     = 0,
    CONSTANT     = 1,
    INTEGER      = 2,

    // Arithmetic (3–6)
    ADDITION       = 3,
    SUBTRACTION    = 4,
    MULTIPLICATION = 5,
    DIVISION       = 6,

    // Power / Root (7–11)
    POWER      = 7,
    SAFE_POWER = 8,
    SQUARE     = 9,
    CUBE       = 10,
    SQRT       = 11,

    // Miscellaneous (12)
    ABS = 12,

    // Exponential / Logarithmic (13–14)
    EXPONENTIAL = 13,
    LOGARITHM   = 14,

    // Trigonometric (15–20)
    SIN    = 15,
    COS    = 16,
    TAN    = 17,
    ARCSIN = 18,
    ARCCOS = 19,
    ARCTAN = 20,

    // Hyperbolic (21–23)
    SINH = 21,
    COSH = 22,
    TANH = 23,
};

/// Total number of defined operators.
inline constexpr std::size_t NUM_OPS = 24;

// ---- Compile-time boolean lookup arrays (indexed by raw Op value) ----

/// True for terminal operators (VARIABLE, CONSTANT, INTEGER).
inline constexpr std::array<bool, NUM_OPS> IS_TERMINAL = [] {
    std::array<bool, NUM_OPS> a{};
    a[static_cast<uint8_t>(Op::VARIABLE)] = true;
    a[static_cast<uint8_t>(Op::CONSTANT)] = true;
    a[static_cast<uint8_t>(Op::INTEGER)]  = true;
    return a;
}();

/// True for binary (arity-2) operators.
inline constexpr std::array<bool, NUM_OPS> IS_ARITY_2 = [] {
    std::array<bool, NUM_OPS> a{};
    a[static_cast<uint8_t>(Op::ADDITION)]       = true;
    a[static_cast<uint8_t>(Op::SUBTRACTION)]    = true;
    a[static_cast<uint8_t>(Op::MULTIPLICATION)] = true;
    a[static_cast<uint8_t>(Op::DIVISION)]       = true;
    a[static_cast<uint8_t>(Op::POWER)]          = true;
    a[static_cast<uint8_t>(Op::SAFE_POWER)]     = true;
    return a;
}();

// ---- Runtime helpers ----

/// Return the set of terminal operator IDs.
const std::unordered_set<uint8_t>& terminal_ids();

/// Return the set of binary (arity-2) operator IDs.
const std::unordered_set<uint8_t>& arity_2_ids();

/// Return the mapping from operator ID to list of common names.
const std::unordered_map<uint8_t, std::vector<std::string>>& operator_names();

}  // namespace cppagraph
