/**
 * @file cas_expression.h
 * @brief CAS expression tree node for algebraic simplification.
 *
 * Port of pyagraph/simplification/cas_expression.py.
 * CASExpression is a recursive tree node used purely for symbolic
 * manipulation in the CAS simplification pipeline.
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <variant>
#include <vector>

#include "cppagraph/operators.h"

namespace cppagraph {

class CASExpression;
using CASExprPtr = std::shared_ptr<CASExpression>;

/**
 * A mathematical expression in tree form.
 *
 * Terminals store an int parameter (constant index, integer value,
 * or variable column).  Non-terminals store child CASExprPtr operands.
 */
class CASExpression : public std::enable_shared_from_this<CASExpression> {
public:
    /// Construct a terminal node.
    CASExpression(uint8_t op, int param);

    /// Construct a non-terminal node.
    CASExpression(uint8_t op, std::vector<CASExprPtr> operands);

    // ---- Accessors ----

    uint8_t op() const { return _op; }
    int terminal_param() const { return _terminal_param; }
    const std::vector<CASExprPtr>& operands() const { return _operands; }

    // ---- Algebraic properties (lazily cached) ----

    CASExprPtr base();
    CASExprPtr exponent();
    CASExprPtr term();
    CASExprPtr coefficient();

    // ---- Predicates ----

    bool is_zero() const;
    bool is_one() const;
    bool is_constant_valued();

    /// The set of dependency tags: "i", "x", or constant indices.
    using DepSet = std::set<std::variant<std::string, int>>;
    const DepSet& depends_on();

    // ---- Comparison / hashing ----

    bool operator==(const CASExpression& other) const;
    bool operator!=(const CASExpression& other) const { return !(*this == other); }
    bool operator<(const CASExpression& other) const;
    std::size_t hash() const;

    bool same_term(const CASExpression& other);

    // ---- Utility ----

    CASExprPtr map(std::function<CASExprPtr(const CASExprPtr&)> fn) const;
    CASExprPtr copy() const;
    std::string to_string() const;

private:
    uint8_t _op;
    int _terminal_param;                ///< For terminals only.
    std::vector<CASExprPtr> _operands;  ///< For non-terminals only.

    // Cached algebraic properties (std::nullopt = not yet computed)
    enum class Tri { UNSET, YES, NO };
    Tri _is_constant_valued_cache = Tri::UNSET;

    bool _deps_computed = false;
    DepSet _depends_on;

    mutable bool _hash_computed = false;
    mutable std::size_t _hash_value = 0;

    // Algebraic property caches (nullptr = not yet computed, "sentinel")
    static constexpr int PROP_UNSET = -999;
    bool _base_set = false;
    CASExprPtr _base;
    bool _exponent_set = false;
    CASExprPtr _exponent;
    bool _term_set = false;
    CASExprPtr _term;
    bool _coefficient_set = false;
    CASExprPtr _coefficient;

    // Internal helpers
    bool _compute_is_constant_valued();
    DepSet _compute_depends_on();
    bool _constant_lt(const CASExpression& other) const;
    bool _general_lt(const CASExpression& other) const;
    bool _power_lt(const CASExpression& other) const;
    bool _associative_lt(const CASExpression& other, uint8_t assoc_op) const;
    static bool _operands_lt(const std::vector<CASExprPtr>& a,
                             const std::vector<CASExprPtr>& b);
};

// ---- Interned singletons ----

CASExprPtr interned_integer(int value);
CASExprPtr interned_variable(int index);
CASExprPtr cas_one();
CASExprPtr cas_zero();
CASExprPtr cas_two();
CASExprPtr cas_neg_one();

// ---- Operator sort-key ----
int operator_order(uint8_t op);

}  // namespace cppagraph

// Hash support for std::unordered_map etc.
namespace std {
template <>
struct hash<cppagraph::CASExprPtr> {
    size_t operator()(const cppagraph::CASExprPtr& p) const {
        return p ? p->hash() : 0;
    }
};
}  // namespace std
