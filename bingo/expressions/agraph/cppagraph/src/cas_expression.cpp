/**
 * @file cas_expression.cpp
 * @brief CASExpression implementation — port of cas_expression.py.
 */

#include "cppagraph/cas_expression.h"

#include <algorithm>
#include <cassert>
#include <sstream>
#include <unordered_map>

namespace cppagraph {

static inline uint8_t u8(Op o) { return static_cast<uint8_t>(o); }

// ------------------------------------------------------------------ //
//  Operator sort-key table (matches Python _OPERATOR_ORDER)           //
// ------------------------------------------------------------------ //

static const std::unordered_map<uint8_t, int> OPERATOR_ORDER_MAP = {
    {u8(Op::INTEGER),        -1},
    {u8(Op::VARIABLE),        0},
    {u8(Op::CONSTANT),        1},
    {u8(Op::ADDITION),        2},
    {u8(Op::SUBTRACTION),     3},
    {u8(Op::MULTIPLICATION),  4},
    {u8(Op::DIVISION),        5},
    {u8(Op::SIN),             6},
    {u8(Op::COS),             7},
    {u8(Op::EXPONENTIAL),     8},
    {u8(Op::LOGARITHM),       9},
    {u8(Op::POWER),          10},
    {u8(Op::ABS),            11},
    {u8(Op::SQRT),           12},
    {u8(Op::SAFE_POWER),     13},
    {u8(Op::SINH),           14},
    {u8(Op::COSH),           15},
    {u8(Op::TAN),            16},
    {u8(Op::ARCSIN),         17},
    {u8(Op::ARCCOS),         18},
    {u8(Op::ARCTAN),         19},
    {u8(Op::TANH),           20},
    {u8(Op::SQUARE),         21},
    {u8(Op::CUBE),           22},
};

int operator_order(uint8_t op) {
    auto it = OPERATOR_ORDER_MAP.find(op);
    return it != OPERATOR_ORDER_MAP.end() ? it->second : 99;
}

// ------------------------------------------------------------------ //
//  Interned singletons                                                //
// ------------------------------------------------------------------ //

static CASExprPtr g_zero;
static CASExprPtr g_one;
static CASExprPtr g_two;
static CASExprPtr g_neg_one;
static std::vector<CASExprPtr> g_variable_cache;
static bool g_singletons_init = false;

static void init_singletons() {
    if (g_singletons_init) return;
    g_zero    = std::make_shared<CASExpression>(u8(Op::INTEGER), 0);
    g_one     = std::make_shared<CASExpression>(u8(Op::INTEGER), 1);
    g_two     = std::make_shared<CASExpression>(u8(Op::INTEGER), 2);
    g_neg_one = std::make_shared<CASExpression>(u8(Op::INTEGER), -1);
    g_variable_cache.reserve(8);
    for (int i = 0; i < 8; ++i)
        g_variable_cache.push_back(
            std::make_shared<CASExpression>(u8(Op::VARIABLE), i));
    g_singletons_init = true;
}

CASExprPtr cas_zero()    { init_singletons(); return g_zero; }
CASExprPtr cas_one()     { init_singletons(); return g_one; }
CASExprPtr cas_two()     { init_singletons(); return g_two; }
CASExprPtr cas_neg_one() { init_singletons(); return g_neg_one; }

CASExprPtr interned_integer(int value) {
    init_singletons();
    if (value == 0)  return g_zero;
    if (value == 1)  return g_one;
    if (value == 2)  return g_two;
    if (value == -1) return g_neg_one;
    return std::make_shared<CASExpression>(u8(Op::INTEGER), value);
}

CASExprPtr interned_variable(int index) {
    init_singletons();
    if (index >= 0 && index < static_cast<int>(g_variable_cache.size()))
        return g_variable_cache[index];
    return std::make_shared<CASExpression>(u8(Op::VARIABLE), index);
}

// ------------------------------------------------------------------ //
//  Construction                                                       //
// ------------------------------------------------------------------ //

CASExpression::CASExpression(uint8_t op, int param)
    : _op(op), _terminal_param(param) {}

CASExpression::CASExpression(uint8_t op, std::vector<CASExprPtr> operands)
    : _op(op), _terminal_param(0), _operands(std::move(operands)) {}

// ------------------------------------------------------------------ //
//  Predicates                                                         //
// ------------------------------------------------------------------ //

bool CASExpression::is_zero() const {
    return _op == u8(Op::INTEGER) && _terminal_param == 0;
}

bool CASExpression::is_one() const {
    return _op == u8(Op::INTEGER) && _terminal_param == 1;
}

bool CASExpression::is_constant_valued() {
    if (_is_constant_valued_cache == Tri::YES) return true;
    if (_is_constant_valued_cache == Tri::NO) return false;
    bool val = _compute_is_constant_valued();
    _is_constant_valued_cache = val ? Tri::YES : Tri::NO;
    return val;
}

bool CASExpression::_compute_is_constant_valued() {
    if (_op == u8(Op::INTEGER) || _op == u8(Op::CONSTANT)) return true;
    if (_op == u8(Op::VARIABLE)) return false;
    for (auto& child : _operands)
        if (!child->is_constant_valued()) return false;
    return true;
}

const CASExpression::DepSet& CASExpression::depends_on() {
    if (!_deps_computed) {
        _depends_on = _compute_depends_on();
        _deps_computed = true;
    }
    return _depends_on;
}

CASExpression::DepSet CASExpression::_compute_depends_on() {
    if (_op == u8(Op::INTEGER))  return {std::string("i")};
    if (_op == u8(Op::VARIABLE)) return {std::string("x")};
    if (_op == u8(Op::CONSTANT)) return {_terminal_param};
    DepSet result;
    for (auto& child : _operands) {
        auto& child_deps = child->depends_on();
        result.insert(child_deps.begin(), child_deps.end());
    }
    return result;
}

// ------------------------------------------------------------------ //
//  Algebraic properties                                               //
// ------------------------------------------------------------------ //

CASExprPtr CASExpression::base() {
    if (!_base_set) {
        _base_set = true;
        if (_op == u8(Op::POWER)) {
            _base = _operands[0];
        } else if (_op == u8(Op::INTEGER)) {
            _base = nullptr;
        } else {
            _base = shared_from_this();
        }
    }
    return _base;
}

CASExprPtr CASExpression::exponent() {
    if (!_exponent_set) {
        _exponent_set = true;
        if (_op == u8(Op::POWER)) {
            _exponent = _operands[1];
        } else if (_op == u8(Op::INTEGER)) {
            _exponent = nullptr;
        } else {
            _exponent = cas_one();
        }
    }
    return _exponent;
}

CASExprPtr CASExpression::term() {
    if (!_term_set) {
        _term_set = true;
        if (_op == u8(Op::MULTIPLICATION)) {
            auto& first = _operands[0];
            if (first->op() == u8(Op::INTEGER) || first->op() == u8(Op::CONSTANT)) {
                std::vector<CASExprPtr> tail(_operands.begin() + 1, _operands.end());
                _term = std::make_shared<CASExpression>(u8(Op::MULTIPLICATION), std::move(tail));
            } else {
                _term = shared_from_this();
            }
        } else if (_op == u8(Op::INTEGER)) {
            _term = nullptr;
        } else {
            _term = std::make_shared<CASExpression>(
                u8(Op::MULTIPLICATION),
                std::vector<CASExprPtr>{shared_from_this()});
        }
    }
    return _term;
}

CASExprPtr CASExpression::coefficient() {
    if (!_coefficient_set) {
        _coefficient_set = true;
        if (_op == u8(Op::MULTIPLICATION) &&
            (_operands[0]->op() == u8(Op::INTEGER) ||
             _operands[0]->op() == u8(Op::CONSTANT))) {
            _coefficient = _operands[0];
        } else if (_op == u8(Op::INTEGER)) {
            _coefficient = nullptr;
        } else {
            _coefficient = cas_one();
        }
    }
    return _coefficient;
}

// ------------------------------------------------------------------ //
//  Comparison / ordering                                              //
// ------------------------------------------------------------------ //

bool CASExpression::operator==(const CASExpression& other) const {
    if (this == &other) return true;
    if (_op != other._op) return false;
    if (IS_TERMINAL[_op]) return _terminal_param == other._terminal_param;
    if (_operands.size() != other._operands.size()) return false;
    for (size_t i = 0; i < _operands.size(); ++i)
        if (!(*_operands[i] == *other._operands[i])) return false;
    return true;
}

bool CASExpression::operator<(const CASExpression& other) const {
    // Need non-const for is_constant_valued; use const_cast (safe
    // since is_constant_valued only mutates cache).
    auto& self_nc = const_cast<CASExpression&>(*this);
    auto& other_nc = const_cast<CASExpression&>(other);
    if (self_nc.is_constant_valued() || other_nc.is_constant_valued())
        return _constant_lt(other);
    uint8_t s_op = _op, o_op = other._op;
    if (s_op == u8(Op::MULTIPLICATION) || o_op == u8(Op::MULTIPLICATION))
        return _associative_lt(other, u8(Op::MULTIPLICATION));
    if (s_op == u8(Op::POWER) || o_op == u8(Op::POWER))
        return _power_lt(other);
    if (s_op == u8(Op::ADDITION) || o_op == u8(Op::ADDITION))
        return _associative_lt(other, u8(Op::ADDITION));
    return _general_lt(other);
}

bool CASExpression::_constant_lt(const CASExpression& other) const {
    auto& self_nc = const_cast<CASExpression&>(*this);
    auto& other_nc = const_cast<CASExpression&>(other);
    if (self_nc.is_constant_valued() != other_nc.is_constant_valued())
        return self_nc.is_constant_valued();
    return _general_lt(other);
}

bool CASExpression::_general_lt(const CASExpression& other) const {
    if (_op != other._op)
        return operator_order(_op) < operator_order(other._op);
    if (IS_TERMINAL[_op])
        return _terminal_param < other._terminal_param;
    return _operands_lt(_operands, other._operands);
}

bool CASExpression::_operands_lt(const std::vector<CASExprPtr>& a,
                                  const std::vector<CASExprPtr>& b) {
    // Compare from the end (reversed iteration, matching Python)
    auto ai = a.rbegin(), bi = b.rbegin();
    for (; ai != a.rend() && bi != b.rend(); ++ai, ++bi) {
        if (!(**ai == **bi)) return **ai < **bi;
    }
    return a.size() < b.size();
}

bool CASExpression::_associative_lt(const CASExpression& other,
                                     uint8_t assoc_op) const {
    if (_op == assoc_op) {
        if (other._op == assoc_op)
            return _operands_lt(_operands, other._operands);
        return _operands_lt(_operands, {const_cast<CASExpression&>(other).shared_from_this()});
    }
    return _operands_lt(
        {const_cast<CASExpression&>(*this).shared_from_this()},
        other._operands);
}

bool CASExpression::_power_lt(const CASExpression& other) const {
    auto s_base = const_cast<CASExpression*>(this)->base();
    auto s_exp  = const_cast<CASExpression*>(this)->exponent();
    auto o_base = const_cast<CASExpression*>(&other)->base();
    auto o_exp  = const_cast<CASExpression*>(&other)->exponent();
    if (*s_base == *o_base)
        return *s_exp < *o_exp;
    return *s_base < *o_base;
}

// ------------------------------------------------------------------ //
//  same_term                                                          //
// ------------------------------------------------------------------ //

bool CASExpression::same_term(const CASExpression& other) {
    if (_op == u8(Op::INTEGER) || other._op == u8(Op::INTEGER))
        return false;

    bool s_has_coeff = (_op == u8(Op::MULTIPLICATION) &&
                        (_operands[0]->op() == u8(Op::INTEGER) ||
                         _operands[0]->op() == u8(Op::CONSTANT)));
    bool o_has_coeff = (other._op == u8(Op::MULTIPLICATION) &&
                        (other._operands[0]->op() == u8(Op::INTEGER) ||
                         other._operands[0]->op() == u8(Op::CONSTANT)));

    auto tail_eq = [](const std::vector<CASExprPtr>& a, size_t a_start,
                      const std::vector<CASExprPtr>& b, size_t b_start) {
        if (a.size() - a_start != b.size() - b_start) return false;
        for (size_t i = 0; i < a.size() - a_start; ++i)
            if (!(*a[a_start + i] == *b[b_start + i])) return false;
        return true;
    };

    if (s_has_coeff && o_has_coeff)
        return tail_eq(_operands, 1, other._operands, 1);

    if (s_has_coeff && !o_has_coeff) {
        if (other._op == u8(Op::MULTIPLICATION))
            return tail_eq(_operands, 1, other._operands, 0);
        return _operands.size() == 2 && *_operands[1] == other;
    }

    if (!s_has_coeff && o_has_coeff) {
        if (_op == u8(Op::MULTIPLICATION))
            return tail_eq(_operands, 0, other._operands, 1);
        return other._operands.size() == 2 && *other._operands[1] == *this;
    }

    return *this == other;
}

// ------------------------------------------------------------------ //
//  Hash                                                               //
// ------------------------------------------------------------------ //

std::size_t CASExpression::hash() const {
    if (_hash_computed) return _hash_value;
    std::size_t h = std::hash<uint8_t>{}(_op);
    if (IS_TERMINAL[_op]) {
        h = h * 31 ^ std::hash<int>{}(_terminal_param);
    } else {
        for (auto& child : _operands)
            h = h * 31 ^ child->hash();
    }
    _hash_value = h;
    _hash_computed = true;
    return h;
}

// ------------------------------------------------------------------ //
//  Utility                                                            //
// ------------------------------------------------------------------ //

CASExprPtr CASExpression::map(std::function<CASExprPtr(const CASExprPtr&)> fn) const {
    if (IS_TERMINAL[_op])
        return const_cast<CASExpression*>(this)->shared_from_this();
    std::vector<CASExprPtr> mapped;
    mapped.reserve(_operands.size());
    bool changed = false;
    for (auto& child : _operands) {
        auto m = fn(child);
        if (m.get() != child.get()) changed = true;
        mapped.push_back(std::move(m));
    }
    if (!changed) return const_cast<CASExpression*>(this)->shared_from_this();
    return std::make_shared<CASExpression>(_op, std::move(mapped));
}

CASExprPtr CASExpression::copy() const {
    if (IS_TERMINAL[_op])
        return std::make_shared<CASExpression>(_op, _terminal_param);
    return std::make_shared<CASExpression>(_op, _operands);
}

std::string CASExpression::to_string() const {
    std::ostringstream ss;
    ss << int(_op) << "(";
    if (IS_TERMINAL[_op]) {
        ss << _terminal_param;
    } else {
        for (size_t i = 0; i < _operands.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << _operands[i]->to_string();
        }
    }
    ss << ")";
    return ss.str();
}

}  // namespace cppagraph
