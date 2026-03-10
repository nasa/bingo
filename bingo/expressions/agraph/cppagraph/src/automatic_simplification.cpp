/**
 * @file automatic_simplification.cpp
 * @brief CAS automatic simplification — port of automatic_simplification.py.
 */

#include "cppagraph/automatic_simplification.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace cppagraph {

static inline uint8_t u8(Op o) { return static_cast<uint8_t>(o); }

// Forward declarations of internal helpers.
static CASExprPtr _simplify_constant_power(
    const CASExprPtr& base, const CASExprPtr& exponent);
static std::vector<CASExprPtr> _simplify_product_rec(
    const std::vector<CASExprPtr>& operands);
static std::vector<CASExprPtr> _merge_products(
    const std::vector<CASExprPtr>& operands_1,
    const std::vector<CASExprPtr>& operands_2);
static std::vector<CASExprPtr> _simplify_sum_rec(
    const std::vector<CASExprPtr>& operands);
static std::vector<CASExprPtr> _merge_sums(
    const std::vector<CASExprPtr>& operands_1,
    const std::vector<CASExprPtr>& operands_2);
static CASExprPtr no_simplification(const CASExprPtr& expression);

// Module-level aliases for readability.
static CASExprPtr NEGATIVE_ONE() { return cas_neg_one(); }
static CASExprPtr ZERO()         { return cas_zero(); }
static CASExprPtr ONE()          { return cas_one(); }
static CASExprPtr TWO()          { return cas_two(); }

// ================================================================== //
//  Dispatch table                                                     //
// ================================================================== //

static bool _contains_zero(const std::vector<CASExprPtr>& ops) {
    for (auto& op : ops)
        if (op->is_zero()) return true;
    return false;
}

// ================================================================== //
//  Power                                                              //
// ================================================================== //

CASExprPtr simplify_power(const CASExprPtr& expression) {
    auto& ops = expression->operands();
    auto& base = ops[0];
    auto& exponent = ops[1];

    if (base->is_one()) return ONE();
    if (base->is_zero() && exponent->op() == u8(Op::INTEGER)
        && exponent->terminal_param() > 0)
        return ZERO();
    if (exponent->op() == u8(Op::INTEGER) || exponent->op() == u8(Op::CONSTANT))
        return _simplify_constant_power(base, exponent);
    return expression;
}

static CASExprPtr _simplify_constant_power(
    const CASExprPtr& base, const CASExprPtr& exponent)
{
    if (exponent->is_one()) return base;
    if (exponent->is_zero()) return ONE();

    // Integer^Integer (positive exponent) → compute directly.
    if (base->op() == u8(Op::INTEGER)
        && exponent->op() == u8(Op::INTEGER)
        && exponent->terminal_param() > 0) {
        int b = base->terminal_param();
        int e = exponent->terminal_param();
        int result = 1;
        for (int i = 0; i < e; ++i) result *= b;
        return interned_integer(result);
    }

    // (base^base_exp)^exponent → base^(base_exp * exponent)
    if (base->op() == u8(Op::POWER)) {
        auto& base_base = base->operands()[0];
        auto& base_exponent = base->operands()[1];
        auto mult_exp = std::make_shared<CASExpression>(
            u8(Op::MULTIPLICATION),
            std::vector<CASExprPtr>{base_exponent, exponent});
        auto new_exponent = simplify_product(mult_exp);
        if (base_exponent->op() == u8(Op::INTEGER)
            || base_exponent->op() == u8(Op::CONSTANT))
            return _simplify_constant_power(base_base, new_exponent);
        return std::make_shared<CASExpression>(
            u8(Op::POWER),
            std::vector<CASExprPtr>{base_base, new_exponent});
    }

    // (a*b*c)^exponent → a^exponent * b^exponent * c^exponent
    if (base->op() == u8(Op::MULTIPLICATION)) {
        auto mapped = base->map([&](const CASExprPtr& bas) {
            return _simplify_constant_power(bas, exponent);
        });
        return simplify_product(mapped);
    }

    return std::make_shared<CASExpression>(
        u8(Op::POWER), std::vector<CASExprPtr>{base, exponent});
}

// ================================================================== //
//  Product                                                            //
// ================================================================== //

CASExprPtr simplify_product(const CASExprPtr& expression) {
    auto& operands = expression->operands();
    if (_contains_zero(operands)) return ZERO();
    if (operands.size() == 1) return operands[0];

    auto simplified = _simplify_product_rec(operands);
    if (simplified.empty()) return ONE();
    if (simplified.size() == 1) return simplified[0];
    return std::make_shared<CASExpression>(
        u8(Op::MULTIPLICATION), std::move(simplified));
}

static std::vector<CASExprPtr> _simplify_product_rec(
    const std::vector<CASExprPtr>& operands)
{
    if (operands.size() == 2) {
        auto& op_1 = operands[0];
        auto& op_2 = operands[1];

        // Integer * Integer
        if (op_1->op() == u8(Op::INTEGER) && op_2->op() == u8(Op::INTEGER)) {
            int new_int = op_1->terminal_param() * op_2->terminal_param();
            auto result = interned_integer(new_int);
            if (result->is_one()) return {};
            return {result};
        }

        if (op_1->op() != u8(Op::MULTIPLICATION)
            && op_2->op() != u8(Op::MULTIPLICATION)) {
            if (op_1->is_one()) return {op_2};
            if (op_2->is_one()) return {op_1};

            auto b1 = op_1->base();
            auto b2 = op_2->base();
            if (b1 && b2 && *b1 == *b2) {
                auto e1 = op_1->exponent();
                auto e2 = op_2->exponent();
                CASExprPtr new_exponent;
                // Fast path: 1+1=2
                if (e1.get() == cas_one().get() && e2.get() == cas_one().get()) {
                    new_exponent = TWO();
                } else {
                    auto sum_expr = std::make_shared<CASExpression>(
                        u8(Op::ADDITION),
                        std::vector<CASExprPtr>{e1, e2});
                    new_exponent = simplify_sum(sum_expr);
                }
                auto combined = std::make_shared<CASExpression>(
                    u8(Op::POWER),
                    std::vector<CASExprPtr>{op_1->base(), new_exponent});
                auto result = simplify_power(combined);
                if (result->is_one()) return {};
                return {result};
            }

            if (*op_2 < *op_1) return {op_2, op_1};
            return {op_1, op_2};
        }

        // At least one is MULTIPLICATION — flatten and merge.
        std::vector<CASExprPtr> to_merge_1, to_merge_2;
        if (op_1->op() == u8(Op::MULTIPLICATION))
            to_merge_1 = op_1->operands();
        else
            to_merge_1 = {op_1};
        if (op_2->op() == u8(Op::MULTIPLICATION))
            to_merge_2 = op_2->operands();
        else
            to_merge_2 = {op_2};
        return _merge_products(to_merge_1, to_merge_2);
    }

    // More than 2 operands: recursive case.
    std::vector<CASExprPtr> rest(operands.begin() + 1, operands.end());
    auto rest_simplified = _simplify_product_rec(rest);
    if (operands[0]->op() == u8(Op::MULTIPLICATION))
        return _merge_products(operands[0]->operands(), rest_simplified);
    return _merge_products({operands[0]}, rest_simplified);
}

static std::vector<CASExprPtr> _merge_products(
    const std::vector<CASExprPtr>& operands_1,
    const std::vector<CASExprPtr>& operands_2)
{
    std::vector<CASExprPtr> result;
    size_t i = 0, j = 0;
    size_t n1 = operands_1.size(), n2 = operands_2.size();
    while (i < n1 && j < n2) {
        auto simplified = _simplify_product_rec({operands_1[i], operands_2[j]});
        auto slen = simplified.size();
        if (slen == 0) {
            ++i; ++j;
        } else if (slen == 1) {
            result.push_back(simplified[0]);
            ++i; ++j;
        } else if (simplified[0].get() == operands_1[i].get()
                   || *simplified[0] == *operands_1[i]) {
            result.push_back(simplified[0]);
            ++i;
        } else {
            result.push_back(simplified[0]);
            ++j;
        }
    }
    for (; i < n1; ++i) result.push_back(operands_1[i]);
    for (; j < n2; ++j) result.push_back(operands_2[j]);
    return result;
}

// ================================================================== //
//  Sum                                                                //
// ================================================================== //

CASExprPtr simplify_sum(const CASExprPtr& expression) {
    auto& operands = expression->operands();
    if (operands.size() == 1) return operands[0];

    auto simplified = _simplify_sum_rec(operands);
    if (simplified.empty()) return ZERO();
    if (simplified.size() == 1) return simplified[0];
    return std::make_shared<CASExpression>(
        u8(Op::ADDITION), std::move(simplified));
}

static std::vector<CASExprPtr> _simplify_sum_rec(
    const std::vector<CASExprPtr>& operands)
{
    if (operands.size() == 2) {
        auto& op_1 = operands[0];
        auto& op_2 = operands[1];

        // Integer + Integer
        if (op_1->op() == u8(Op::INTEGER) && op_2->op() == u8(Op::INTEGER)) {
            int new_int = op_1->terminal_param() + op_2->terminal_param();
            auto result = interned_integer(new_int);
            if (result->is_zero()) return {};
            return {result};
        }

        if (op_1->op() != u8(Op::ADDITION)
            && op_2->op() != u8(Op::ADDITION)) {
            if (op_1->is_zero()) return {op_2};
            if (op_2->is_zero()) return {op_1};

            if (op_1->same_term(*op_2)) {
                auto c1 = op_1->coefficient();
                auto c2 = op_2->coefficient();
                CASExprPtr new_coefficient;
                // Fast path: 1+1=2
                if (c1.get() == cas_one().get() && c2.get() == cas_one().get()) {
                    new_coefficient = TWO();
                } else {
                    auto sum_expr = std::make_shared<CASExpression>(
                        u8(Op::ADDITION),
                        std::vector<CASExprPtr>{c1, c2});
                    new_coefficient = simplify_sum(sum_expr);
                }
                auto combined = std::make_shared<CASExpression>(
                    u8(Op::MULTIPLICATION),
                    std::vector<CASExprPtr>{new_coefficient, op_1->term()});
                auto result = simplify_product(combined);
                if (result->is_zero()) return {};
                return {result};
            }

            if (*op_2 < *op_1) return {op_2, op_1};
            return {op_1, op_2};
        }

        // At least one is ADDITION — flatten and merge.
        std::vector<CASExprPtr> to_merge_1, to_merge_2;
        if (op_1->op() == u8(Op::ADDITION))
            to_merge_1 = op_1->operands();
        else
            to_merge_1 = {op_1};
        if (op_2->op() == u8(Op::ADDITION))
            to_merge_2 = op_2->operands();
        else
            to_merge_2 = {op_2};
        return _merge_sums(to_merge_1, to_merge_2);
    }

    // More than 2 operands: recursive case.
    std::vector<CASExprPtr> rest(operands.begin() + 1, operands.end());
    auto rest_simplified = _simplify_sum_rec(rest);
    if (operands[0]->op() == u8(Op::ADDITION))
        return _merge_sums(operands[0]->operands(), rest_simplified);
    return _merge_sums({operands[0]}, rest_simplified);
}

static std::vector<CASExprPtr> _merge_sums(
    const std::vector<CASExprPtr>& operands_1,
    const std::vector<CASExprPtr>& operands_2)
{
    std::vector<CASExprPtr> result;
    size_t i = 0, j = 0;
    size_t n1 = operands_1.size(), n2 = operands_2.size();
    while (i < n1 && j < n2) {
        auto simplified = _simplify_sum_rec({operands_1[i], operands_2[j]});
        auto slen = simplified.size();
        if (slen == 0) {
            ++i; ++j;
        } else if (slen == 1) {
            result.push_back(simplified[0]);
            ++i; ++j;
        } else if (simplified[0].get() == operands_1[i].get()
                   || *simplified[0] == *operands_1[i]) {
            result.push_back(simplified[0]);
            ++i;
        } else {
            result.push_back(simplified[0]);
            ++j;
        }
    }
    for (; i < n1; ++i) result.push_back(operands_1[i]);
    for (; j < n2; ++j) result.push_back(operands_2[j]);
    return result;
}

// ================================================================== //
//  Quotient and Difference                                            //
// ================================================================== //

CASExprPtr simplify_quotient(const CASExprPtr& expression) {
    auto& ops = expression->operands();
    auto& numerator = ops[0];
    auto& denominator = ops[1];
    auto denom_inv = std::make_shared<CASExpression>(
        u8(Op::POWER),
        std::vector<CASExprPtr>{denominator, NEGATIVE_ONE()});
    denom_inv = simplify_power(denom_inv);
    auto product = std::make_shared<CASExpression>(
        u8(Op::MULTIPLICATION),
        std::vector<CASExprPtr>{numerator, denom_inv});
    return simplify_product(product);
}

CASExprPtr simplify_difference(const CASExprPtr& expression) {
    auto& ops = expression->operands();
    auto& first = ops[0];
    auto& second = ops[1];

    std::vector<CASExprPtr> new_operands = {first};
    if (second->op() == u8(Op::ADDITION)) {
        for (auto& operand : second->operands()) {
            auto neg = std::make_shared<CASExpression>(
                u8(Op::MULTIPLICATION),
                std::vector<CASExprPtr>{NEGATIVE_ONE(), operand});
            new_operands.push_back(simplify_product(neg));
        }
    } else {
        auto neg = std::make_shared<CASExpression>(
            u8(Op::MULTIPLICATION),
            std::vector<CASExprPtr>{NEGATIVE_ONE(), second});
        new_operands.push_back(simplify_product(neg));
    }
    auto diff_as_sum = std::make_shared<CASExpression>(
        u8(Op::ADDITION), std::move(new_operands));
    return simplify_sum(diff_as_sum);
}

// ================================================================== //
//  Trigonometric                                                      //
// ================================================================== //

static CASExprPtr simplify_sin(const CASExprPtr& expression) {
    auto& operand = expression->operands()[0];
    if (operand->is_zero()) return ZERO();
    if (operand->op() == u8(Op::ARCSIN)) return operand->operands()[0];
    return expression;
}

static CASExprPtr simplify_cos(const CASExprPtr& expression) {
    auto& operand = expression->operands()[0];
    if (operand->is_zero()) return ONE();
    if (operand->op() == u8(Op::ARCCOS)) return operand->operands()[0];
    return expression;
}

static CASExprPtr simplify_tan(const CASExprPtr& expression) {
    auto& operand = expression->operands()[0];
    if (operand->is_zero()) return ZERO();
    if (operand->op() == u8(Op::ARCTAN)) return operand->operands()[0];
    return expression;
}

static CASExprPtr simplify_logarithm(const CASExprPtr& expression) {
    auto& operand = expression->operands()[0];
    if (operand->is_one()) return ZERO();
    if (operand->op() == u8(Op::EXPONENTIAL)) return operand->operands()[0];
    return expression;
}

static CASExprPtr simplify_exponential(const CASExprPtr& expression) {
    auto& operand = expression->operands()[0];
    if (operand->is_zero()) return ONE();
    if (operand->op() == u8(Op::LOGARITHM)) return operand->operands()[0];
    return expression;
}

// ================================================================== //
//  Hyperbolic                                                         //
// ================================================================== //

static CASExprPtr simplify_sinh(const CASExprPtr& expression) {
    if (expression->operands()[0]->is_zero()) return ZERO();
    return expression;
}

static CASExprPtr simplify_cosh(const CASExprPtr& expression) {
    if (expression->operands()[0]->is_zero()) return ONE();
    return expression;
}

static CASExprPtr simplify_tanh(const CASExprPtr& expression) {
    if (expression->operands()[0]->is_zero()) return ZERO();
    return expression;
}

// ================================================================== //
//  Inverse trigonometric                                              //
// ================================================================== //

static CASExprPtr simplify_asin(const CASExprPtr& expression) {
    if (expression->operands()[0]->is_zero()) return ZERO();
    return expression;
}

static CASExprPtr simplify_acos(const CASExprPtr& expression) {
    if (expression->operands()[0]->is_one()) return ZERO();
    return expression;
}

static CASExprPtr simplify_atan(const CASExprPtr& expression) {
    if (expression->operands()[0]->is_zero()) return ZERO();
    return expression;
}

// ================================================================== //
//  SQRT / ABS                                                         //
// ================================================================== //

static CASExprPtr simplify_sqrt(const CASExprPtr& expression) {
    auto& operand = expression->operands()[0];
    if (operand->is_zero()) return ZERO();
    if (operand->is_one()) return ONE();
    // sqrt(x^2) → abs(x)
    if (operand->op() == u8(Op::POWER)
        && operand->operands()[1]->op() == u8(Op::INTEGER)
        && operand->operands()[1]->terminal_param() == 2)
        return std::make_shared<CASExpression>(
            u8(Op::ABS),
            std::vector<CASExprPtr>{operand->operands()[0]});
    return expression;
}

static CASExprPtr simplify_abs(const CASExprPtr& expression) {
    auto& operand = expression->operands()[0];
    if (operand->is_zero()) return ZERO();
    if (operand->is_one()) return ONE();
    if (operand->op() == u8(Op::INTEGER))
        return interned_integer(std::abs(operand->terminal_param()));
    if (operand->op() == u8(Op::ABS)) return operand;  // abs(abs(x))
    return expression;
}

// ================================================================== //
//  Square / Cube (for stacks that still contain these opcodes)        //
// ================================================================== //

static CASExprPtr simplify_square(const CASExprPtr& expression) {
    auto& operand = expression->operands()[0];
    if (operand->is_zero()) return ZERO();
    if (operand->is_one()) return ONE();
    if (operand->op() == u8(Op::INTEGER)) {
        int v = operand->terminal_param();
        return interned_integer(v * v);
    }
    if (operand->op() == u8(Op::SQRT)) return operand->operands()[0];
    return expression;
}

static CASExprPtr simplify_cube(const CASExprPtr& expression) {
    auto& operand = expression->operands()[0];
    if (operand->is_zero()) return ZERO();
    if (operand->is_one()) return ONE();
    if (operand->op() == u8(Op::INTEGER)) {
        int v = operand->terminal_param();
        return interned_integer(v * v * v);
    }
    return expression;
}

// ================================================================== //
//  No-op                                                              //
// ================================================================== //

static CASExprPtr no_simplification(const CASExprPtr& expression) {
    return expression;
}

// ================================================================== //
//  Dispatch table                                                     //
// ================================================================== //

const std::unordered_map<uint8_t, SimplifyFn>& simplification_functions() {
    static const std::unordered_map<uint8_t, SimplifyFn> table = {
        {u8(Op::POWER),          simplify_power},
        {u8(Op::MULTIPLICATION), simplify_product},
        {u8(Op::ADDITION),       simplify_sum},
        {u8(Op::DIVISION),       simplify_quotient},
        {u8(Op::SUBTRACTION),    simplify_difference},
        {u8(Op::SIN),            simplify_sin},
        {u8(Op::COS),            simplify_cos},
        {u8(Op::TAN),            simplify_tan},
        {u8(Op::LOGARITHM),      simplify_logarithm},
        {u8(Op::EXPONENTIAL),    simplify_exponential},
        {u8(Op::ABS),            simplify_abs},
        {u8(Op::SQRT),           simplify_sqrt},
        {u8(Op::SAFE_POWER),     simplify_power},
        {u8(Op::SINH),           simplify_sinh},
        {u8(Op::COSH),           simplify_cosh},
        {u8(Op::TANH),           simplify_tanh},
        {u8(Op::ARCSIN),         simplify_asin},
        {u8(Op::ARCCOS),         simplify_acos},
        {u8(Op::ARCTAN),         simplify_atan},
        {u8(Op::SQUARE),         simplify_square},
        {u8(Op::CUBE),           simplify_cube},
    };
    return table;
}

// ================================================================== //
//  automatic_simplify                                                 //
// ================================================================== //

CASExprPtr automatic_simplify(const CASExprPtr& expression) {
    auto op = expression->op();
    if (op == u8(Op::CONSTANT) || op == u8(Op::INTEGER) || op == u8(Op::VARIABLE))
        return expression;

    auto simplified_operands = expression->map(
        [](const CASExprPtr& child) { return automatic_simplify(child); });

    auto& table = simplification_functions();
    auto it = table.find(simplified_operands->op());
    if (it != table.end())
        return it->second(simplified_operands);
    return simplified_operands;
}

}  // namespace cppagraph
