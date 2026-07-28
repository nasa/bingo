/**
 * @file optional_modifications.cpp
 * @brief Optional post-simplification modifications — port of optional_modifications.py.
 */

#include "cppagraph/optional_modifications.h"

#include <unordered_set>
#include <vector>

namespace cppagraph {

static inline uint8_t u8(Op o) { return static_cast<uint8_t>(o); }

static const std::unordered_set<uint8_t> TERMINAL_OPS = {
    u8(Op::CONSTANT), u8(Op::INTEGER), u8(Op::VARIABLE)
};

static constexpr int SOME_BIG_INT = 1'000'000;

// Forward declarations.
static CASExprPtr _insert_subtraction(const CASExprPtr& expression);
static CASExprPtr _insert_division(const CASExprPtr& expression);
static CASExprPtr _insert_square_cube(const CASExprPtr& expression);
static CASExprPtr _replace_integer_powers(const CASExprPtr& expression);
static CASExprPtr _replace_integers_with_constants(const CASExprPtr& expression);

// ------------------------------------------------------------------ //
//  Helpers                                                            //
// ------------------------------------------------------------------ //

static bool _is_inverse(const CASExprPtr& expr) {
    return expr->op() == u8(Op::POWER)
        && expr->operands()[1]->op() == u8(Op::INTEGER)
        && expr->operands()[1]->terminal_param() == -1;
}

// ------------------------------------------------------------------ //
//  a + (-1)*b → a - b                                                 //
// ------------------------------------------------------------------ //

static CASExprPtr _insert_subtraction(const CASExprPtr& expression) {
    uint8_t op = expression->op();
    if (TERMINAL_OPS.count(op)) return expression;

    auto& orig_operands = expression->operands();
    std::vector<CASExprPtr> operands_w_sub;
    operands_w_sub.reserve(orig_operands.size());
    bool changed = false;
    for (auto& child : orig_operands) {
        auto sub = _insert_subtraction(child);
        operands_w_sub.push_back(sub);
        if (sub.get() != child.get()) changed = true;
    }

    if (op != u8(Op::ADDITION)) {
        if (!changed) return expression;
        return std::make_shared<CASExpression>(op, std::move(operands_w_sub));
    }

    auto NEG_ONE = cas_neg_one();
    std::vector<CASExprPtr> additive, subtractive;
    for (auto& operand : operands_w_sub) {
        auto coeff = operand->coefficient();
        if ((coeff.get() == NEG_ONE.get()) || (coeff && *coeff == *NEG_ONE)) {
            auto term = operand->term();
            if (term->operands().size() == 1)
                subtractive.push_back(term->operands()[0]);
            else
                subtractive.push_back(term);
        } else {
            additive.push_back(operand);
        }
    }

    if (subtractive.empty())
        return std::make_shared<CASExpression>(u8(Op::ADDITION), std::move(additive));

    if (additive.empty()) {
        auto sub_sum = std::make_shared<CASExpression>(u8(Op::ADDITION), std::move(subtractive));
        return std::make_shared<CASExpression>(
            u8(Op::MULTIPLICATION),
            std::vector<CASExprPtr>{NEG_ONE, sub_sum});
    }

    CASExprPtr sub_exp;
    if (subtractive.size() == 1) sub_exp = subtractive[0];
    else sub_exp = std::make_shared<CASExpression>(u8(Op::ADDITION), std::move(subtractive));

    CASExprPtr add_exp;
    if (additive.size() == 1) add_exp = additive[0];
    else add_exp = std::make_shared<CASExpression>(u8(Op::ADDITION), std::move(additive));

    return std::make_shared<CASExpression>(
        u8(Op::SUBTRACTION), std::vector<CASExprPtr>{add_exp, sub_exp});
}

// ------------------------------------------------------------------ //
//  a * b^(-1) → a / b                                                 //
// ------------------------------------------------------------------ //

static CASExprPtr _insert_division(const CASExprPtr& expression) {
    uint8_t op = expression->op();
    if (TERMINAL_OPS.count(op)) return expression;

    auto& orig_operands = expression->operands();

    // Standalone inverse: POWER(x, -1) → DIVISION(1, x)
    if (op == u8(Op::POWER) && _is_inverse(expression)) {
        auto base = _insert_division(orig_operands[0]);
        return std::make_shared<CASExpression>(
            u8(Op::DIVISION),
            std::vector<CASExprPtr>{cas_one(), base});
    }

    if (op != u8(Op::MULTIPLICATION)) {
        std::vector<CASExprPtr> new_ops;
        new_ops.reserve(orig_operands.size());
        bool changed = false;
        for (auto& child : orig_operands) {
            auto d = _insert_division(child);
            new_ops.push_back(d);
            if (d.get() != child.get()) changed = true;
        }
        if (!changed) return expression;
        return std::make_shared<CASExpression>(op, std::move(new_ops));
    }

    // Split into numerator / denominator.
    std::vector<CASExprPtr> numerator_ops, denominator_ops;
    for (auto& operand : orig_operands) {
        if (_is_inverse(operand))
            denominator_ops.push_back(_insert_division(operand->operands()[0]));
        else
            numerator_ops.push_back(_insert_division(operand));
    }

    if (denominator_ops.empty()) {
        bool changed = false;
        for (size_t i = 0; i < numerator_ops.size(); ++i)
            if (numerator_ops[i].get() != orig_operands[i].get()) changed = true;
        if (!changed) return expression;
        return std::make_shared<CASExpression>(u8(Op::MULTIPLICATION), std::move(numerator_ops));
    }

    CASExprPtr numerator;
    if (numerator_ops.empty()) numerator = cas_one();
    else if (numerator_ops.size() == 1) numerator = numerator_ops[0];
    else numerator = std::make_shared<CASExpression>(u8(Op::MULTIPLICATION), std::move(numerator_ops));

    CASExprPtr denominator;
    if (denominator_ops.size() == 1) denominator = denominator_ops[0];
    else denominator = std::make_shared<CASExpression>(u8(Op::MULTIPLICATION), std::move(denominator_ops));

    return std::make_shared<CASExpression>(
        u8(Op::DIVISION), std::vector<CASExprPtr>{numerator, denominator});
}

// ------------------------------------------------------------------ //
//  x^2 → SQUARE(x),  x^3 → CUBE(x),  x^(2^-n) → nested SQRT          //
// ------------------------------------------------------------------ //

/**
 * @brief Helper to check if an integer is a power of 2.
 */
static bool is_power_of_two(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

/**
 * @brief Compute log base 2 of a power of 2.
 */
static int log2_int(int n) {
    int result = 0;
    while (n > 1) { n >>= 1; ++result; }
    return result;
}

/**
 * @brief Check if expr represents 2^(-n) and return the depth n.
 *
 * Recognizes two patterns after _insert_division runs:
 * 1. POWER(2, negative_integer) → depth = |negative_integer|
 * 2. DIVISION(1, power_of_two_int) → depth = log2(power_of_two_int)
 *
 * @return The positive depth n if expr == 2^(-n), else 0.
 */
static int _is_power_of_two_inverse(const CASExprPtr& expr) {
    // Pattern 1: POWER(2, -n)
    if (expr->op() == u8(Op::POWER)
        && expr->operands()[0]->op() == u8(Op::INTEGER)
        && expr->operands()[0]->terminal_param() == 2
        && expr->operands()[1]->op() == u8(Op::INTEGER)
        && expr->operands()[1]->terminal_param() < 0) {
        return -expr->operands()[1]->terminal_param();
    }
    // Pattern 2: DIVISION(1, 2^n) - produced by _insert_division
    if (expr->op() == u8(Op::DIVISION)
        && expr->operands()[0]->op() == u8(Op::INTEGER)
        && expr->operands()[0]->terminal_param() == 1
        && expr->operands()[1]->op() == u8(Op::INTEGER)) {
        int denom = expr->operands()[1]->terminal_param();
        if (is_power_of_two(denom)) {
            return log2_int(denom);
        }
    }
    return 0;
}

/**
 * @brief Build nested SQRT calls.
 * depth=1 → SQRT(base), depth=2 → SQRT(SQRT(base)), etc.
 */
static CASExprPtr _build_nested_sqrt(const CASExprPtr& base, int depth) {
    CASExprPtr result = base;
    for (int i = 0; i < depth; ++i) {
        result = std::make_shared<CASExpression>(
            u8(Op::SQRT), std::vector<CASExprPtr>{result});
    }
    return result;
}

static CASExprPtr _make_square(const CASExprPtr& base) {
    if (base->op() == u8(Op::SQRT)) return base->operands()[0];
    return std::make_shared<CASExpression>(
        u8(Op::SQUARE), std::vector<CASExprPtr>{base});
}

static CASExprPtr _insert_square_cube(const CASExprPtr& expression) {
    uint8_t op = expression->op();
    if (TERMINAL_OPS.count(op)) return expression;

    auto& orig_operands = expression->operands();
    std::vector<CASExprPtr> new_ops;
    new_ops.reserve(orig_operands.size());
    bool changed = false;
    for (auto& child : orig_operands) {
        auto sc = _insert_square_cube(child);
        new_ops.push_back(sc);
        if (sc.get() != child.get()) changed = true;
    }

    if (op == u8(Op::POWER)) {
        auto& exponent = new_ops[1];

        // x^2 → SQUARE(x), x^3 → CUBE(x)
        if (exponent->op() == u8(Op::INTEGER)) {
            int exp_val = exponent->terminal_param();
            if (exp_val == 2)
                return _make_square(new_ops[0]);
            if (exp_val == 3)
                return std::make_shared<CASExpression>(u8(Op::CUBE), std::vector<CASExprPtr>{new_ops[0]});
        }

        // x^(2^(-n)) → nested SQRT(x)
        int sqrt_depth = _is_power_of_two_inverse(exponent);
        if (sqrt_depth > 0) {
            return _build_nested_sqrt(new_ops[0], sqrt_depth);
        }

        // x^(m/n) → simplify based on the fraction
        // This handles DIVISION(m, n) produced by _insert_division
        if (exponent->op() == u8(Op::DIVISION)
            && exponent->operands()[0]->op() == u8(Op::INTEGER)
            && exponent->operands()[1]->op() == u8(Op::INTEGER)) {
            int numer = exponent->operands()[0]->terminal_param();
            int denom = exponent->operands()[1]->terminal_param();

            if (numer > 0 && denom > 0) {
                // Check if the exponent reduces to an integer
                if (numer % denom == 0) {
                    int int_exp = numer / denom;
                    if (int_exp == 1) {
                        return new_ops[0];  // x^1 = x
                    } else if (int_exp == 2) {
                        return _make_square(new_ops[0]);
                    } else if (int_exp == 3) {
                        return std::make_shared<CASExpression>(
                            u8(Op::CUBE), std::vector<CASExprPtr>{new_ops[0]});
                    } else {
                        auto int_expr = interned_integer(int_exp);
                        return std::make_shared<CASExpression>(
                            u8(Op::POWER), std::vector<CASExprPtr>{new_ops[0], int_expr});
                    }
                }

                // Check if denominator is a power of 2 (fractional sqrt power)
                if (denom > 1 && is_power_of_two(denom)) {
                    int depth = log2_int(denom);
                    auto nested_sqrt = _build_nested_sqrt(new_ops[0], depth);
                    if (numer == 1) {
                        return nested_sqrt;
                    } else if (numer == 2) {
                        return _make_square(nested_sqrt);
                    } else if (numer == 3) {
                        return std::make_shared<CASExpression>(
                            u8(Op::CUBE), std::vector<CASExprPtr>{nested_sqrt});
                    } else {
                        auto int_expr = interned_integer(numer);
                        return std::make_shared<CASExpression>(
                            u8(Op::POWER), std::vector<CASExprPtr>{nested_sqrt, int_expr});
                    }
                }
            }
        }

        // x^(m * 2^(-n)) → integer_power(nested_sqrt(x))
        if (exponent->op() == u8(Op::MULTIPLICATION)
            && exponent->operands().size() == 2) {
            int int_part = 0;
            int sqrt_part_depth = 0;
            for (auto& sub_op : exponent->operands()) {
                if (sub_op->op() == u8(Op::INTEGER)) {
                    int_part = sub_op->terminal_param();
                } else {
                    sqrt_part_depth = _is_power_of_two_inverse(sub_op);
                }
            }
            if (int_part > 0 && sqrt_part_depth > 0) {
                auto nested_sqrt = _build_nested_sqrt(new_ops[0], sqrt_part_depth);
                if (int_part == 1) {
                    return nested_sqrt;
                } else if (int_part == 2) {
                    return _make_square(nested_sqrt);
                } else if (int_part == 3) {
                    return std::make_shared<CASExpression>(
                        u8(Op::CUBE), std::vector<CASExprPtr>{nested_sqrt});
                } else {
                    // For larger integer powers, leave as POWER
                    auto int_expr = interned_integer(int_part);
                    return std::make_shared<CASExpression>(
                        u8(Op::POWER), std::vector<CASExprPtr>{nested_sqrt, int_expr});
                }
            }
        }
    }

    if (!changed) return expression;
    return std::make_shared<CASExpression>(op, std::move(new_ops));
}

// ------------------------------------------------------------------ //
//  a^n → expanded multiplication                                      //
// ------------------------------------------------------------------ //

static CASExprPtr _replace_integer_powers(const CASExprPtr& expression) {
    uint8_t op = expression->op();
    if (TERMINAL_OPS.count(op)) return expression;

    auto& orig_operands = expression->operands();
    std::vector<CASExprPtr> replaced;
    replaced.reserve(orig_operands.size());
    bool changed = false;
    for (auto& child : orig_operands) {
        auto r = _replace_integer_powers(child);
        replaced.push_back(r);
        if (r.get() != child.get()) changed = true;
    }

    if (op != u8(Op::POWER)
        || replaced[1]->op() != u8(Op::INTEGER)
        || replaced[1]->terminal_param() <= 0) {
        if (!changed) return expression;
        return std::make_shared<CASExpression>(op, std::move(replaced));
    }

    int power = replaced[1]->terminal_param();
    auto& base = replaced[0];
    std::vector<CASExprPtr> factors(power, base);
    return std::make_shared<CASExpression>(u8(Op::MULTIPLICATION), std::move(factors));
}

// ------------------------------------------------------------------ //
//  integer coefficients → constants                                   //
// ------------------------------------------------------------------ //

static CASExprPtr _replace_integers_with_constants(const CASExprPtr& expression) {
    uint8_t op = expression->op();
    if (op == u8(Op::CONSTANT) || op == u8(Op::VARIABLE)) return expression;
    if (op == u8(Op::INTEGER))
        return std::make_shared<CASExpression>(
            u8(Op::CONSTANT), SOME_BIG_INT + expression->terminal_param());

    auto& orig_operands = expression->operands();
    std::vector<CASExprPtr> replaced;
    replaced.reserve(orig_operands.size());
    bool changed = false;
    for (auto& child : orig_operands) {
        auto r = _replace_integers_with_constants(child);
        replaced.push_back(r);
        if (r.get() != child.get()) changed = true;
    }
    if (!changed) return expression;
    return std::make_shared<CASExpression>(op, std::move(replaced));
}

// ------------------------------------------------------------------ //
//  optional_modifications (public)                                    //
// ------------------------------------------------------------------ //

CASExprPtr optional_modifications(
    const CASExprPtr& expression,
    const OptionalModFlags& flags)
{
    CASExprPtr result = expression;
    if (flags.insert_subtraction)
        result = _insert_subtraction(result);
    if (flags.insert_division)
        result = _insert_division(result);
    if (flags.insert_square_cube)
        result = _insert_square_cube(result);
    if (flags.replace_integer_powers)
        result = _replace_integer_powers(result);
    if (flags.replace_integers_with_constants)
        result = _replace_integers_with_constants(result);
    return result;
}

}  // namespace cppagraph
