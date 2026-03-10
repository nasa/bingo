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
//  x^2 → SQUARE(x),  x^3 → CUBE(x)                                   //
// ------------------------------------------------------------------ //

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

    if (op == u8(Op::POWER)
        && new_ops[1]->op() == u8(Op::INTEGER)) {
        int exp_val = new_ops[1]->terminal_param();
        if (exp_val == 2)
            return std::make_shared<CASExpression>(u8(Op::SQUARE), std::vector<CASExprPtr>{new_ops[0]});
        if (exp_val == 3)
            return std::make_shared<CASExpression>(u8(Op::CUBE), std::vector<CASExprPtr>{new_ops[0]});
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
