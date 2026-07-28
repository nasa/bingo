/**
 * @file test_cas_simplify.cpp
 * @brief Google Test suite for the full CAS simplification pipeline.
 *
 * Tests cover:
 *   - CASExpression construction, equality, hashing, ordering
 *   - Interpreter: build_cas_expression, build_agraph_stack
 *   - Automatic simplification rules
 *   - Constant folding
 *   - Optional modifications
 *   - Full pipeline (cas_simplify)
 */

#include <gtest/gtest.h>

#include "cppagraph/cas_expression.h"
#include "cppagraph/automatic_simplification.h"
#include "cppagraph/interpreter.h"
#include "cppagraph/constant_folding.h"
#include "cppagraph/optional_modifications.h"
#include "cppagraph/cas_simplify.h"

using namespace cppagraph;

static inline uint8_t u8(Op o) { return static_cast<uint8_t>(o); }

// Helper to build a stack from a vector of rows.
static StackMatrix make_stack(
    const std::vector<std::array<uint8_t, 3>>& rows)
{
    StackMatrix s(rows.size(), 3);
    for (size_t i = 0; i < rows.size(); ++i)
        for (int j = 0; j < 3; ++j)
            s(i, j) = rows[i][j];
    return s;
}

// ================================================================== //
//  CASExpression tests                                                //
// ================================================================== //

class TestCASExpression : public ::testing::Test {};

TEST_F(TestCASExpression, TerminalConstruction) {
    auto x = interned_variable(0);
    EXPECT_EQ(x->op(), u8(Op::VARIABLE));
    EXPECT_EQ(x->terminal_param(), 0);
}

TEST_F(TestCASExpression, IntegerSingletons) {
    auto z1 = cas_zero();
    auto z2 = cas_zero();
    EXPECT_EQ(z1.get(), z2.get());  // Same object
    EXPECT_TRUE(z1->is_zero());
    EXPECT_FALSE(z1->is_one());

    auto o1 = cas_one();
    EXPECT_TRUE(o1->is_one());
    EXPECT_FALSE(o1->is_zero());

    auto two = cas_two();
    EXPECT_EQ(two->terminal_param(), 2);
}

TEST_F(TestCASExpression, InternedIntegerDedup) {
    auto a = interned_integer(0);
    auto b = cas_zero();
    EXPECT_EQ(a.get(), b.get());

    auto c = interned_integer(42);
    auto d = interned_integer(42);
    EXPECT_NE(c.get(), d.get());  // Not interned
    EXPECT_EQ(*c, *d);            // But equal
}

TEST_F(TestCASExpression, Equality) {
    auto x0 = interned_variable(0);
    auto x1 = interned_variable(1);
    EXPECT_EQ(*x0, *x0);
    EXPECT_NE(*x0, *x1);

    auto add1 = std::make_shared<CASExpression>(
        u8(Op::ADDITION), std::vector<CASExprPtr>{x0, x1});
    auto add2 = std::make_shared<CASExpression>(
        u8(Op::ADDITION), std::vector<CASExprPtr>{x0, x1});
    EXPECT_EQ(*add1, *add2);
}

TEST_F(TestCASExpression, Hash) {
    auto x0 = interned_variable(0);
    auto x1 = interned_variable(1);
    auto add1 = std::make_shared<CASExpression>(
        u8(Op::ADDITION), std::vector<CASExprPtr>{x0, x1});
    auto add2 = std::make_shared<CASExpression>(
        u8(Op::ADDITION), std::vector<CASExprPtr>{x0, x1});
    EXPECT_EQ(add1->hash(), add2->hash());
}

TEST_F(TestCASExpression, Ordering) {
    auto x = interned_variable(0);
    auto c = std::make_shared<CASExpression>(u8(Op::CONSTANT), 0);
    auto i = interned_integer(5);
    // Constants come first (is_constant_valued)
    EXPECT_TRUE(*i < *x);
    EXPECT_TRUE(*c < *x);
}

TEST_F(TestCASExpression, BaseExponent) {
    auto x = interned_variable(0);
    auto two = interned_integer(2);
    auto pow = std::make_shared<CASExpression>(
        u8(Op::POWER), std::vector<CASExprPtr>{x, two});
    EXPECT_EQ(*pow->base(), *x);
    EXPECT_EQ(*pow->exponent(), *two);

    // Non-power: base=self, exponent=1
    EXPECT_EQ(*x->base(), *x);
    EXPECT_EQ(*x->exponent(), *cas_one());
}

TEST_F(TestCASExpression, TermCoefficient) {
    auto x = interned_variable(0);
    auto three = interned_integer(3);
    auto prod = std::make_shared<CASExpression>(
        u8(Op::MULTIPLICATION), std::vector<CASExprPtr>{three, x});
    EXPECT_EQ(*prod->coefficient(), *three);
    EXPECT_EQ(prod->term()->operands().size(), 1u);
    EXPECT_EQ(*prod->term()->operands()[0], *x);
}

TEST_F(TestCASExpression, SameTerm) {
    auto x = interned_variable(0);
    auto two = interned_integer(2);
    auto three = interned_integer(3);
    auto prod1 = std::make_shared<CASExpression>(
        u8(Op::MULTIPLICATION), std::vector<CASExprPtr>{two, x});
    auto prod2 = std::make_shared<CASExpression>(
        u8(Op::MULTIPLICATION), std::vector<CASExprPtr>{three, x});
    EXPECT_TRUE(prod1->same_term(*prod2));
    EXPECT_TRUE(x->same_term(*x));
}

TEST_F(TestCASExpression, IsConstantValued) {
    auto x = interned_variable(0);
    auto c = std::make_shared<CASExpression>(u8(Op::CONSTANT), 0);
    auto i = interned_integer(5);
    EXPECT_TRUE(i->is_constant_valued());
    EXPECT_TRUE(c->is_constant_valued());
    EXPECT_FALSE(x->is_constant_valued());

    auto sum = std::make_shared<CASExpression>(
        u8(Op::ADDITION), std::vector<CASExprPtr>{i, c});
    EXPECT_TRUE(sum->is_constant_valued());
}

TEST_F(TestCASExpression, DependsOn) {
    auto x = interned_variable(0);
    auto c = std::make_shared<CASExpression>(u8(Op::CONSTANT), 3);
    auto i = interned_integer(5);
    auto sum = std::make_shared<CASExpression>(
        u8(Op::ADDITION), std::vector<CASExprPtr>{x, c, i});

    auto& deps = sum->depends_on();
    EXPECT_TRUE(deps.count(std::string("x")));
    EXPECT_TRUE(deps.count(3));
    EXPECT_TRUE(deps.count(std::string("i")));
}

TEST_F(TestCASExpression, CopyAndMap) {
    auto x = interned_variable(0);
    auto y = interned_variable(1);
    auto sum = std::make_shared<CASExpression>(
        u8(Op::ADDITION), std::vector<CASExprPtr>{x, y});

    auto cp = sum->copy();
    EXPECT_EQ(*cp, *sum);
    EXPECT_NE(cp.get(), sum.get());

    // Map: replace all variables with constant 0
    auto mapped = sum->map([](const CASExprPtr& child) -> CASExprPtr {
        if (child->op() == u8(Op::VARIABLE))
            return std::make_shared<CASExpression>(u8(Op::CONSTANT), 0);
        return child;
    });
    EXPECT_EQ(mapped->operands()[0]->op(), u8(Op::CONSTANT));
    EXPECT_EQ(mapped->operands()[1]->op(), u8(Op::CONSTANT));
}

// ================================================================== //
//  Automatic simplification tests                                     //
// ================================================================== //

class TestAutomaticSimplify : public ::testing::Test {};

TEST_F(TestAutomaticSimplify, PowerOfOne) {
    auto x = interned_variable(0);
    auto one = cas_one();
    auto pow = std::make_shared<CASExpression>(
        u8(Op::POWER), std::vector<CASExprPtr>{x, one});
    auto result = simplify_power(pow);
    EXPECT_EQ(*result, *x);
}

TEST_F(TestAutomaticSimplify, PowerOfZeroExp) {
    auto x = interned_variable(0);
    auto zero = cas_zero();
    auto pow = std::make_shared<CASExpression>(
        u8(Op::POWER), std::vector<CASExprPtr>{x, zero});
    auto result = simplify_power(pow);
    EXPECT_TRUE(result->is_one());
}

TEST_F(TestAutomaticSimplify, IntegerPower) {
    auto three = interned_integer(3);
    auto two = interned_integer(2);
    auto pow = std::make_shared<CASExpression>(
        u8(Op::POWER), std::vector<CASExprPtr>{three, two});
    auto result = simplify_power(pow);
    EXPECT_EQ(result->op(), u8(Op::INTEGER));
    EXPECT_EQ(result->terminal_param(), 9);
}

TEST_F(TestAutomaticSimplify, OneBaseReturnOne) {
    auto one = cas_one();
    auto x = interned_variable(0);
    auto pow = std::make_shared<CASExpression>(
        u8(Op::POWER), std::vector<CASExprPtr>{one, x});
    auto result = simplify_power(pow);
    EXPECT_TRUE(result->is_one());
}

TEST_F(TestAutomaticSimplify, ProductWithZero) {
    auto x = interned_variable(0);
    auto zero = cas_zero();
    auto prod = std::make_shared<CASExpression>(
        u8(Op::MULTIPLICATION), std::vector<CASExprPtr>{x, zero});
    auto result = simplify_product(prod);
    EXPECT_TRUE(result->is_zero());
}

TEST_F(TestAutomaticSimplify, ProductWithOne) {
    auto x = interned_variable(0);
    auto one = cas_one();
    auto prod = std::make_shared<CASExpression>(
        u8(Op::MULTIPLICATION), std::vector<CASExprPtr>{one, x});
    auto result = simplify_product(prod);
    EXPECT_EQ(*result, *x);
}

TEST_F(TestAutomaticSimplify, ProductOfIntegers) {
    auto three = interned_integer(3);
    auto four = interned_integer(4);
    auto prod = std::make_shared<CASExpression>(
        u8(Op::MULTIPLICATION), std::vector<CASExprPtr>{three, four});
    auto result = simplify_product(prod);
    EXPECT_EQ(result->terminal_param(), 12);
}

TEST_F(TestAutomaticSimplify, ProductCombinesBases) {
    // x * x = x^2
    auto x = interned_variable(0);
    auto prod = std::make_shared<CASExpression>(
        u8(Op::MULTIPLICATION), std::vector<CASExprPtr>{x, x});
    auto result = simplify_product(prod);
    EXPECT_EQ(result->op(), u8(Op::POWER));
    EXPECT_EQ(*result->operands()[0], *x);
    EXPECT_EQ(result->operands()[1]->terminal_param(), 2);
}

TEST_F(TestAutomaticSimplify, SumWithZero) {
    auto x = interned_variable(0);
    auto zero = cas_zero();
    auto sum = std::make_shared<CASExpression>(
        u8(Op::ADDITION), std::vector<CASExprPtr>{x, zero});
    auto result = simplify_sum(sum);
    EXPECT_EQ(*result, *x);
}

TEST_F(TestAutomaticSimplify, SumOfIntegers) {
    auto three = interned_integer(3);
    auto four = interned_integer(4);
    auto sum = std::make_shared<CASExpression>(
        u8(Op::ADDITION), std::vector<CASExprPtr>{three, four});
    auto result = simplify_sum(sum);
    EXPECT_EQ(result->terminal_param(), 7);
}

TEST_F(TestAutomaticSimplify, SumCombinesSameTerms) {
    // x + x = 2*x
    auto x = interned_variable(0);
    auto sum = std::make_shared<CASExpression>(
        u8(Op::ADDITION), std::vector<CASExprPtr>{x, x});
    auto result = simplify_sum(sum);
    EXPECT_EQ(result->op(), u8(Op::MULTIPLICATION));
}

TEST_F(TestAutomaticSimplify, DifferenceToSum) {
    // x - y → x + (-1)*y
    auto x = interned_variable(0);
    auto y = interned_variable(1);
    auto diff = std::make_shared<CASExpression>(
        u8(Op::SUBTRACTION), std::vector<CASExprPtr>{x, y});
    auto result = simplify_difference(diff);
    EXPECT_EQ(result->op(), u8(Op::ADDITION));
}

TEST_F(TestAutomaticSimplify, DivisionToProduct) {
    // x / y → x * y^(-1)
    auto x = interned_variable(0);
    auto y = interned_variable(1);
    auto div = std::make_shared<CASExpression>(
        u8(Op::DIVISION), std::vector<CASExprPtr>{x, y});
    auto result = simplify_quotient(div);
    EXPECT_EQ(result->op(), u8(Op::MULTIPLICATION));
}

TEST_F(TestAutomaticSimplify, SinOfZero) {
    auto zero = cas_zero();
    auto sinexpr = std::make_shared<CASExpression>(
        u8(Op::SIN), std::vector<CASExprPtr>{zero});
    auto& table = simplification_functions();
    auto result = table.at(u8(Op::SIN))(sinexpr);
    EXPECT_TRUE(result->is_zero());
}

TEST_F(TestAutomaticSimplify, CosOfZero) {
    auto zero = cas_zero();
    auto cosexpr = std::make_shared<CASExpression>(
        u8(Op::COS), std::vector<CASExprPtr>{zero});
    auto& table = simplification_functions();
    auto result = table.at(u8(Op::COS))(cosexpr);
    EXPECT_TRUE(result->is_one());
}

TEST_F(TestAutomaticSimplify, ExpOfZero) {
    auto zero = cas_zero();
    auto expexpr = std::make_shared<CASExpression>(
        u8(Op::EXPONENTIAL), std::vector<CASExprPtr>{zero});
    auto& table = simplification_functions();
    auto result = table.at(u8(Op::EXPONENTIAL))(expexpr);
    EXPECT_TRUE(result->is_one());
}

TEST_F(TestAutomaticSimplify, LogOfOne) {
    auto one = cas_one();
    auto logexpr = std::make_shared<CASExpression>(
        u8(Op::LOGARITHM), std::vector<CASExprPtr>{one});
    auto& table = simplification_functions();
    auto result = table.at(u8(Op::LOGARITHM))(logexpr);
    EXPECT_TRUE(result->is_zero());
}

TEST_F(TestAutomaticSimplify, SinOfArcsin) {
    auto x = interned_variable(0);
    auto arcsin = std::make_shared<CASExpression>(
        u8(Op::ARCSIN), std::vector<CASExprPtr>{x});
    auto sinexpr = std::make_shared<CASExpression>(
        u8(Op::SIN), std::vector<CASExprPtr>{arcsin});
    auto& table = simplification_functions();
    auto result = table.at(u8(Op::SIN))(sinexpr);
    EXPECT_EQ(*result, *x);
}

TEST_F(TestAutomaticSimplify, ExpOfLog) {
    auto x = interned_variable(0);
    auto logexpr = std::make_shared<CASExpression>(
        u8(Op::LOGARITHM), std::vector<CASExprPtr>{x});
    auto expexpr = std::make_shared<CASExpression>(
        u8(Op::EXPONENTIAL), std::vector<CASExprPtr>{logexpr});
    auto& table = simplification_functions();
    auto result = table.at(u8(Op::EXPONENTIAL))(expexpr);
    EXPECT_EQ(*result, *x);
}

TEST_F(TestAutomaticSimplify, SqrtOfZero) {
    auto zero = cas_zero();
    auto sqrtexpr = std::make_shared<CASExpression>(
        u8(Op::SQRT), std::vector<CASExprPtr>{zero});
    auto& table = simplification_functions();
    auto result = table.at(u8(Op::SQRT))(sqrtexpr);
    EXPECT_TRUE(result->is_zero());
}

TEST_F(TestAutomaticSimplify, SqrtOfXSquared) {
    // sqrt(x^2) → abs(x)
    auto x = interned_variable(0);
    auto two = interned_integer(2);
    auto pow = std::make_shared<CASExpression>(
        u8(Op::POWER), std::vector<CASExprPtr>{x, two});
    auto sqrtexpr = std::make_shared<CASExpression>(
        u8(Op::SQRT), std::vector<CASExprPtr>{pow});
    auto& table = simplification_functions();
    auto result = table.at(u8(Op::SQRT))(sqrtexpr);
    EXPECT_EQ(result->op(), u8(Op::ABS));
    EXPECT_EQ(*result->operands()[0], *x);
}

TEST_F(TestAutomaticSimplify, AbsOfAbs) {
    auto x = interned_variable(0);
    auto abs1 = std::make_shared<CASExpression>(
        u8(Op::ABS), std::vector<CASExprPtr>{x});
    auto abs2 = std::make_shared<CASExpression>(
        u8(Op::ABS), std::vector<CASExprPtr>{abs1});
    auto& table = simplification_functions();
    auto result = table.at(u8(Op::ABS))(abs2);
    // abs(abs(x)) → abs(x)
    EXPECT_EQ(result->op(), u8(Op::ABS));
    EXPECT_EQ(*result->operands()[0], *x);
}

TEST_F(TestAutomaticSimplify, SelfSubtraction) {
    // x - x = 0
    auto x = interned_variable(0);
    auto diff = std::make_shared<CASExpression>(
        u8(Op::SUBTRACTION), std::vector<CASExprPtr>{x, x});
    auto result = automatic_simplify(diff);
    EXPECT_TRUE(result->is_zero());
}

TEST_F(TestAutomaticSimplify, SelfDivision) {
    // x / x = 1
    auto x = interned_variable(0);
    auto div = std::make_shared<CASExpression>(
        u8(Op::DIVISION), std::vector<CASExprPtr>{x, x});
    auto result = automatic_simplify(div);
    EXPECT_TRUE(result->is_one());
}

TEST_F(TestAutomaticSimplify, NestedPowers) {
    // (x^2)^3 = x^6
    auto x = interned_variable(0);
    auto two = interned_integer(2);
    auto three = interned_integer(3);
    auto inner = std::make_shared<CASExpression>(
        u8(Op::POWER), std::vector<CASExprPtr>{x, two});
    auto outer = std::make_shared<CASExpression>(
        u8(Op::POWER), std::vector<CASExprPtr>{inner, three});
    auto result = automatic_simplify(outer);
    EXPECT_EQ(result->op(), u8(Op::POWER));
    EXPECT_EQ(*result->operands()[0], *x);
    EXPECT_EQ(result->operands()[1]->terminal_param(), 6);
}

// ================================================================== //
//  Interpreter tests                                                  //
// ================================================================== //

class TestInterpreter : public ::testing::Test {};

TEST_F(TestInterpreter, BuildCASExpressionSimple) {
    // Stack: X_0 → row 0, result is X_0
    auto stack = make_stack({{u8(Op::VARIABLE), 0, 0}});
    auto expr = build_cas_expression(stack, {}, {});
    EXPECT_EQ(expr->op(), u8(Op::VARIABLE));
    EXPECT_EQ(expr->terminal_param(), 0);
}

TEST_F(TestInterpreter, BuildCASExpressionAddition) {
    // Stack: X_0, X_1, X_0 + X_1
    auto stack = make_stack({
        {u8(Op::VARIABLE), 0, 0},
        {u8(Op::VARIABLE), 1, 1},
        {u8(Op::ADDITION), 0, 1}
    });
    auto expr = build_cas_expression(stack, {}, {});
    EXPECT_EQ(expr->op(), u8(Op::ADDITION));
    EXPECT_EQ(expr->operands().size(), 2u);
}

TEST_F(TestInterpreter, BuildCASExpressionSquareNormalized) {
    // SQUARE(X_0) → POWER(X_0, 2)
    auto stack = make_stack({
        {u8(Op::VARIABLE), 0, 0},
        {u8(Op::SQUARE), 0, 0}
    });
    auto expr = build_cas_expression(stack, {}, {});
    EXPECT_EQ(expr->op(), u8(Op::POWER));
    EXPECT_EQ(expr->operands()[1]->terminal_param(), 2);
}

TEST_F(TestInterpreter, BuildSimplifiedXMinusX) {
    // X_0 - X_0 = 0
    auto stack = make_stack({
        {u8(Op::VARIABLE), 0, 0},
        {u8(Op::SUBTRACTION), 0, 0}
    });
    auto expr = build_simplified_cas_expression(stack, {}, {});
    EXPECT_TRUE(expr->is_zero());
}

TEST_F(TestInterpreter, BuildAgraphStackRoundtrip) {
    // Build a tree, convert to stack, build tree again, compare.
    auto x = interned_variable(0);
    auto y = interned_variable(1);
    auto sum = std::make_shared<CASExpression>(
        u8(Op::ADDITION), std::vector<CASExprPtr>{x, y});

    auto result = build_agraph_stack(sum, {});
    EXPECT_EQ(result.stack.rows(), 3);  // 2 variables + 1 addition

    // Rebuild and verify
    auto rebuilt = build_cas_expression(result.stack, result.constants, result.integers);
    EXPECT_EQ(rebuilt->op(), u8(Op::ADDITION));
}

TEST_F(TestInterpreter, BuildAgraphStackWithConstants) {
    auto c0 = std::make_shared<CASExpression>(u8(Op::CONSTANT), 0);
    auto c1 = std::make_shared<CASExpression>(u8(Op::CONSTANT), 1);
    auto sum = std::make_shared<CASExpression>(
        u8(Op::ADDITION), std::vector<CASExprPtr>{c0, c1});

    std::vector<double> orig_consts = {3.14, 2.71};
    auto result = build_agraph_stack(sum, orig_consts);
    EXPECT_EQ(result.constants.size(), 2u);
    EXPECT_DOUBLE_EQ(result.constants[0], 3.14);
    EXPECT_DOUBLE_EQ(result.constants[1], 2.71);
}

TEST_F(TestInterpreter, IntegerDedup) {
    auto i1 = interned_integer(5);
    auto i2 = interned_integer(5);
    auto sum = std::make_shared<CASExpression>(
        u8(Op::ADDITION), std::vector<CASExprPtr>{i1, i2});
    auto result = build_agraph_stack(sum, {});
    // Only one INTEGER row since values are deduped.
    int int_count = 0;
    for (int r = 0; r < result.stack.rows(); ++r)
        if (result.stack(r, 0) == u8(Op::INTEGER)) ++int_count;
    EXPECT_EQ(int_count, 1);
}

// ================================================================== //
//  Constant folding tests                                             //
// ================================================================== //

class TestConstantFolding : public ::testing::Test {};

TEST_F(TestConstantFolding, NoFoldingNeeded) {
    auto x = interned_variable(0);
    auto result = fold_constants(x);
    EXPECT_EQ(*result, *x);
}

TEST_F(TestConstantFolding, SimpleFolding) {
    // C_0 + C_1 + X_0 → C_fold + X_0 with one constant
    auto c0 = std::make_shared<CASExpression>(u8(Op::CONSTANT), 0);
    auto c1 = std::make_shared<CASExpression>(u8(Op::CONSTANT), 1);
    auto x = interned_variable(0);
    auto inner = std::make_shared<CASExpression>(
        u8(Op::ADDITION), std::vector<CASExprPtr>{c0, c1});
    auto sum = std::make_shared<CASExpression>(
        u8(Op::ADDITION), std::vector<CASExprPtr>{inner, x});
    auto result = fold_constants(sum);
    // Result should still have an addition with x
    EXPECT_EQ(result->op(), u8(Op::ADDITION));
}

TEST_F(TestConstantFolding, EightConstantsFoldSharedFactors) {
    auto c0 = std::make_shared<CASExpression>(u8(Op::CONSTANT), 0);
    std::vector<CASExprPtr> terms;
    for (int index = 1; index < 8; ++index) {
        auto factor = std::make_shared<CASExpression>(
            u8(Op::MULTIPLICATION),
            std::vector<CASExprPtr>{
                c0,
                std::make_shared<CASExpression>(u8(Op::CONSTANT), index),
            });
        terms.push_back(std::make_shared<CASExpression>(
            u8(Op::MULTIPLICATION),
            std::vector<CASExprPtr>{factor, interned_variable(index % 4)}));
    }

    auto result = fold_constants(std::make_shared<CASExpression>(
        u8(Op::ADDITION), std::move(terms)));
    auto stack_result = build_agraph_stack(result, {});

    EXPECT_EQ(stack_result.constants.size(), 7u);
}

// ================================================================== //
//  Optional modifications tests                                       //
// ================================================================== //

class TestOptionalModifications : public ::testing::Test {};

TEST_F(TestOptionalModifications, InsertSubtraction) {
    // x + (-1)*y → x - y
    auto x = interned_variable(0);
    auto y = interned_variable(1);
    auto neg_y = std::make_shared<CASExpression>(
        u8(Op::MULTIPLICATION),
        std::vector<CASExprPtr>{cas_neg_one(), y});
    auto sum = std::make_shared<CASExpression>(
        u8(Op::ADDITION), std::vector<CASExprPtr>{x, neg_y});

    OptionalModFlags flags;
    auto result = optional_modifications(sum, flags);
    EXPECT_EQ(result->op(), u8(Op::SUBTRACTION));
}

TEST_F(TestOptionalModifications, InsertDivision) {
    // x * y^(-1) → x / y
    auto x = interned_variable(0);
    auto y = interned_variable(1);
    auto y_inv = std::make_shared<CASExpression>(
        u8(Op::POWER),
        std::vector<CASExprPtr>{y, cas_neg_one()});
    auto prod = std::make_shared<CASExpression>(
        u8(Op::MULTIPLICATION),
        std::vector<CASExprPtr>{x, y_inv});

    OptionalModFlags flags;
    auto result = optional_modifications(prod, flags);
    EXPECT_EQ(result->op(), u8(Op::DIVISION));
}

TEST_F(TestOptionalModifications, InsertSquareCube) {
    // x^2 → SQUARE(x), x^3 → CUBE(x)
    auto x = interned_variable(0);
    auto two = interned_integer(2);
    auto three = interned_integer(3);
    auto pow2 = std::make_shared<CASExpression>(
        u8(Op::POWER), std::vector<CASExprPtr>{x, two});
    auto pow3 = std::make_shared<CASExpression>(
        u8(Op::POWER), std::vector<CASExprPtr>{x, three});

    OptionalModFlags flags;
    auto r2 = optional_modifications(pow2, flags);
    EXPECT_EQ(r2->op(), u8(Op::SQUARE));
    auto r3 = optional_modifications(pow3, flags);
    EXPECT_EQ(r3->op(), u8(Op::CUBE));
}

TEST_F(TestOptionalModifications, PowerHalfToSqrt) {
    // x^(2^(-1)) → SQRT(x)
    auto x = interned_variable(0);
    auto two = interned_integer(2);
    auto neg_one = interned_integer(-1);
    auto half_exp = std::make_shared<CASExpression>(
        u8(Op::POWER), std::vector<CASExprPtr>{two, neg_one});
    auto pow_half = std::make_shared<CASExpression>(
        u8(Op::POWER), std::vector<CASExprPtr>{x, half_exp});

    OptionalModFlags flags;
    auto result = optional_modifications(pow_half, flags);
    EXPECT_EQ(result->op(), u8(Op::SQRT));
    EXPECT_EQ(*result->operands()[0], *x);
}

TEST_F(TestOptionalModifications, PowerQuarterToNestedSqrt) {
    // x^(2^(-2)) → SQRT(SQRT(x))
    auto x = interned_variable(0);
    auto two = interned_integer(2);
    auto neg_two = interned_integer(-2);
    auto quarter_exp = std::make_shared<CASExpression>(
        u8(Op::POWER), std::vector<CASExprPtr>{two, neg_two});
    auto pow_quarter = std::make_shared<CASExpression>(
        u8(Op::POWER), std::vector<CASExprPtr>{x, quarter_exp});

    OptionalModFlags flags;
    auto result = optional_modifications(pow_quarter, flags);
    EXPECT_EQ(result->op(), u8(Op::SQRT));
    EXPECT_EQ(result->operands()[0]->op(), u8(Op::SQRT));
    EXPECT_EQ(*result->operands()[0]->operands()[0], *x);
}

TEST_F(TestOptionalModifications, PowerThreeHalvesToCubeSqrt) {
    // x^(3 * 2^(-1)) → CUBE(SQRT(x))
    auto x = interned_variable(0);
    auto three = interned_integer(3);
    auto two = interned_integer(2);
    auto neg_one = interned_integer(-1);
    auto half_exp = std::make_shared<CASExpression>(
        u8(Op::POWER), std::vector<CASExprPtr>{two, neg_one});
    auto three_halves = std::make_shared<CASExpression>(
        u8(Op::MULTIPLICATION), std::vector<CASExprPtr>{three, half_exp});
    auto pow_three_halves = std::make_shared<CASExpression>(
        u8(Op::POWER), std::vector<CASExprPtr>{x, three_halves});

    OptionalModFlags flags;
    auto result = optional_modifications(pow_three_halves, flags);
    EXPECT_EQ(result->op(), u8(Op::CUBE));
    EXPECT_EQ(result->operands()[0]->op(), u8(Op::SQRT));
    EXPECT_EQ(*result->operands()[0]->operands()[0], *x);
}

TEST_F(TestOptionalModifications, StandaloneDivision) {
    // y^(-1) → 1/y
    auto y = interned_variable(1);
    auto inv = std::make_shared<CASExpression>(
        u8(Op::POWER),
        std::vector<CASExprPtr>{y, cas_neg_one()});

    OptionalModFlags flags;
    auto result = optional_modifications(inv, flags);
    EXPECT_EQ(result->op(), u8(Op::DIVISION));
    EXPECT_TRUE(result->operands()[0]->is_one());
}

TEST_F(TestOptionalModifications, DisabledFlags) {
    auto x = interned_variable(0);
    auto two = interned_integer(2);
    auto pow2 = std::make_shared<CASExpression>(
        u8(Op::POWER), std::vector<CASExprPtr>{x, two});

    OptionalModFlags flags;
    flags.insert_square_cube = false;
    auto result = optional_modifications(pow2, flags);
    EXPECT_EQ(result->op(), u8(Op::POWER));  // Not converted
}

// ================================================================== //
//  Full pipeline tests                                                //
// ================================================================== //

class TestCASSimplifyPipeline : public ::testing::Test {};

TEST_F(TestCASSimplifyPipeline, IdentityXPlusZero) {
    // (0)(X_0)(X_0 + int(0)) → simplifies to X_0
    auto stack = make_stack({
        {u8(Op::VARIABLE), 0, 0},
        {u8(Op::INTEGER), 0, 0},
        {u8(Op::ADDITION), 0, 1}
    });
    std::vector<double> constants = {};
    std::vector<int> integers = {0};

    auto result = cas_simplify(stack, constants, integers);
    // After simplification, should just be X_0.
    EXPECT_EQ(result.stack.rows(), 1);
    EXPECT_EQ(result.stack(0, 0), u8(Op::VARIABLE));
}

TEST_F(TestCASSimplifyPipeline, XMinusX) {
    // X_0 - X_0 → 0
    auto stack = make_stack({
        {u8(Op::VARIABLE), 0, 0},
        {u8(Op::SUBTRACTION), 0, 0}
    });
    auto result = cas_simplify(stack, {}, {});
    EXPECT_EQ(result.stack.rows(), 1);
    EXPECT_EQ(result.stack(0, 0), u8(Op::INTEGER));
    EXPECT_EQ(result.integers.size(), 1u);
    EXPECT_EQ(result.integers[0], 0);
}

TEST_F(TestCASSimplifyPipeline, XDivX) {
    // X_0 / X_0 → 1
    auto stack = make_stack({
        {u8(Op::VARIABLE), 0, 0},
        {u8(Op::DIVISION), 0, 0}
    });
    auto result = cas_simplify(stack, {}, {});
    EXPECT_EQ(result.stack.rows(), 1);
    EXPECT_EQ(result.stack(0, 0), u8(Op::INTEGER));
    EXPECT_EQ(result.integers[0], 1);
}

TEST_F(TestCASSimplifyPipeline, XTimesOne) {
    // X_0 * 1 → X_0
    auto stack = make_stack({
        {u8(Op::VARIABLE), 0, 0},
        {u8(Op::INTEGER), 0, 0},
        {u8(Op::MULTIPLICATION), 0, 1}
    });
    auto result = cas_simplify(stack, {}, {1});
    EXPECT_EQ(result.stack.rows(), 1);
    EXPECT_EQ(result.stack(0, 0), u8(Op::VARIABLE));
}

TEST_F(TestCASSimplifyPipeline, XSquaredViaSquareOp) {
    // SQUARE(X_0) → SQUARE(X_0) in output
    auto stack = make_stack({
        {u8(Op::VARIABLE), 0, 0},
        {u8(Op::SQUARE), 0, 0}
    });
    auto result = cas_simplify(stack, {}, {});
    // After simplification with INSERT_SQUARE_CUBE, x^2 → SQUARE(x)
    EXPECT_EQ(result.stack(result.stack.rows() - 1, 0), u8(Op::SQUARE));
}

TEST_F(TestCASSimplifyPipeline, ConstantMappingPreserved) {
    // C_0 * X_0 → C_0 * X_0, constant_mapping maps correctly
    auto stack = make_stack({
        {u8(Op::CONSTANT), 0, 0},
        {u8(Op::VARIABLE), 0, 0},
        {u8(Op::MULTIPLICATION), 0, 1}
    });
    std::vector<double> constants = {3.14};
    auto result = cas_simplify(stack, constants, {});
    EXPECT_FALSE(result.constant_mapping.empty());
    EXPECT_EQ(result.constant_mapping[0], 0);
}

TEST_F(TestCASSimplifyPipeline, DeadCodeElimination) {
    // Row 0: X_0
    // Row 1: sin(X_0)  — not used
    // Row 2: X_0 + X_0
    // Output = row 2.  Row 1 is dead code.
    auto stack = make_stack({
        {u8(Op::VARIABLE), 0, 0},
        {u8(Op::SIN), 0, 0},
        {u8(Op::ADDITION), 0, 0}
    });
    auto result = cas_simplify(stack, {}, {});
    // X_0 + X_0 = 2*X_0, which is MULTIPLICATION(2, X_0).
    // With INSERT_SQUARE_CUBE off by default, this should be a product.
    // The dead sin row should be gone.
    EXPECT_LE(result.stack.rows(), 3);
}

TEST_F(TestCASSimplifyPipeline, SqrtSquaredSimplifiesToVariable) {
    // sqrt(X_0)^2 → X_0 via power rule simplification
    auto stack = make_stack({
        {u8(Op::VARIABLE), 0, 0},
        {u8(Op::SQRT), 0, 0},
        {u8(Op::INTEGER), 0, 0},
        {u8(Op::POWER), 1, 2}
    });
    auto result = cas_simplify(stack, {}, {2});
    // Should simplify to just X_0
    EXPECT_EQ(result.stack.rows(), 1);
    EXPECT_EQ(result.stack(0, 0), u8(Op::VARIABLE));
}

TEST_F(TestCASSimplifyPipeline, SquareOfSqrtSimplifiesToVariable) {
    // SQUARE(SQRT(X_0)) → X_0 via power rule simplification
    auto stack = make_stack({
        {u8(Op::VARIABLE), 0, 0},
        {u8(Op::SQRT), 0, 0},
        {u8(Op::SQUARE), 1, 1}
    });
    auto result = cas_simplify(stack, {}, {});
    // Should simplify to just X_0
    EXPECT_EQ(result.stack.rows(), 1);
    EXPECT_EQ(result.stack(0, 0), u8(Op::VARIABLE));
}

TEST_F(TestCASSimplifyPipeline, NestedSqrtFourthPowerSimplifies) {
    // sqrt(sqrt(X_0))^4 → X_0 via power rule simplification
    auto stack = make_stack({
        {u8(Op::VARIABLE), 0, 0},
        {u8(Op::SQRT), 0, 0},
        {u8(Op::SQRT), 1, 1},
        {u8(Op::INTEGER), 0, 0},
        {u8(Op::POWER), 2, 3}
    });
    auto result = cas_simplify(stack, {}, {4});
    // Should simplify to just X_0
    EXPECT_EQ(result.stack.rows(), 1);
    EXPECT_EQ(result.stack(0, 0), u8(Op::VARIABLE));
}

TEST_F(TestCASSimplifyPipeline, SqrtRoundtrip) {
    // sqrt(X_0) → CAS pipeline → SQRT(X_0)
    auto stack = make_stack({
        {u8(Op::VARIABLE), 0, 0},
        {u8(Op::SQRT), 0, 0}
    });
    auto result = cas_simplify(stack, {}, {});
    // Should produce SQRT
    EXPECT_EQ(result.stack(result.stack.rows() - 1, 0), u8(Op::SQRT));
}

// ================================================================== //
//  Main                                                               //
// ================================================================== //
