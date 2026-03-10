/**
 * @file test_evaluation.cpp
 * @brief Google Test suite for the cppagraph evaluation engine.
 *
 * Tests forward evaluation, reverse-mode autodiff, and the CachedEvaluator,
 * matching the scenarios from the pyagraph test suite.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "cppagraph/operators.h"
#include "cppagraph/operator_eval.h"
#include "cppagraph/evaluation.h"
#include "cppagraph/cached_evaluation.h"

using namespace cppagraph;

// ================================================================
//  Helpers
// ================================================================

/// Build an Nx3 stack from an initializer list of {node, p1, p2} rows.
static StackMatrix make_stack(
        std::initializer_list<std::array<uint8_t, 3>> rows) {
    StackMatrix s(static_cast<Eigen::Index>(rows.size()), 3);
    int i = 0;
    for (auto& r : rows) {
        s(i, 0) = r[0];
        s(i, 1) = r[1];
        s(i, 2) = r[2];
        ++i;
    }
    return s;
}

/// 3x2 test data: [[1,2],[3,4],[5,6]]
static RowMatrixXd simple_x() {
    RowMatrixXd x(3, 2);
    x << 1, 2, 3, 4, 5, 6;
    return x;
}

// ================================================================
//  Forward operator-level tests
// ================================================================

class OperatorForwardTest : public ::testing::Test {
protected:
    RowMatrixXd x = (RowMatrixXd(2, 2) << 1.0, 2.0, 3.0, 4.0).finished();
    std::vector<double> constants = {2.5, -1.0};
    std::vector<int> integers = {3, 7};
    ForwardBuf fwd;

    void SetUp() override {
        fwd.resize(2);
        fwd[0] = x.col(0);  // [1, 3]
        fwd[1] = x.col(1);  // [2, 4]
    }
};

TEST_F(OperatorForwardTest, Variable) {
    auto r = forward_eval_one(
        static_cast<uint8_t>(Op::VARIABLE), 0, 0,
        x, constants, integers, fwd);
    EXPECT_DOUBLE_EQ(r(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(r(1, 0), 3.0);
}

TEST_F(OperatorForwardTest, Constant) {
    auto r = forward_eval_one(
        static_cast<uint8_t>(Op::CONSTANT), 0, 0,
        x, constants, integers, fwd);
    EXPECT_DOUBLE_EQ(r(0, 0), 2.5);
}

TEST_F(OperatorForwardTest, Integer) {
    auto r = forward_eval_one(
        static_cast<uint8_t>(Op::INTEGER), 1, 0,
        x, constants, integers, fwd);
    EXPECT_DOUBLE_EQ(r(0, 0), 7.0);
}

TEST_F(OperatorForwardTest, Addition) {
    auto r = forward_eval_one(
        static_cast<uint8_t>(Op::ADDITION), 0, 1,
        x, constants, integers, fwd);
    EXPECT_DOUBLE_EQ(r(0, 0), 3.0);   // 1+2
    EXPECT_DOUBLE_EQ(r(1, 0), 7.0);   // 3+4
}

TEST_F(OperatorForwardTest, Subtraction) {
    auto r = forward_eval_one(
        static_cast<uint8_t>(Op::SUBTRACTION), 0, 1,
        x, constants, integers, fwd);
    EXPECT_DOUBLE_EQ(r(0, 0), -1.0);
    EXPECT_DOUBLE_EQ(r(1, 0), -1.0);
}

TEST_F(OperatorForwardTest, Multiplication) {
    auto r = forward_eval_one(
        static_cast<uint8_t>(Op::MULTIPLICATION), 0, 1,
        x, constants, integers, fwd);
    EXPECT_DOUBLE_EQ(r(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(r(1, 0), 12.0);
}

TEST_F(OperatorForwardTest, Division) {
    auto r = forward_eval_one(
        static_cast<uint8_t>(Op::DIVISION), 0, 1,
        x, constants, integers, fwd);
    EXPECT_DOUBLE_EQ(r(0, 0), 0.5);
    EXPECT_DOUBLE_EQ(r(1, 0), 0.75);
}

TEST_F(OperatorForwardTest, Square) {
    auto r = forward_eval_one(
        static_cast<uint8_t>(Op::SQUARE), 0, 0,
        x, constants, integers, fwd);
    EXPECT_DOUBLE_EQ(r(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(r(1, 0), 9.0);
}

TEST_F(OperatorForwardTest, Cube) {
    auto r = forward_eval_one(
        static_cast<uint8_t>(Op::CUBE), 0, 0,
        x, constants, integers, fwd);
    EXPECT_DOUBLE_EQ(r(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(r(1, 0), 27.0);
}

TEST_F(OperatorForwardTest, SqrtProtected) {
    // sqrt(|x|) — test with negative value
    ForwardBuf neg_fwd = {(RowMatrixXd(2, 1) << -4.0, 9.0).finished()};
    auto r = forward_eval_one(
        static_cast<uint8_t>(Op::SQRT), 0, 0,
        x, constants, integers, neg_fwd);
    EXPECT_DOUBLE_EQ(r(0, 0), 2.0);  // sqrt(|-4|)
    EXPECT_DOUBLE_EQ(r(1, 0), 3.0);  // sqrt(|9|)
}

TEST_F(OperatorForwardTest, Abs) {
    ForwardBuf neg_fwd = {(RowMatrixXd(2, 1) << -3.0, 5.0).finished()};
    auto r = forward_eval_one(
        static_cast<uint8_t>(Op::ABS), 0, 0,
        x, constants, integers, neg_fwd);
    EXPECT_DOUBLE_EQ(r(0, 0), 3.0);
    EXPECT_DOUBLE_EQ(r(1, 0), 5.0);
}

TEST_F(OperatorForwardTest, Exponential) {
    auto r = forward_eval_one(
        static_cast<uint8_t>(Op::EXPONENTIAL), 0, 0,
        x, constants, integers, fwd);
    EXPECT_NEAR(r(0, 0), std::exp(1.0), 1e-12);
    EXPECT_NEAR(r(1, 0), std::exp(3.0), 1e-12);
}

TEST_F(OperatorForwardTest, LogProtected) {
    // log(|x|)
    ForwardBuf neg_fwd = {(RowMatrixXd(2, 1) << -1.0, std::exp(2.0)).finished()};
    auto r = forward_eval_one(
        static_cast<uint8_t>(Op::LOGARITHM), 0, 0,
        x, constants, integers, neg_fwd);
    EXPECT_NEAR(r(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(r(1, 0), 2.0, 1e-12);
}

TEST_F(OperatorForwardTest, Sin) {
    auto r = forward_eval_one(
        static_cast<uint8_t>(Op::SIN), 0, 0,
        x, constants, integers, fwd);
    EXPECT_NEAR(r(0, 0), std::sin(1.0), 1e-12);
    EXPECT_NEAR(r(1, 0), std::sin(3.0), 1e-12);
}

TEST_F(OperatorForwardTest, Cos) {
    auto r = forward_eval_one(
        static_cast<uint8_t>(Op::COS), 0, 0,
        x, constants, integers, fwd);
    EXPECT_NEAR(r(0, 0), std::cos(1.0), 1e-12);
    EXPECT_NEAR(r(1, 0), std::cos(3.0), 1e-12);
}

TEST_F(OperatorForwardTest, Power) {
    auto r = forward_eval_one(
        static_cast<uint8_t>(Op::POWER), 0, 1,
        x, constants, integers, fwd);
    EXPECT_NEAR(r(0, 0), std::pow(1.0, 2.0), 1e-12);
    EXPECT_NEAR(r(1, 0), std::pow(3.0, 4.0), 1e-12);
}

TEST_F(OperatorForwardTest, SafePower) {
    ForwardBuf neg_fwd = {
        (RowMatrixXd(2, 1) << -2.0, -4.0).finished(),
        (RowMatrixXd(2, 1) << 3.0, 2.0).finished()
    };
    auto r = forward_eval_one(
        static_cast<uint8_t>(Op::SAFE_POWER), 0, 1,
        x, constants, integers, neg_fwd);
    EXPECT_NEAR(r(0, 0), std::pow(2.0, 3.0), 1e-12);
    EXPECT_NEAR(r(1, 0), std::pow(4.0, 2.0), 1e-12);
}

TEST_F(OperatorForwardTest, TrigAndHyperbolic) {
    // Spot-check a few more
    ForwardBuf small_fwd = {(RowMatrixXd(2,1) << 0.3, 0.5).finished()};
    auto t = forward_eval_one(static_cast<uint8_t>(Op::TAN), 0, 0,
                              x, constants, integers, small_fwd);
    EXPECT_NEAR(t(0,0), std::tan(0.3), 1e-12);

    auto sh = forward_eval_one(static_cast<uint8_t>(Op::SINH), 0, 0,
                               x, constants, integers, small_fwd);
    EXPECT_NEAR(sh(0,0), std::sinh(0.3), 1e-12);

    auto ch = forward_eval_one(static_cast<uint8_t>(Op::COSH), 0, 0,
                               x, constants, integers, small_fwd);
    EXPECT_NEAR(ch(0,0), std::cosh(0.3), 1e-12);

    auto th = forward_eval_one(static_cast<uint8_t>(Op::TANH), 0, 0,
                               x, constants, integers, small_fwd);
    EXPECT_NEAR(th(0,0), std::tanh(0.3), 1e-12);

    auto as = forward_eval_one(static_cast<uint8_t>(Op::ARCSIN), 0, 0,
                               x, constants, integers, small_fwd);
    EXPECT_NEAR(as(0,0), std::asin(0.3), 1e-12);

    auto ac = forward_eval_one(static_cast<uint8_t>(Op::ARCCOS), 0, 0,
                               x, constants, integers, small_fwd);
    EXPECT_NEAR(ac(0,0), std::acos(0.3), 1e-12);

    auto at = forward_eval_one(static_cast<uint8_t>(Op::ARCTAN), 0, 0,
                               x, constants, integers, small_fwd);
    EXPECT_NEAR(at(0,0), std::atan(0.3), 1e-12);
}

// ================================================================
//  Full-stack evaluate() tests
// ================================================================

TEST(Evaluate, SingleVariable) {
    auto x = simple_x();
    auto stack = make_stack({{0, 0, 0}});  // X0
    auto r = evaluate(stack, x, {}, {});
    EXPECT_EQ(r.rows(), 3);
    EXPECT_EQ(r.cols(), 1);
    EXPECT_DOUBLE_EQ(r(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(r(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(r(2, 0), 5.0);
}

TEST(Evaluate, Constant) {
    auto x = simple_x();
    auto stack = make_stack({{1, 0, 0}});  // C0
    auto r = evaluate(stack, x, {3.14}, {});
    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(r(i, 0), 3.14, 1e-12);
}

TEST(Evaluate, Integer) {
    auto x = simple_x();
    auto stack = make_stack({{2, 0, 0}});  // I0
    auto r = evaluate(stack, x, {}, {5});
    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(r(i, 0), 5.0, 1e-12);
}

TEST(Evaluate, X0PlusC0) {
    auto x = simple_x();
    auto stack = make_stack({
        {0, 0, 0},  // X0
        {1, 0, 0},  // C0
        {3, 0, 1},  // ADD(0, 1)
    });
    auto r = evaluate(stack, x, {10.0}, {});
    EXPECT_NEAR(r(0, 0), 11.0, 1e-12);
    EXPECT_NEAR(r(1, 0), 13.0, 1e-12);
    EXPECT_NEAR(r(2, 0), 15.0, 1e-12);
}

TEST(Evaluate, X0TimesX1) {
    auto x = simple_x();
    auto stack = make_stack({
        {0, 0, 0},  // X0
        {0, 1, 1},  // X1
        {5, 0, 1},  // MUL(0, 1)
    });
    auto r = evaluate(stack, x, {}, {});
    EXPECT_NEAR(r(0, 0), 2.0, 1e-12);
    EXPECT_NEAR(r(1, 0), 12.0, 1e-12);
    EXPECT_NEAR(r(2, 0), 30.0, 1e-12);
}

TEST(Evaluate, SinX0) {
    auto x = simple_x();
    auto stack = make_stack({
        {0, 0, 0},   // X0
        {15, 0, 0},  // SIN(0)
    });
    auto r = evaluate(stack, x, {}, {});
    EXPECT_NEAR(r(0, 0), std::sin(1.0), 1e-12);
    EXPECT_NEAR(r(1, 0), std::sin(3.0), 1e-12);
    EXPECT_NEAR(r(2, 0), std::sin(5.0), 1e-12);
}

// ================================================================
//  Derivative tests (evaluate_with_derivative)
// ================================================================

TEST(Derivative, XGradientOfX0) {
    // f(x) = X0  =>  df/dX0 = 1, df/dX1 = 0
    auto x = simple_x();
    auto stack = make_stack({{0, 0, 0}});
    auto [f, df] = evaluate_with_derivative(stack, x, {}, {}, true);
    EXPECT_NEAR(f(0, 0), 1.0, 1e-12);
    EXPECT_EQ(df.rows(), 3);
    EXPECT_EQ(df.cols(), 2);
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(df(i, 0), 1.0, 1e-12);
        EXPECT_NEAR(df(i, 1), 0.0, 1e-12);
    }
}

TEST(Derivative, XGradientOfSum) {
    // f(x) = X0 + X1  =>  df/dX0 = 1, df/dX1 = 1
    auto x = simple_x();
    auto stack = make_stack({
        {0, 0, 0},
        {0, 1, 1},
        {3, 0, 1},
    });
    auto [f, df] = evaluate_with_derivative(stack, x, {}, {}, true);
    EXPECT_NEAR(f(0, 0), 3.0, 1e-12);  // 1+2
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(df(i, 0), 1.0, 1e-12);
        EXPECT_NEAR(df(i, 1), 1.0, 1e-12);
    }
}

TEST(Derivative, ConstGradient) {
    // f(x) = C0 * X0  =>  df/dC0 = X0
    auto x = simple_x();
    auto stack = make_stack({
        {1, 0, 0},  // C0
        {0, 0, 0},  // X0
        {5, 0, 1},  // MUL(C0, X0)
    });
    auto [f, df] = evaluate_with_derivative(stack, x, {2.0}, {}, false);
    EXPECT_NEAR(f(0, 0), 2.0, 1e-12);    // 2.0 * 1
    EXPECT_NEAR(f(1, 0), 6.0, 1e-12);    // 2.0 * 3
    EXPECT_NEAR(f(2, 0), 10.0, 1e-12);   // 2.0 * 5
    // df/dC0 = X0
    EXPECT_NEAR(df(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(df(1, 0), 3.0, 1e-12);
    EXPECT_NEAR(df(2, 0), 5.0, 1e-12);
}

TEST(Derivative, XGradientOfSubtraction) {
    // f(x) = X0 - X1  =>  df/dX0 = 1, df/dX1 = -1
    auto x = simple_x();
    auto stack = make_stack({
        {0, 0, 0},
        {0, 1, 1},
        {4, 0, 1},  // SUB
    });
    auto [f, df] = evaluate_with_derivative(stack, x, {}, {}, true);
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(df(i, 0), 1.0, 1e-12);
        EXPECT_NEAR(df(i, 1), -1.0, 1e-12);
    }
}

TEST(Derivative, XGradientOfSin) {
    // f(x) = sin(X0)  =>  df/dX0 = cos(X0)
    auto x = simple_x();
    auto stack = make_stack({
        {0, 0, 0},
        {15, 0, 0},  // SIN
    });
    auto [f, df] = evaluate_with_derivative(stack, x, {}, {}, true);
    EXPECT_NEAR(df(0, 0), std::cos(1.0), 1e-12);
    EXPECT_NEAR(df(1, 0), std::cos(3.0), 1e-12);
    EXPECT_NEAR(df(2, 0), std::cos(5.0), 1e-12);
    // df/dX1 = 0
    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(df(i, 1), 0.0, 1e-12);
}

// ================================================================
//  CachedEvaluator tests
// ================================================================

TEST(CachedEvaluator, ForwardEvalMatchesEvaluate) {
    auto x = simple_x();
    auto stack = make_stack({
        {1, 0, 0},  // C0
        {0, 0, 0},  // X0
        {5, 0, 1},  // MUL
    });
    std::vector<double> c = {2.0};
    std::vector<int> ints = {};

    auto expected = evaluate(stack, x, c, ints);

    CachedEvaluator ce(stack, x, ints);
    auto result = ce.forward_eval(c);

    EXPECT_EQ(result.rows(), expected.rows());
    EXPECT_EQ(result.cols(), expected.cols());
    for (int i = 0; i < result.rows(); ++i)
        EXPECT_NEAR(result(i, 0), expected(i, 0), 1e-12);
}

TEST(CachedEvaluator, DerivativeMatchesEvaluateWithDerivative) {
    auto x = simple_x();
    auto stack = make_stack({
        {1, 0, 0},  // C0
        {0, 0, 0},  // X0
        {5, 0, 1},  // MUL
    });
    std::vector<double> c = {2.0};
    std::vector<int> ints = {};

    auto [exp_f, exp_df] = evaluate_with_derivative(stack, x, c, ints, false);

    CachedEvaluator ce(stack, x, ints);
    auto [f, df] = ce.forward_eval_with_const_derivative(c);

    for (int i = 0; i < f.rows(); ++i) {
        EXPECT_NEAR(f(i, 0), exp_f(i, 0), 1e-12);
        EXPECT_NEAR(df(i, 0), exp_df(i, 0), 1e-12);
    }
}

TEST(CachedEvaluator, FusedResidualJacobianSkipsForward) {
    auto x = simple_x();
    auto stack = make_stack({
        {1, 0, 0},
        {0, 0, 0},
        {5, 0, 1},
    });
    std::vector<double> c = {3.0};
    std::vector<int> ints = {};

    CachedEvaluator ce(stack, x, ints);
    // First forward_eval caches the forward buffer.
    auto f = ce.forward_eval(c);
    // Second call with same constants should reuse the cached forward.
    auto [f2, df2] = ce.forward_eval_with_const_derivative(c);

    for (int i = 0; i < f.rows(); ++i)
        EXPECT_NEAR(f(i, 0), f2(i, 0), 1e-12);
}

TEST(CachedEvaluator, RepeatedCallsWithDifferentConstants) {
    auto x = simple_x();
    auto stack = make_stack({
        {1, 0, 0},
        {0, 0, 0},
        {3, 0, 1},  // C0 + X0
    });
    std::vector<int> ints = {};

    CachedEvaluator ce(stack, x, ints);

    // First call
    auto r1 = ce.forward_eval({10.0});
    EXPECT_NEAR(r1(0, 0), 11.0, 1e-12);

    // Second call — different constant, static cache should be used
    auto r2 = ce.forward_eval({20.0});
    EXPECT_NEAR(r2(0, 0), 21.0, 1e-12);
    EXPECT_NEAR(r2(1, 0), 23.0, 1e-12);
}

TEST(CachedEvaluator, CachedReversePassConsistency) {
    // Multiple derivative calls should give consistent results.
    auto x = simple_x();
    auto stack = make_stack({
        {1, 0, 0},  // C0
        {0, 0, 0},  // X0
        {5, 0, 1},  // MUL
    });
    std::vector<int> ints = {};

    CachedEvaluator ce(stack, x, ints);

    auto [f1, df1] = ce.forward_eval_with_const_derivative({2.0});
    auto [f2, df2] = ce.forward_eval_with_const_derivative({5.0});

    // df/dC0 = X0 regardless of C0 value
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(df1(i, 0), x(i, 0), 1e-12);
        EXPECT_NEAR(df2(i, 0), x(i, 0), 1e-12);
    }
    // f values change: C0 * X0
    EXPECT_NEAR(f1(0, 0), 2.0, 1e-12);
    EXPECT_NEAR(f2(0, 0), 5.0, 1e-12);
}

// ================================================================
//  Dependency mask tests
// ================================================================

TEST(DependencyMask, BasicMask) {
    // Stack: [X0, C0, X0+C0]
    auto stack = make_stack({
        {0, 0, 0},
        {1, 0, 0},
        {3, 0, 1},
    });
    auto mask = build_dependency_mask(stack);
    EXPECT_FALSE(mask[0]);  // X0
    EXPECT_TRUE(mask[1]);   // C0
    EXPECT_TRUE(mask[2]);   // X0+C0 depends on C0
}

TEST(DependencyMask, NoConstants) {
    auto stack = make_stack({
        {0, 0, 0},
        {0, 1, 1},
        {3, 0, 1},
    });
    auto mask = build_dependency_mask(stack);
    for (auto v : mask) EXPECT_FALSE(v);
}
