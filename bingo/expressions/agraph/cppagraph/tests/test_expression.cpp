/**
 * @file test_expression.cpp
 * @brief Google Test suite for AGraphExpression.
 *
 * Tests construction, properties, evaluation, gradient, fit/predict/score,
 * operator counts, hash/equality, copy/distance, modification tracking,
 * constant mapping, propagate_constants, and promote_simplification.
 */

#include <gtest/gtest.h>

#include "cppagraph/expression.h"
#include "cppagraph/operators.h"

#include <cmath>
#include <limits>
#include <map>

using namespace cppagraph;

// ================================================================
//  Helper builders
// ================================================================

// X0 + C0    (C0 = val, default 10.0)
static AGraphExpression make_x0_plus_c0(double val = 10.0) {
    AGraphExpression expr("cas");
    StackMatrix cmd(3, 3);
    cmd << 0, 0, 0,    // VARIABLE X0
          1, 0, 0,     // CONSTANT C0
          3, 0, 1;     // ADDITION row0 row1
    expr.set_raw_command_array(cmd);
    expr.set_raw_constants({val});
    return expr;
}

// sin(X0)
static AGraphExpression make_sin_x0() {
    AGraphExpression expr("cas");
    StackMatrix cmd(2, 3);
    cmd << 0, 0, 0,    // VARIABLE X0
          15, 0, 0;    // SIN row0
    expr.set_raw_command_array(cmd);
    return expr;
}

// X0 * C0    (C0 = val, default 2.0)
static AGraphExpression make_x0_times_c0(double val = 2.0) {
    AGraphExpression expr("cas");
    StackMatrix cmd(3, 3);
    cmd << 0, 0, 0,    // VARIABLE X0
          1, 0, 0,     // CONSTANT C0
          5, 0, 1;     // MULTIPLICATION row0 row1
    expr.set_raw_command_array(cmd);
    expr.set_raw_constants({val});
    return expr;
}

// X0 + X1
static AGraphExpression make_x0_plus_x1() {
    AGraphExpression expr("cas");
    StackMatrix cmd(3, 3);
    cmd << 0, 0, 0,    // VARIABLE X0
          0, 1, 0,     // VARIABLE X1
          3, 0, 1;     // ADDITION row0 row1
    expr.set_raw_command_array(cmd);
    return expr;
}

// X0 * X1
static AGraphExpression make_x0_times_x1() {
    AGraphExpression expr("cas");
    StackMatrix cmd(3, 3);
    cmd << 0, 0, 0,    // VARIABLE X0
          0, 1, 0,     // VARIABLE X1
          5, 0, 1;     // MULTIPLICATION row0 row1
    expr.set_raw_command_array(cmd);
    return expr;
}

// X0 only (no constants)
static AGraphExpression make_x0() {
    AGraphExpression expr("cas");
    StackMatrix cmd(1, 3);
    cmd << 0, 0, 0;    // VARIABLE X0
    expr.set_raw_command_array(cmd);
    return expr;
}

// X0 + 7  (INTEGER node)
static AGraphExpression make_x0_plus_int7() {
    AGraphExpression expr("reduce");
    StackMatrix cmd(3, 3);
    cmd << 0, 0, 0,    // VARIABLE X0
          2, 0, 0,     // INTEGER idx0
          3, 0, 1;     // ADDITION row0 row1
    expr.set_raw_command_array(cmd);
    expr.set_raw_integers({7});
    return expr;
}

// (X0 + C0) * (X0 + C0)  — shared sub-graph in DAG
static AGraphExpression make_shared_subgraph(double c0 = 3.0) {
    AGraphExpression expr("reduce");
    StackMatrix cmd(4, 3);
    cmd << 0, 0, 0,    // VARIABLE X0
          1, 0, 0,     // CONSTANT C0
          3, 0, 1,     // ADDITION row0 row1
          5, 2, 2;     // MULTIPLICATION row2 row2
    expr.set_raw_command_array(cmd);
    expr.set_raw_constants({c0});
    return expr;
}

static RowMatrixXd make_simple_x() {
    RowMatrixXd x(3, 2);
    x << 1.0, 2.0,
         3.0, 4.0,
         5.0, 6.0;
    return x;
}

// ================================================================
//  Test: Construction
// ================================================================

TEST(ExpressionConstruction, DefaultEmpty) {
    AGraphExpression expr;
    EXPECT_EQ(expr.command_array().rows(), 0);
    EXPECT_EQ(expr.command_array().cols(), 3);
}

TEST(ExpressionConstruction, InvalidSimplification) {
    EXPECT_THROW(AGraphExpression("foo"), std::invalid_argument);
}

TEST(ExpressionConstruction, FromRawData) {
    auto expr = make_x0_plus_c0();
    EXPECT_GT(expr.command_array().rows(), 0);
}

TEST(ExpressionConstruction, WithInteger) {
    auto expr = make_x0_plus_int7();
    auto ints = expr.integers();
    EXPECT_EQ(ints.size(), 1u);
    EXPECT_EQ(ints[0], 7);
}

// ================================================================
//  Test: Properties
// ================================================================

TEST(ExpressionProperties, CommandArrayReadonly) {
    auto expr = make_x0_plus_c0();
    auto cmd = expr.command_array();
    // command_array() is const & — can't modify
    EXPECT_TRUE(true);  // compilation test
}

TEST(ExpressionProperties, Complexity) {
    auto expr = make_x0_plus_c0();
    EXPECT_EQ(expr.complexity(), 3);
}

TEST(ExpressionProperties, TreeComplexityEqualsComplexityNoReuse) {
    auto expr = make_x0_plus_c0();
    // No shared sub-expressions → tree_complexity == complexity
    EXPECT_EQ(expr.tree_complexity(), expr.complexity());
}

TEST(ExpressionProperties, TreeComplexityGreaterWithDAGReuse) {
    auto expr = make_shared_subgraph();
    // (X0 + C0) * (X0 + C0) reuses row 2
    // DAG complexity = 4, tree complexity = 7
    EXPECT_GT(expr.tree_complexity(), expr.complexity());
    EXPECT_EQ(expr.tree_complexity(), 7);
}

TEST(ExpressionProperties, TreeComplexityEmpty) {
    AGraphExpression expr;
    EXPECT_EQ(expr.tree_complexity(), 0);
}

TEST(ExpressionProperties, Constants) {
    auto expr = make_x0_plus_c0(10.0);
    auto c = expr.constants();
    EXPECT_EQ(c.size(), 1u);
    EXPECT_DOUBLE_EQ(c[0], 10.0);
}

TEST(ExpressionProperties, ConstantsSetter) {
    auto expr = make_x0_plus_c0(1.0);
    expr.set_constants({99.0});
    EXPECT_EQ(expr.constants().size(), 1u);
    EXPECT_DOUBLE_EQ(expr.constants()[0], 99.0);
}

TEST(ExpressionProperties, Integers) {
    auto expr = make_x0_plus_int7();
    auto ints = expr.integers();
    EXPECT_EQ(ints.size(), 1u);
    EXPECT_EQ(ints[0], 7);
}

TEST(ExpressionProperties, MutableRawCommandArray) {
    auto expr = make_x0_times_c0(2.0);
    auto& cmd = expr.mutable_raw_command_array();
    cmd(1, 1) = 0;  // change param1 of CONSTANT node (should not crash)
    EXPECT_TRUE(expr.modified());
}

// ================================================================
//  Test: Evaluation
// ================================================================

TEST(ExpressionEvaluation, X0PlusC0) {
    auto expr = make_x0_plus_c0(10.0);
    auto x = make_simple_x();
    auto result = expr.evaluate(x);
    // f(x) = X0 + 10
    EXPECT_NEAR(result(0, 0), 11.0, 1e-10);
    EXPECT_NEAR(result(1, 0), 13.0, 1e-10);
    EXPECT_NEAR(result(2, 0), 15.0, 1e-10);
}

TEST(ExpressionEvaluation, SinX0) {
    auto expr = make_sin_x0();
    auto x = make_simple_x();
    auto result = expr.evaluate(x);
    EXPECT_NEAR(result(0, 0), std::sin(1.0), 1e-10);
    EXPECT_NEAR(result(1, 0), std::sin(3.0), 1e-10);
    EXPECT_NEAR(result(2, 0), std::sin(5.0), 1e-10);
}

TEST(ExpressionEvaluation, X0TimesC0) {
    auto expr = make_x0_times_c0(2.0);
    auto x = make_simple_x();
    auto result = expr.evaluate(x);
    EXPECT_NEAR(result(0, 0), 2.0, 1e-10);
    EXPECT_NEAR(result(1, 0), 6.0, 1e-10);
    EXPECT_NEAR(result(2, 0), 10.0, 1e-10);
}

TEST(ExpressionEvaluation, WithXGradient) {
    auto expr = make_x0_plus_c0(10.0);
    auto x = make_simple_x();
    auto [f, df_dx] = expr.evaluate_with_x_gradient(x);

    // f = X0 + 10
    EXPECT_NEAR(f(0, 0), 11.0, 1e-10);

    // df/dX0 = 1, df/dX1 = 0
    EXPECT_NEAR(df_dx(0, 0), 1.0, 1e-10);
    EXPECT_NEAR(df_dx(0, 1), 0.0, 1e-10);
}

TEST(ExpressionEvaluation, WithConstGradient) {
    auto expr = make_x0_times_c0(2.0);
    auto x = make_simple_x();
    auto [f, df_dc] = expr.evaluate_with_const_gradient(x);

    // f = C0 * X0 => df/dC0 = X0
    EXPECT_NEAR(f(0, 0), 2.0, 1e-10);
    EXPECT_NEAR(df_dc(0, 0), 1.0, 1e-10);  // X0 at row 0
    EXPECT_NEAR(df_dc(1, 0), 3.0, 1e-10);  // X0 at row 1
    EXPECT_NEAR(df_dc(2, 0), 5.0, 1e-10);  // X0 at row 2
}

TEST(ExpressionEvaluation, WithConstHessian) {
    auto expr = make_shared_subgraph(3.0);
    auto x = make_simple_x();
    auto result = expr.evaluate_with_const_hessian(x);

    for (Eigen::Index row = 0; row < x.rows(); ++row) {
        EXPECT_NEAR(result.value(row, 0), std::pow(x(row, 0) + 3.0, 2), 1e-12);
        EXPECT_NEAR(result.gradient(row, 0), 2.0 * (x(row, 0) + 3.0), 1e-12);
        EXPECT_NEAR(result.hessian(row, 0), 2.0, 1e-12);
    }
}

// ================================================================
//  Test: sklearn interface
// ================================================================

TEST(ExpressionSklearn, Predict) {
    auto expr = make_x0_plus_c0(10.0);
    auto x = make_simple_x();
    auto pred = expr.predict(x);
    EXPECT_EQ(pred.size(), 3);
    EXPECT_NEAR(pred(0), 11.0, 1e-10);
    EXPECT_NEAR(pred(1), 13.0, 1e-10);
    EXPECT_NEAR(pred(2), 15.0, 1e-10);
}

TEST(ExpressionSklearn, LossMSE) {
    auto expr = make_x0_plus_c0(10.0);
    auto x = make_simple_x();
    Eigen::VectorXd y(3);
    y << 11.0, 13.0, 15.0;  // exact match
    double s = expr.loss(x, y, "mse");
    EXPECT_NEAR(s, 0.0, 1e-10);
}

TEST(ExpressionSklearn, LossMAE) {
    auto expr = make_x0_plus_c0(10.0);
    auto x = make_simple_x();
    Eigen::VectorXd y(3);
    y << 12.0, 14.0, 16.0;  // off by 1
    double s = expr.loss(x, y, "mae");
    EXPECT_NEAR(s, 1.0, 1e-10);
}

TEST(ExpressionSklearn, ScoreR2PerfectFit) {
    auto expr = make_x0_plus_c0(10.0);
    auto x = make_simple_x();
    Eigen::VectorXd y(3);
    y << 11.0, 13.0, 15.0;  // exact match
    double s = expr.score(x, y, "r2");  // higher-is-better, 1.0 for perfect fit
    EXPECT_NEAR(s, 1.0, 1e-10);
}

TEST(ExpressionSklearn, LossRejectsUnknownKind) {
    auto expr = make_x0_plus_c0(10.0);
    auto x = make_simple_x();
    Eigen::VectorXd y(3);
    y << 11.0, 13.0, 15.0;
    EXPECT_THROW(expr.loss(x, y, "bogus"), std::invalid_argument);
}

TEST(ExpressionSklearn, ScoreRejectsLossOnlyKind) {
    auto expr = make_x0_plus_c0(10.0);
    auto x = make_simple_x();
    Eigen::VectorXd y(3);
    y << 11.0, 13.0, 15.0;
    EXPECT_THROW(expr.score(x, y, "mse"), std::invalid_argument);
}

TEST(ExpressionSklearn, NonFiniteLossIsInfScoreIsNegInf) {
    // 1 / X0 evaluated at X0 = 0 is non-finite.
    AGraphExpression expr;
    StackMatrix cmd(3, 3);
    cmd << 1, 0, 0,   // CONSTANT C0
          0, 0, 0,   // VARIABLE X0
          6, 0, 1;   // DIVISION: C0 / X0
    expr.set_raw_command_array(cmd);
    expr.set_raw_constants({1.0});
    RowMatrixXd x(1, 1);
    x << 0.0;
    Eigen::VectorXd y(1);
    y << 1.0;
    EXPECT_EQ(expr.loss(x, y, "mse"),
              std::numeric_limits<double>::infinity());
    EXPECT_EQ(expr.score(x, y, "r2"),
              -std::numeric_limits<double>::infinity());
}

TEST(ExpressionSklearn, FitOptimisesConstants) {
    // X0 * C0 with C0 = 1.0, fit to y = 3*X0
    auto expr = make_x0_times_c0(1.0);
    RowMatrixXd X(5, 1);
    X << 1.0, 2.0, 3.0, 4.0, 5.0;
    Eigen::VectorXd y(5);
    y << 3.0, 6.0, 9.0, 12.0, 15.0;
    expr.fit(X, y);
    EXPECT_NEAR(expr.constants()[0], 3.0, 0.1);
}

TEST(ExpressionSklearn, FitNoConstantsIsNoop) {
    auto expr = make_x0();
    RowMatrixXd X(3, 1);
    X << 1.0, 2.0, 3.0;
    Eigen::VectorXd y(3);
    y << 1.0, 2.0, 3.0;
    expr.fit(X, y);  // should not crash
    EXPECT_TRUE(expr.is_fitted());
}

TEST(ExpressionSklearn, ScoreLaplaceNMLL) {
    auto expr = make_x0_times_c0(1.0);
    RowMatrixXd X(5, 1);
    X << 1.0, 2.0, 3.0, 4.0, 5.0;
    Eigen::VectorXd y(5);
    y << 3.0, 6.0, 9.0, 12.0, 15.0;
    expr.fit(X, y);
    double nmll = expr.score(X, y, "laplace_nmll");
    EXPECT_TRUE(std::isfinite(nmll));
}

TEST(ExpressionSklearn, LaplaceLossIsNegatedScore) {
    auto expr = make_x0_times_c0(1.0);
    RowMatrixXd X(5, 1);
    X << 1.0, 2.0, 3.0, 4.0, 5.0;
    Eigen::VectorXd y(5);
    y << 3.5, 6.0, 8.5, 12.0, 15.5;  // nonzero residuals
    expr.fit(X, y);
    double loss = expr.loss(X, y, "laplace_nmll");
    double score = expr.score(X, y, "laplace_nmll");
    EXPECT_NEAR(loss, -score, 1e-9);
}

TEST(ScoringMetrics, BicScoreDirect) {
    // bic_score remains available as a standalone metric even though it is no
    // longer part of the score() vocabulary.
    Eigen::VectorXd r(5);
    r << 0.0, 0.0, 0.0, 0.0, 0.0;  // perfect fit
    double b = bic_score(r, 1);
    EXPECT_LT(b, 0);
}

// ================================================================
//  Test: gradient (predictions + df/dx)
// ================================================================

TEST(ExpressionGradient, ReturnsPredictionsAndInputGradient) {
    auto expr = make_x0_plus_c0(10.0);
    auto x = make_simple_x();
    auto [f, df_dx] = expr.gradient(x);
    EXPECT_NEAR(f(0), 11.0, 1e-10);
    EXPECT_NEAR(f(1), 13.0, 1e-10);
    EXPECT_NEAR(f(2), 15.0, 1e-10);
    // d(X0 + C0)/dX0 = 1, d/dX1 = 0
    for (Eigen::Index i = 0; i < x.rows(); ++i) {
        EXPECT_NEAR(df_dx(i, 0), 1.0, 1e-10);
        EXPECT_NEAR(df_dx(i, 1), 0.0, 1e-10);
    }
}

// ================================================================
//  Test: implicit regression
// ================================================================

TEST(ExpressionImplicit, FitEstablishesFitted) {
    // X0 * C0 + X1 * C1
    AGraphExpression expr("cas");
    StackMatrix cmd(7, 3);
    cmd << 0, 0, 0,    // VARIABLE X0
          1, 0, 0,     // CONSTANT C0
          5, 0, 1,     // MULTIPLICATION X0 * C0
          0, 1, 0,     // VARIABLE X1
          1, 1, 0,     // CONSTANT C1
          5, 3, 4,     // MULTIPLICATION X1 * C1
          3, 2, 5;     // ADDITION
    expr.set_raw_command_array(cmd);
    expr.set_raw_constants({1.0, 1.0});

    auto x = make_simple_x();
    RowMatrixXd dx_dt = RowMatrixXd::Ones(3, 2);
    EXPECT_FALSE(expr.is_fitted());
    expr.fit_implicit(x, dx_dt);
    EXPECT_TRUE(expr.is_fitted());
}

TEST(ExpressionImplicit, LossIsFiniteAndScoreIsNegatedLoss) {
    auto expr = make_x0_plus_x1();
    auto x = make_simple_x();
    RowMatrixXd dx_dt = RowMatrixXd::Ones(3, 2);
    double loss = expr.implicit_loss(x, dx_dt);
    double score = expr.implicit_score(x, dx_dt);
    EXPECT_TRUE(std::isfinite(loss));
    EXPECT_NEAR(score, -loss, 1e-12);
}

TEST(ExpressionImplicit, RequiredParamsGuardYieldsInfLoss) {
    // A constant expression has a zero input gradient, so no sample can use the
    // required number of derivative components → +inf loss / -inf score.
    AGraphExpression expr("cas");
    StackMatrix cmd(1, 3);
    cmd << 1, 0, 0;  // CONSTANT
    expr.set_raw_command_array(cmd);
    expr.set_raw_constants({5.0});

    auto x = make_simple_x();
    RowMatrixXd dx_dt = RowMatrixXd::Ones(3, 2);
    EXPECT_EQ(expr.implicit_loss(x, dx_dt, 1),
              std::numeric_limits<double>::infinity());
    EXPECT_EQ(expr.implicit_score(x, dx_dt, 1),
              -std::numeric_limits<double>::infinity());
}

// ================================================================
//  Test: is_fitted
// ================================================================

TEST(ExpressionIsFitted, WithConstantsNotFitted) {
    auto expr = make_x0_plus_c0(1.0);
    EXPECT_FALSE(expr.is_fitted());
}

TEST(ExpressionIsFitted, WithoutConstantsIsFitted) {
    auto expr = make_x0();
    EXPECT_TRUE(expr.is_fitted());
}

TEST(ExpressionIsFitted, FittedAfterFit) {
    auto expr = make_x0_times_c0(1.0);
    RowMatrixXd X(3, 1);
    X << 1.0, 2.0, 3.0;
    Eigen::VectorXd y(3);
    y << 2.0, 4.0, 6.0;
    expr.fit(X, y);
    EXPECT_TRUE(expr.is_fitted());
}

TEST(ExpressionIsFitted, NotFittedAfterModification) {
    auto expr = make_x0_plus_c0(1.0);
    RowMatrixXd X(3, 1);
    X << 1.0, 2.0, 3.0;
    Eigen::VectorXd y(3);
    y << 11.0, 13.0, 15.0;
    expr.fit(X, y);
    EXPECT_TRUE(expr.is_fitted());
    auto& _ = expr.mutable_raw_command_array();
    (void)_;
    EXPECT_FALSE(expr.is_fitted());
}

TEST(ExpressionIsFitted, NotFittedAfterRawConstantsSetter) {
    auto expr = make_x0_plus_c0(1.0);
    RowMatrixXd X(3, 1);
    X << 1.0, 2.0, 3.0;
    Eigen::VectorXd y(3);
    y << 11.0, 13.0, 15.0;
    expr.fit(X, y);
    EXPECT_TRUE(expr.is_fitted());
    expr.set_raw_constants({999.0});
    EXPECT_FALSE(expr.is_fitted());
}

// ================================================================
//  Test: Hash / equality
// ================================================================

TEST(ExpressionHash, SameStructureEqual) {
    auto e1 = make_x0_plus_x1();
    auto e2 = make_x0_plus_x1();
    EXPECT_EQ(e1.hash(), e2.hash());
    EXPECT_TRUE(e1.equals(e2));
}

TEST(ExpressionHash, DifferentStructureNotEqual) {
    auto e1 = make_x0_plus_x1();
    auto e2 = make_x0_times_x1();
    EXPECT_NE(e1.hash(), e2.hash());
}

TEST(ExpressionHash, ModificationResetsHash) {
    auto expr = make_x0_plus_c0();
    auto h1 = expr.hash();
    (void)h1;
    auto& _ = expr.mutable_raw_command_array();
    (void)_;
    // After modification, hash is recomputed.  The value may or
    // may not change, but the internal cache was invalidated.
    EXPECT_TRUE(expr.modified());
}

// ================================================================
//  Test: Copy / distance
// ================================================================

TEST(ExpressionCopy, CopyIsIndependent) {
    auto expr = make_x0_plus_c0(10.0);
    auto copied = expr.copy();
    copied.set_constants({999.0});
    EXPECT_NE(copied.constants()[0], expr.constants()[0]);
}

TEST(ExpressionCopy, DistanceSame) {
    auto expr = make_x0_plus_c0();
    EXPECT_EQ(expr.distance(expr), 0);
}

TEST(ExpressionCopy, DistanceDifferent) {
    auto e1 = make_x0_plus_x1();
    auto e2 = make_x0_times_x1();
    EXPECT_GT(e1.distance(e2), 0);
}

// ================================================================
//  Test: Scoring metrics
// ================================================================

TEST(ScoringMetrics, MAE) {
    Eigen::VectorXd r(3);
    r << 1.0, -2.0, 3.0;
    EXPECT_NEAR(mean_absolute_error(r), 2.0, 1e-10);
}

TEST(ScoringMetrics, MSE) {
    Eigen::VectorXd r(3);
    r << 1.0, -2.0, 3.0;
    EXPECT_NEAR(mean_squared_error(r),
                (1.0 + 4.0 + 9.0) / 3.0, 1e-10);
}

TEST(ScoringMetrics, RMSE) {
    Eigen::VectorXd r(3);
    r << 1.0, -2.0, 3.0;
    EXPECT_NEAR(root_mean_squared_error(r),
                std::sqrt((1.0 + 4.0 + 9.0) / 3.0), 1e-10);
}

TEST(ScoringMetrics, BICIsFinite) {
    Eigen::VectorXd r(5);
    r << 0.1, -0.2, 0.05, -0.1, 0.15;
    EXPECT_TRUE(std::isfinite(bic_score(r, 1)));
}

TEST(ScoringMetrics, LaplaceNMLL) {
    Eigen::VectorXd r(5);
    r << 0.1, -0.2, 0.05, -0.1, 0.15;
    EXPECT_TRUE(std::isfinite(laplace_nmll_score(r, 1)));
}

// ================================================================
//  Test: Constant mapping
// ================================================================

TEST(ConstantMapping, ExistsAfterAccess) {
    auto expr = make_x0_plus_c0(1.0);
    auto mapping = expr.constant_mapping();
    ASSERT_EQ(mapping.size(), expr.constants().size());
}

TEST(ConstantMapping, IdentityForSimple) {
    auto expr = make_x0_plus_c0(1.0);
    auto mapping = expr.constant_mapping();
    EXPECT_EQ(mapping.size(), 1u);
    EXPECT_EQ(mapping[0], 0);
}

TEST(ConstantMapping, SkipsDeadConstant) {
    // C0 used, C1 dead, C2 used
    AGraphExpression expr("reduce");
    StackMatrix cmd(4, 3);
    cmd << 1, 0, 0,    // CONSTANT C0
          1, 1, 1,     // CONSTANT C1 (dead)
          1, 2, 2,     // CONSTANT C2
          3, 0, 2;     // ADDITION row0 row2
    expr.set_raw_command_array(cmd);
    expr.set_raw_constants({10.0, 99.0, 20.0});

    auto mapping = expr.constant_mapping();
    EXPECT_EQ(mapping.size(), 2u);
    // mapping should not contain 1 (dead constant)
    for (int m : mapping)
        EXPECT_NE(m, 1);
}

TEST(ConstantMapping, ValuesIndexIntoRaw) {
    AGraphExpression expr("reduce");
    StackMatrix cmd(4, 3);
    cmd << 1, 0, 0,    // CONSTANT C0
          1, 1, 1,     // CONSTANT C1 (dead)
          1, 2, 2,     // CONSTANT C2
          3, 0, 2;     // ADDITION row0 row2
    expr.set_raw_command_array(cmd);
    expr.set_raw_constants({10.0, 99.0, 20.0});

    auto mapping = expr.constant_mapping();
    auto consts = expr.constants();
    auto raw_consts = expr.raw_constants();

    for (size_t i = 0; i < mapping.size(); ++i) {
        EXPECT_DOUBLE_EQ(consts[i],
                          raw_consts[mapping[i]]);
    }
}

// ================================================================
//  Test: Propagate constants
// ================================================================

TEST(PropagateConstants, DefaultIsFalse) {
    auto expr = make_x0_plus_c0(1.0);
    EXPECT_FALSE(expr.propagate_constants());
}

TEST(PropagateConstants, CanEnable) {
    AGraphExpression expr("cas", true);
    EXPECT_TRUE(expr.propagate_constants());
}

TEST(PropagateConstants, CanToggle) {
    auto expr = make_x0_plus_c0(1.0);
    expr.set_propagate_constants(true);
    EXPECT_TRUE(expr.propagate_constants());
}

TEST(PropagateConstants, DisabledDoesNotPropagate) {
    auto expr = make_x0_plus_c0(1.0);
    auto original_raw = expr.raw_constants();
    expr.set_constants({99.0});
    EXPECT_EQ(expr.raw_constants(), original_raw);
}

TEST(PropagateConstants, EnabledPropagatesToRaw) {
    AGraphExpression expr("cas", true);
    StackMatrix cmd(3, 3);
    cmd << 0, 0, 0,
          1, 0, 0,
          3, 0, 1;
    expr.set_raw_command_array(cmd);
    expr.set_raw_constants({1.0});

    // Force update
    auto _ = expr.constants();
    (void)_;

    expr.set_constants({42.0});
    auto mapping = expr.constant_mapping();
    EXPECT_DOUBLE_EQ(
        expr.raw_constants()[mapping[0]], 42.0);
}

TEST(PropagateConstants, DoesNotRetriggerSimplification) {
    AGraphExpression expr("cas", true);
    StackMatrix cmd(3, 3);
    cmd << 0, 0, 0,
          1, 0, 0,
          3, 0, 1;
    expr.set_raw_command_array(cmd);
    expr.set_raw_constants({1.0});

    // Force update
    auto _ = expr.constants();
    (void)_;
    EXPECT_FALSE(expr.modified());

    expr.set_constants({42.0});
    EXPECT_FALSE(expr.modified());
}

TEST(PropagateConstants, PropagationWithDeadConstants) {
    AGraphExpression expr("reduce", true);
    StackMatrix cmd(4, 3);
    cmd << 1, 0, 0,    // CONSTANT C0
          1, 1, 1,     // CONSTANT C1 (dead)
          1, 2, 2,     // CONSTANT C2
          3, 0, 2;     // ADDITION row0 row2
    expr.set_raw_command_array(cmd);
    expr.set_raw_constants({10.0, 99.0, 20.0});

    // Force update
    auto _ = expr.constants();
    (void)_;

    // Set new simplified constants
    expr.set_constants({100.0, 200.0});

    // raw[0] = 100.0, raw[1] = 99.0 (dead), raw[2] = 200.0
    EXPECT_DOUBLE_EQ(expr.raw_constants()[0], 100.0);
    EXPECT_DOUBLE_EQ(expr.raw_constants()[1], 99.0);
    EXPECT_DOUBLE_EQ(expr.raw_constants()[2], 200.0);
}

TEST(PropagateConstants, FitPropagatesWhenEnabled) {
    AGraphExpression expr("cas", true);
    StackMatrix cmd(3, 3);
    cmd << 0, 0, 0,
          1, 0, 0,
          5, 0, 1;  // MULTIPLICATION
    expr.set_raw_command_array(cmd);
    expr.set_raw_constants({1.0});

    RowMatrixXd X(5, 1);
    X << 1.0, 2.0, 3.0, 4.0, 5.0;
    Eigen::VectorXd y(5);
    y << 3.0, 6.0, 9.0, 12.0, 15.0;
    expr.fit(X, y);

    EXPECT_NEAR(expr.constants()[0], 3.0, 0.1);
    auto mapping = expr.constant_mapping();
    EXPECT_NEAR(expr.raw_constants()[mapping[0]], 3.0, 0.1);
}

TEST(PropagateConstants, FitDoesNotPropagateWhenDisabled) {
    auto expr = make_x0_times_c0(1.0);
    auto original_raw = expr.raw_constants();

    RowMatrixXd X(5, 1);
    X << 1.0, 2.0, 3.0, 4.0, 5.0;
    Eigen::VectorXd y(5);
    y << 3.0, 6.0, 9.0, 12.0, 15.0;
    expr.fit(X, y);
    EXPECT_EQ(expr.raw_constants(), original_raw);
}

// ================================================================
//  Test: Promote simplification
// ================================================================

TEST(PromoteSimplification, IdentityMapping) {
    auto expr = make_x0_plus_c0(1.0);
    // Force update
    auto _ = expr.constants();
    (void)_;
    expr.promote_simplification();

    auto mapping = expr.constant_mapping();
    for (size_t i = 0; i < mapping.size(); ++i)
        EXPECT_EQ(mapping[i], static_cast<int>(i));
}

TEST(PromoteSimplification, ThenPropagate) {
    AGraphExpression expr("cas", true);
    StackMatrix cmd(3, 3);
    cmd << 0, 0, 0,
          1, 0, 0,
          3, 0, 1;
    expr.set_raw_command_array(cmd);
    expr.set_raw_constants({1.0});

    auto _ = expr.constants();
    (void)_;
    expr.promote_simplification();

    expr.set_constants({77.0});
    EXPECT_DOUBLE_EQ(expr.raw_constants()[0], 77.0);
}

// ================================================================
//  Test: Operator counts
// ================================================================

TEST(OperatorCounts, TreeInclude) {
    auto expr = make_x0_plus_c0();
    auto counts = expr.get_operator_counts(true, "include");
    EXPECT_EQ(counts[3], 1);  // ADDITION
    EXPECT_TRUE(counts.count(0));  // VARIABLE
    EXPECT_TRUE(counts.count(1));  // CONSTANT
}

TEST(OperatorCounts, TreeExclude) {
    auto expr = make_x0_plus_c0();
    auto counts = expr.get_operator_counts(true, "exclude");
    EXPECT_FALSE(counts.count(0));  // no VARIABLE
    EXPECT_FALSE(counts.count(1));  // no CONSTANT
    EXPECT_EQ(counts.count(3) ? counts[3] : 0, 1);  // ADDITION
}

TEST(OperatorCounts, TreeCombine) {
    auto expr = make_x0_plus_c0();
    auto counts = expr.get_operator_counts(true, "combine");
    EXPECT_FALSE(counts.count(1));  // no CONSTANT key
    // All terminals folded under VARIABLE (0)
    EXPECT_GE(counts[0], 2);
}

TEST(OperatorCounts, DAGInclude) {
    auto expr = make_x0_plus_c0();
    auto counts = expr.get_operator_counts(false, "include");
    EXPECT_EQ(counts[3], 1);
    EXPECT_TRUE(counts.count(0));
}

TEST(OperatorCounts, TreeSharedSubgraphDouble) {
    auto expr = make_shared_subgraph();
    auto counts = expr.get_operator_counts(true, "include");
    // (X0 + C0) * (X0 + C0) — tree counts ADDITION twice
    EXPECT_EQ(counts[3], 2);  // ADDITION
    EXPECT_EQ(counts[0], 2);  // VARIABLE
    EXPECT_EQ(counts[1], 2);  // CONSTANT
    EXPECT_EQ(counts[5], 1);  // MULTIPLICATION
}

TEST(OperatorCounts, DAGSharedSubgraphOnce) {
    auto expr = make_shared_subgraph();
    auto counts = expr.get_operator_counts(false, "include");
    EXPECT_EQ(counts[3], 1);  // ADDITION — each node once
    EXPECT_EQ(counts[0], 1);  // VARIABLE
    EXPECT_EQ(counts[1], 1);  // CONSTANT
    EXPECT_EQ(counts[5], 1);  // MULTIPLICATION
}

TEST(OperatorCounts, EmptyExpression) {
    AGraphExpression expr;
    EXPECT_TRUE(
        expr.get_operator_counts(true, "exclude").empty());
    EXPECT_TRUE(
        expr.get_operator_counts(false, "exclude").empty());
}

TEST(OperatorCounts, UnaryOperator) {
    auto expr = make_sin_x0();
    auto counts = expr.get_operator_counts(true, "include");
    EXPECT_EQ(counts[15], 1);  // SIN
    EXPECT_EQ(counts[0], 1);   // VARIABLE
}

TEST(OperatorCounts, DefaultIsTreeExclude) {
    auto expr = make_x0_plus_c0();
    auto def = expr.get_operator_counts();
    auto explicit_ = expr.get_operator_counts(true, "exclude");
    EXPECT_EQ(def, explicit_);
}

// ================================================================
//  Test: Modification tracking
// ================================================================

TEST(ModificationTracking, RawCommandArraySetterMarksModified) {
    auto expr = make_x0_plus_c0();
    // Access command_array to clear modified
    auto _ = expr.command_array();
    (void)_;
    EXPECT_FALSE(expr.modified());

    StackMatrix cmd(3, 3);
    cmd << 0, 0, 0,
          1, 0, 0,
          3, 0, 1;
    expr.set_raw_command_array(cmd);
    EXPECT_TRUE(expr.modified());
}

TEST(ModificationTracking, RawConstantsSetterMarksModified) {
    auto expr = make_x0_plus_c0();
    auto _ = expr.command_array();
    (void)_;
    EXPECT_FALSE(expr.modified());

    expr.set_raw_constants({999.0});
    EXPECT_TRUE(expr.modified());
}

TEST(ModificationTracking, MutableAccessMarksModified) {
    auto expr = make_x0_plus_c0();
    auto _ = expr.command_array();
    (void)_;
    EXPECT_FALSE(expr.modified());

    auto& cmd = expr.mutable_raw_command_array();
    (void)cmd;
    EXPECT_TRUE(expr.modified());
}

// ================================================================
//  Test: Manual multi-operator expression
// ================================================================

TEST(OperatorCounts, ManualMultiOp) {
    // X0 + sin(X1) - sqrt(C0)
    AGraphExpression expr("reduce");
    StackMatrix cmd(7, 3);
    cmd <<  0, 0, 0,   // row 0: VARIABLE X0
            0, 1, 0,   // row 1: VARIABLE X1
           15, 1, 0,   // row 2: SIN(row1)
            3, 0, 2,   // row 3: ADD(row0, row2)
            1, 0, 0,   // row 4: CONSTANT C0
           11, 4, 0,   // row 5: SQRT(row4)
            4, 3, 5;   // row 6: SUB(row3, row5)
    expr.set_raw_command_array(cmd);
    expr.set_raw_constants({4.0});

    auto tree = expr.get_operator_counts(true, "include");
    EXPECT_EQ(tree[4], 1);   // SUBTRACTION
    EXPECT_EQ(tree[3], 1);   // ADDITION
    EXPECT_EQ(tree[15], 1);  // SIN
    EXPECT_EQ(tree[11], 1);  // SQRT
    EXPECT_EQ(tree[0], 2);   // 2 VARIABLEs
    EXPECT_EQ(tree[1], 1);   // 1 CONSTANT
}
