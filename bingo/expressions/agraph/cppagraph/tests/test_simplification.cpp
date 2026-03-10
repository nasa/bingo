/**
 * @file test_simplification.cpp
 * @brief Google Test suite for cppagraph stack reduction.
 *
 * Mirrors the test scenarios from
 * tests/unit/expressions/agraph/pyagraph/test_simplification.py.
 */

#include <gtest/gtest.h>

#include <vector>

#include "cppagraph/operators.h"
#include "cppagraph/simplification.h"

using namespace cppagraph;

// ================================================================
//  Helpers
// ================================================================

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

// Operator shorthands
static constexpr uint8_t VAR  = static_cast<uint8_t>(Op::VARIABLE);
static constexpr uint8_t CON  = static_cast<uint8_t>(Op::CONSTANT);
static constexpr uint8_t INT  = static_cast<uint8_t>(Op::INTEGER);
static constexpr uint8_t ADD  = static_cast<uint8_t>(Op::ADDITION);
static constexpr uint8_t MUL  = static_cast<uint8_t>(Op::MULTIPLICATION);
static constexpr uint8_t SSIN = static_cast<uint8_t>(Op::SIN);

// ================================================================
//  get_utilized_commands tests
// ================================================================

TEST(GetUtilizedCommands, SingleTerminal) {
    auto stack = make_stack({{VAR, 0, 0}});
    auto util = get_utilized_commands(stack);
    ASSERT_EQ(util.size(), 1u);
    EXPECT_TRUE(util[0]);
}

TEST(GetUtilizedCommands, UnusedCommand) {
    // Row 0: X0, Row 1: X1 (unused), Row 2: sin(X0)
    auto stack = make_stack({
        {VAR, 0, 0},
        {VAR, 1, 1},
        {SSIN, 0, 0},
    });
    auto util = get_utilized_commands(stack);
    EXPECT_TRUE(util[0]);
    EXPECT_FALSE(util[1]);
    EXPECT_TRUE(util[2]);
}

TEST(GetUtilizedCommands, BinaryOpMarksBothChildren) {
    auto stack = make_stack({
        {VAR, 0, 0},
        {CON, 0, 0},
        {ADD, 0, 1},
    });
    auto util = get_utilized_commands(stack);
    EXPECT_TRUE(util[0]);
    EXPECT_TRUE(util[1]);
    EXPECT_TRUE(util[2]);
}

TEST(GetUtilizedCommands, ChainOfDependencies) {
    // X0, X1, X0+X1, sin(X0+X1)
    auto stack = make_stack({
        {VAR, 0, 0},
        {VAR, 1, 1},
        {ADD, 0, 1},
        {SSIN, 2, 2},
    });
    auto util = get_utilized_commands(stack);
    for (int i = 0; i < 4; ++i)
        EXPECT_TRUE(util[i]);
}

TEST(GetUtilizedCommands, EmptyStack) {
    StackMatrix stack(0, 3);
    auto util = get_utilized_commands(stack);
    EXPECT_TRUE(util.empty());
}

// ================================================================
//  reduce tests
// ================================================================

TEST(Reduce, NoReductionNeeded) {
    auto stack = make_stack({
        {VAR, 0, 0},
        {CON, 0, 0},
        {ADD, 0, 1},
    });
    auto r = reduce(stack, {3.14}, {});

    ASSERT_EQ(r.stack.rows(), 3);
    EXPECT_EQ(r.stack(0, 0), VAR);
    EXPECT_EQ(r.stack(0, 1), 0);
    EXPECT_EQ(r.stack(1, 0), CON);
    EXPECT_EQ(r.stack(1, 1), 0);
    EXPECT_EQ(r.stack(2, 0), ADD);
    EXPECT_EQ(r.stack(2, 1), 0);
    EXPECT_EQ(r.stack(2, 2), 1);
    ASSERT_EQ(r.constants.size(), 1u);
    EXPECT_DOUBLE_EQ(r.constants[0], 3.14);
    EXPECT_TRUE(r.integers.empty());
}

TEST(Reduce, RemovesUnusedRow) {
    // Row 0: X0, Row 1: X1 (unused), Row 2: sin(X0)
    auto stack = make_stack({
        {VAR, 0, 0},
        {VAR, 1, 1},
        {SSIN, 0, 0},
    });
    auto r = reduce(stack, {}, {});

    ASSERT_EQ(r.stack.rows(), 2);
    EXPECT_EQ(r.stack(0, 0), VAR);
    EXPECT_EQ(r.stack(0, 1), 0);
    EXPECT_EQ(r.stack(1, 0), SSIN);
    EXPECT_EQ(r.stack(1, 1), 0);
}

TEST(Reduce, RemapsOperatorReferences) {
    // Row 0: X0, Row 1: X1 (unused), Row 2: X1, Row 3: X0 + X1
    auto stack = make_stack({
        {VAR, 0, 0},
        {VAR, 1, 1},   // unused
        {VAR, 1, 1},
        {ADD, 0, 2},
    });
    auto r = reduce(stack, {}, {});

    ASSERT_EQ(r.stack.rows(), 3);
    EXPECT_EQ(r.stack(0, 0), VAR);
    EXPECT_EQ(r.stack(0, 1), 0);
    EXPECT_EQ(r.stack(1, 0), VAR);
    EXPECT_EQ(r.stack(1, 1), 1);
    EXPECT_EQ(r.stack(2, 0), ADD);
    EXPECT_EQ(r.stack(2, 1), 0);
    EXPECT_EQ(r.stack(2, 2), 1);
}

TEST(Reduce, RenumbersConstantIndices) {
    // C0, C1 (unused), C2, C0*C2
    auto stack = make_stack({
        {CON, 0, 0},
        {CON, 1, 1},   // unused
        {CON, 2, 2},
        {MUL, 0, 2},
    });
    auto r = reduce(stack, {1.0, 99.0, 2.0}, {});

    ASSERT_EQ(r.stack.rows(), 3);
    EXPECT_EQ(r.stack(0, 0), CON);
    EXPECT_EQ(r.stack(0, 1), 0);
    EXPECT_EQ(r.stack(1, 0), CON);
    EXPECT_EQ(r.stack(1, 1), 1);
    EXPECT_EQ(r.stack(2, 0), MUL);
    EXPECT_EQ(r.stack(2, 1), 0);
    EXPECT_EQ(r.stack(2, 2), 1);
    ASSERT_EQ(r.constants.size(), 2u);
    EXPECT_DOUBLE_EQ(r.constants[0], 1.0);
    EXPECT_DOUBLE_EQ(r.constants[1], 2.0);
}

TEST(Reduce, RenumbersIntegerIndices) {
    // I0, I1 (unused), I2, I0+I2
    auto stack = make_stack({
        {INT, 0, 0},
        {INT, 1, 1},   // unused
        {INT, 2, 2},
        {ADD, 0, 2},
    });
    auto r = reduce(stack, {}, {10, 99, 20});

    ASSERT_EQ(r.stack.rows(), 3);
    EXPECT_EQ(r.stack(0, 0), INT);
    EXPECT_EQ(r.stack(0, 1), 0);
    EXPECT_EQ(r.stack(1, 0), INT);
    EXPECT_EQ(r.stack(1, 1), 1);
    EXPECT_EQ(r.stack(2, 0), ADD);
    EXPECT_EQ(r.stack(2, 1), 0);
    EXPECT_EQ(r.stack(2, 2), 1);
    ASSERT_EQ(r.integers.size(), 2u);
    EXPECT_EQ(r.integers[0], 10);
    EXPECT_EQ(r.integers[1], 20);
}

TEST(Reduce, EmptyStack) {
    StackMatrix stack(0, 3);
    auto r = reduce(stack, {}, {});
    EXPECT_EQ(r.stack.rows(), 0);
    EXPECT_TRUE(r.constants.empty());
    EXPECT_TRUE(r.integers.empty());
    EXPECT_TRUE(r.constant_mapping.empty());
}

// ================================================================
//  Constant mapping tests
// ================================================================

TEST(ReduceConstantMapping, IdentityMappingSingleConstant) {
    auto stack = make_stack({
        {VAR, 0, 0},
        {CON, 0, 0},
        {ADD, 0, 1},
    });
    auto r = reduce(stack, {3.14}, {});
    ASSERT_EQ(r.constant_mapping.size(), 1u);
    EXPECT_EQ(r.constant_mapping[0], 0);
}

TEST(ReduceConstantMapping, SkipsUnusedConstant) {
    // C0 used, C1 unused, C2 used → mapping is {0, 2}
    auto stack = make_stack({
        {CON, 0, 0},
        {CON, 1, 1},   // unused
        {CON, 2, 2},
        {MUL, 0, 2},
    });
    auto r = reduce(stack, {1.0, 99.0, 2.0}, {});
    ASSERT_EQ(r.constant_mapping.size(), 2u);
    EXPECT_EQ(r.constant_mapping[0], 0);
    EXPECT_EQ(r.constant_mapping[1], 2);
}

TEST(ReduceConstantMapping, PreservesOrder) {
    // C2 first, then C0 → mapping is {2, 0}
    auto stack = make_stack({
        {CON, 2, 2},
        {CON, 0, 0},
        {ADD, 0, 1},
    });
    auto r = reduce(stack, {10.0, 20.0, 30.0}, {});
    ASSERT_EQ(r.constant_mapping.size(), 2u);
    EXPECT_EQ(r.constant_mapping[0], 2);
    EXPECT_EQ(r.constant_mapping[1], 0);
}

TEST(ReduceConstantMapping, EmptyMappingNoConstants) {
    auto stack = make_stack({
        {VAR, 0, 0},
        {SSIN, 0, 0},
    });
    auto r = reduce(stack, {}, {});
    EXPECT_TRUE(r.constant_mapping.empty());
}

TEST(ReduceConstantMapping, MappingValuesIndexIntoRawConstants) {
    auto stack = make_stack({
        {CON, 0, 0},
        {CON, 1, 1},   // unused
        {CON, 2, 2},
        {MUL, 0, 2},
    });
    std::vector<double> raw_consts = {10.0, 99.0, 20.0};
    auto r = reduce(stack, raw_consts, {});
    for (size_t i = 0; i < r.constant_mapping.size(); ++i) {
        EXPECT_DOUBLE_EQ(r.constants[i], raw_consts[r.constant_mapping[i]]);
    }
}
