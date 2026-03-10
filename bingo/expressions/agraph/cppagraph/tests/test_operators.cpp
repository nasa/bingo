/**
 * @file test_operators.cpp
 * @brief Google Test suite for cppagraph operator definitions.
 *
 * Verifies that operator IDs, property arrays, runtime sets, and
 * name tables all match the pyagraph reference implementation.
 */

#include <gtest/gtest.h>
#include "cppagraph/operators.h"

#include <algorithm>

using namespace cppagraph;

// ---- Operator ID value tests ----

TEST(OperatorIds, TerminalValues) {
    EXPECT_EQ(static_cast<uint8_t>(Op::VARIABLE), 0);
    EXPECT_EQ(static_cast<uint8_t>(Op::CONSTANT), 1);
    EXPECT_EQ(static_cast<uint8_t>(Op::INTEGER),  2);
}

TEST(OperatorIds, ArithmeticValues) {
    EXPECT_EQ(static_cast<uint8_t>(Op::ADDITION),       3);
    EXPECT_EQ(static_cast<uint8_t>(Op::SUBTRACTION),    4);
    EXPECT_EQ(static_cast<uint8_t>(Op::MULTIPLICATION), 5);
    EXPECT_EQ(static_cast<uint8_t>(Op::DIVISION),       6);
}

TEST(OperatorIds, PowerValues) {
    EXPECT_EQ(static_cast<uint8_t>(Op::POWER),      7);
    EXPECT_EQ(static_cast<uint8_t>(Op::SAFE_POWER), 8);
    EXPECT_EQ(static_cast<uint8_t>(Op::SQUARE),     9);
    EXPECT_EQ(static_cast<uint8_t>(Op::CUBE),       10);
    EXPECT_EQ(static_cast<uint8_t>(Op::SQRT),       11);
}

TEST(OperatorIds, MiscAndExpValues) {
    EXPECT_EQ(static_cast<uint8_t>(Op::ABS),         12);
    EXPECT_EQ(static_cast<uint8_t>(Op::EXPONENTIAL), 13);
    EXPECT_EQ(static_cast<uint8_t>(Op::LOGARITHM),   14);
}

TEST(OperatorIds, TrigValues) {
    EXPECT_EQ(static_cast<uint8_t>(Op::SIN),    15);
    EXPECT_EQ(static_cast<uint8_t>(Op::COS),    16);
    EXPECT_EQ(static_cast<uint8_t>(Op::TAN),    17);
    EXPECT_EQ(static_cast<uint8_t>(Op::ARCSIN), 18);
    EXPECT_EQ(static_cast<uint8_t>(Op::ARCCOS), 19);
    EXPECT_EQ(static_cast<uint8_t>(Op::ARCTAN), 20);
}

TEST(OperatorIds, HyperbolicValues) {
    EXPECT_EQ(static_cast<uint8_t>(Op::SINH), 21);
    EXPECT_EQ(static_cast<uint8_t>(Op::COSH), 22);
    EXPECT_EQ(static_cast<uint8_t>(Op::TANH), 23);
}

TEST(OperatorIds, NumOps) {
    EXPECT_EQ(NUM_OPS, 24u);
}

// ---- Compile-time boolean array tests ----

TEST(IsTerminal, OnlyTerminalsAreTrue) {
    for (std::size_t i = 0; i < NUM_OPS; ++i) {
        bool expected = (i == 0 || i == 1 || i == 2);
        EXPECT_EQ(IS_TERMINAL[i], expected) << "Mismatch at index " << i;
    }
}

TEST(IsArity2, OnlyBinaryOpsAreTrue) {
    const std::unordered_set<std::size_t> expected_true{3, 4, 5, 6, 7, 8};
    for (std::size_t i = 0; i < NUM_OPS; ++i) {
        bool expected = expected_true.count(i) > 0;
        EXPECT_EQ(IS_ARITY_2[i], expected) << "Mismatch at index " << i;
    }
}

// ---- Runtime set tests ----

TEST(TerminalIds, ContainsExactSet) {
    const auto& ids = terminal_ids();
    EXPECT_EQ(ids.size(), 3u);
    EXPECT_TRUE(ids.count(0));  // VARIABLE
    EXPECT_TRUE(ids.count(1));  // CONSTANT
    EXPECT_TRUE(ids.count(2));  // INTEGER
}

TEST(Arity2Ids, ContainsExactSet) {
    const auto& ids = arity_2_ids();
    EXPECT_EQ(ids.size(), 6u);
    EXPECT_TRUE(ids.count(3));  // ADDITION
    EXPECT_TRUE(ids.count(4));  // SUBTRACTION
    EXPECT_TRUE(ids.count(5));  // MULTIPLICATION
    EXPECT_TRUE(ids.count(6));  // DIVISION
    EXPECT_TRUE(ids.count(7));  // POWER
    EXPECT_TRUE(ids.count(8));  // SAFE_POWER
}

// ---- Operator names tests ----

TEST(OperatorNames, AllOpsHaveNames) {
    const auto& names = operator_names();
    EXPECT_EQ(names.size(), NUM_OPS);
    for (std::size_t i = 0; i < NUM_OPS; ++i) {
        EXPECT_TRUE(names.count(static_cast<uint8_t>(i)))
            << "Missing names for operator " << i;
    }
}

TEST(OperatorNames, SpotCheckVariable) {
    const auto& names = operator_names();
    const auto& var_names = names.at(static_cast<uint8_t>(Op::VARIABLE));
    ASSERT_EQ(var_names.size(), 2u);
    EXPECT_EQ(var_names[0], "load");
    EXPECT_EQ(var_names[1], "x");
}

TEST(OperatorNames, SpotCheckAddition) {
    const auto& names = operator_names();
    const auto& add_names = names.at(static_cast<uint8_t>(Op::ADDITION));
    ASSERT_EQ(add_names.size(), 3u);
    EXPECT_EQ(add_names[0], "add");
    EXPECT_EQ(add_names[1], "addition");
    EXPECT_EQ(add_names[2], "+");
}

TEST(OperatorNames, SpotCheckTanh) {
    const auto& names = operator_names();
    const auto& tanh_names = names.at(static_cast<uint8_t>(Op::TANH));
    ASSERT_EQ(tanh_names.size(), 2u);
    EXPECT_EQ(tanh_names[0], "tangenth");
    EXPECT_EQ(tanh_names[1], "tanh");
}
