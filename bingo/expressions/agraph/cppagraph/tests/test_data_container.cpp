/**
 * @file test_data_container.cpp
 * @brief Google Test suite for cppagraph DataContainer.
 *
 * Verifies construction, validation, properties, and row-slicing
 * behaviour matching the pyagraph reference implementation.
 */

#include <gtest/gtest.h>
#include "cppagraph/data_container.h"

using namespace cppagraph;

// ---- Construction tests ----

TEST(DataContainer, ConstructFromMatrices) {
    RowMatrixXd x(3, 2);
    x << 1, 2,
         3, 4,
         5, 6;
    RowMatrixXd y(3, 1);
    y << 10, 20, 30;

    DataContainer dc(x, y);
    EXPECT_EQ(dc.size(), 3);
    EXPECT_EQ(dc.x().rows(), 3);
    EXPECT_EQ(dc.x().cols(), 2);
    EXPECT_EQ(dc.y().rows(), 3);
    EXPECT_EQ(dc.y().cols(), 1);
}

TEST(DataContainer, RowMismatchThrows) {
    RowMatrixXd x(3, 2);
    x << 1, 2, 3, 4, 5, 6;
    RowMatrixXd y(2, 1);
    y << 10, 20;

    EXPECT_THROW(DataContainer(x, y), std::invalid_argument);
}

TEST(DataContainer, EmptyDataAllowed) {
    RowMatrixXd x(0, 2);
    RowMatrixXd y(0, 1);
    DataContainer dc(x, y);
    EXPECT_EQ(dc.size(), 0);
}

// ---- Property tests ----

TEST(DataContainer, XValuesPreserved) {
    RowMatrixXd x(2, 3);
    x << 1.5, 2.5, 3.5,
         4.5, 5.5, 6.5;
    RowMatrixXd y(2, 1);
    y << 0, 0;

    DataContainer dc(x, y);
    EXPECT_DOUBLE_EQ(dc.x()(0, 0), 1.5);
    EXPECT_DOUBLE_EQ(dc.x()(0, 2), 3.5);
    EXPECT_DOUBLE_EQ(dc.x()(1, 1), 5.5);
}

TEST(DataContainer, YValuesPreserved) {
    RowMatrixXd x(2, 1);
    x << 0, 0;
    RowMatrixXd y(2, 2);
    y << 7.0, 8.0,
         9.0, 10.0;

    DataContainer dc(x, y);
    EXPECT_DOUBLE_EQ(dc.y()(0, 0), 7.0);
    EXPECT_DOUBLE_EQ(dc.y()(1, 1), 10.0);
}

// ---- Slice tests ----

TEST(DataContainer, SliceByIndices) {
    RowMatrixXd x(5, 2);
    x << 1, 2,
         3, 4,
         5, 6,
         7, 8,
         9, 10;
    RowMatrixXd y(5, 1);
    y << 10, 20, 30, 40, 50;

    DataContainer dc(x, y);

    Eigen::VectorXi idx(3);
    idx << 0, 2, 4;
    auto sliced = dc.slice(idx);

    EXPECT_EQ(sliced.size(), 3);
    EXPECT_DOUBLE_EQ(sliced.x()(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(sliced.x()(1, 0), 5.0);
    EXPECT_DOUBLE_EQ(sliced.x()(2, 0), 9.0);
    EXPECT_DOUBLE_EQ(sliced.y()(0, 0), 10.0);
    EXPECT_DOUBLE_EQ(sliced.y()(1, 0), 30.0);
    EXPECT_DOUBLE_EQ(sliced.y()(2, 0), 50.0);
}

TEST(DataContainer, SliceSingleRow) {
    RowMatrixXd x(3, 2);
    x << 1, 2, 3, 4, 5, 6;
    RowMatrixXd y(3, 1);
    y << 10, 20, 30;

    DataContainer dc(x, y);

    Eigen::VectorXi idx(1);
    idx << 1;
    auto sliced = dc.slice(idx);

    EXPECT_EQ(sliced.size(), 1);
    EXPECT_DOUBLE_EQ(sliced.x()(0, 0), 3.0);
    EXPECT_DOUBLE_EQ(sliced.x()(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(sliced.y()(0, 0), 20.0);
}

TEST(DataContainer, SliceEmptyIndices) {
    RowMatrixXd x(3, 2);
    x << 1, 2, 3, 4, 5, 6;
    RowMatrixXd y(3, 1);
    y << 10, 20, 30;

    DataContainer dc(x, y);

    Eigen::VectorXi idx(0);
    auto sliced = dc.slice(idx);
    EXPECT_EQ(sliced.size(), 0);
}
