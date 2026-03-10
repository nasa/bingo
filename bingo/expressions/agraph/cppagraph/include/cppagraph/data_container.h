/**
 * @file data_container.h
 * @brief Lightweight container for training/test data.
 *
 * Mirrors bingo.expressions.agraph.pyagraph.data_container.DataContainer.
 * Uses Eigen RowMajor matrices for natural interop with NumPy (C-order).
 */

#pragma once

#include <Eigen/Core>
#include <stdexcept>
#include <string>

namespace cppagraph {

/// Row-major double matrix — matches NumPy's default C-order layout.
using RowMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/**
 * @class DataContainer
 * @brief Container for feature data (x) and target data (y).
 *
 * Provides validation that the number of rows match, and supports
 * row-slicing via an index vector.
 */
class DataContainer {
public:
    /**
     * Construct a DataContainer from x and y matrices.
     *
     * @param x Feature data (M rows, D columns).
     * @param y Target data  (M rows, K columns).
     * @throws std::invalid_argument if row counts don't match.
     */
    DataContainer(RowMatrixXd x, RowMatrixXd y)
        : x_(std::move(x)), y_(std::move(y))
    {
        if (x_.rows() != y_.rows()) {
            throw std::invalid_argument(
                "Number of rows in x (" + std::to_string(x_.rows()) +
                ") and y (" + std::to_string(y_.rows()) + ") must match.");
        }
    }

    /// Number of data points (rows).
    Eigen::Index size() const { return x_.rows(); }

    /// Feature data (M x D).
    const RowMatrixXd& x() const { return x_; }

    /// Target data (M x K).
    const RowMatrixXd& y() const { return y_; }

    /**
     * Row-slice: return a new DataContainer with the specified row indices.
     *
     * @param indices  Vector of row indices to select.
     * @return A new DataContainer containing only those rows.
     */
    DataContainer slice(const Eigen::VectorXi& indices) const {
        RowMatrixXd sx(indices.size(), x_.cols());
        RowMatrixXd sy(indices.size(), y_.cols());
        for (Eigen::Index i = 0; i < indices.size(); ++i) {
            sx.row(i) = x_.row(indices(i));
            sy.row(i) = y_.row(indices(i));
        }
        return DataContainer(std::move(sx), std::move(sy));
    }

private:
    RowMatrixXd x_;
    RowMatrixXd y_;
};

}  // namespace cppagraph
