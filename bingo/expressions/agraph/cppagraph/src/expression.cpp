/**
 * @file expression.cpp
 * @brief AGraphExpression implementation.
 *
 * Mirrors bingo.expressions.agraph.pyagraph.expression.AGraphExpression.
 */

#include "cppagraph/expression.h"

#include "cppagraph/cached_evaluation.h"
#include "cppagraph/cas_simplify.h"
#include "cppagraph/evaluation.h"
#include "cppagraph/operators.h"
#include "cppagraph/simplification.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace cppagraph {

// ================================================================
//  Scoring metrics
// ================================================================

double mean_absolute_error(const Eigen::VectorXd& r) {
    return r.cwiseAbs().mean();
}

double mean_squared_error(const Eigen::VectorXd& r) {
    return r.squaredNorm() / static_cast<double>(r.size());
}

double root_mean_squared_error(const Eigen::VectorXd& r) {
    return std::sqrt(mean_squared_error(r));
}

double bic_score(const Eigen::VectorXd& r, int n_constants) {
    const double n = static_cast<double>(r.size());
    const double k = static_cast<double>(n_constants + 1);
    double mse = r.squaredNorm() / n;
    if (mse <= 0.0) mse = std::numeric_limits<double>::min();
    const double log_like =
        -n / 2.0 * std::log(mse) - n / 2.0 -
        n / 2.0 * std::log(2.0 * M_PI);
    return k * std::log(n) - 2.0 * log_like;
}

double laplace_nmll_score(const Eigen::VectorXd& r, int n_constants) {
    const double n = static_cast<double>(r.size());
    const double k = static_cast<double>(n_constants + 1);
    const double b = 1.0 / std::sqrt(n);
    double mse = r.squaredNorm() / n;
    if (mse <= 0.0) mse = std::numeric_limits<double>::min();
    const double log_like =
        -n / 2.0 * std::log(mse) - n / 2.0 -
        n / 2.0 * std::log(2.0 * M_PI);
    return (1.0 - b) * log_like + std::log(b) / 2.0 * k;
}

// ================================================================
//  Construction
// ================================================================

AGraphExpression::AGraphExpression(const std::string& simplification,
                                   bool propagate_constants)
    : simplification_(simplification),
      propagate_constants_(propagate_constants),
      raw_command_array_(0, 3),
      command_array_(0, 3),
      is_fitted_(true),
      modified_(false)
{
    if (simplification != "reduce" && simplification != "cas") {
        throw std::invalid_argument(
            "simplification must be \"reduce\" or \"cas\", got \"" +
            simplification + "\"");
    }
}

// ================================================================
//  Raw layer
// ================================================================

const StackMatrix& AGraphExpression::raw_command_array() const {
    return raw_command_array_;
}

void AGraphExpression::set_raw_command_array(StackMatrix cmd) {
    raw_command_array_ = std::move(cmd);
    notify_modification();
}

StackMatrix& AGraphExpression::mutable_raw_command_array() {
    notify_modification();
    return raw_command_array_;
}

const std::vector<double>& AGraphExpression::raw_constants() const {
    return raw_constants_;
}

void AGraphExpression::set_raw_constants(std::vector<double> c) {
    raw_constants_ = std::move(c);
    notify_modification();
}

const std::vector<int>& AGraphExpression::raw_integers() const {
    return raw_integers_;
}

void AGraphExpression::set_raw_integers(std::vector<int> i) {
    raw_integers_ = std::move(i);
    notify_modification();
}

// ================================================================
//  Simplified layer
// ================================================================

const StackMatrix& AGraphExpression::command_array() {
    if (modified_) update();
    return command_array_;
}

const std::vector<double>& AGraphExpression::constants() {
    if (modified_) update();
    return constants_;
}

void AGraphExpression::set_constants(std::vector<double> c) {
    if (modified_) update();
    constants_ = std::move(c);
    // Propagate back to raw if enabled.
    if (propagate_constants_ && !constant_mapping_.empty()) {
        for (std::size_t i = 0; i < constant_mapping_.size(); ++i) {
            int raw_idx = constant_mapping_[i];
            if (i < constants_.size() &&
                raw_idx >= 0 &&
                static_cast<std::size_t>(raw_idx) < raw_constants_.size()) {
                raw_constants_[raw_idx] = constants_[i];
            }
        }
    }
}

const std::vector<int>& AGraphExpression::integers() {
    if (modified_) update();
    return integers_;
}

const std::vector<int>& AGraphExpression::constant_mapping() {
    if (modified_) update();
    return constant_mapping_;
}

Eigen::Index AGraphExpression::complexity() {
    if (modified_) update();
    return command_array_.rows();
}

Eigen::Index AGraphExpression::tree_complexity() {
    if (modified_) update();
    const auto& cmd = command_array_;
    if (cmd.rows() == 0) return 0;
    Eigen::Index count = 0;
    std::vector<int> stk;
    stk.push_back(static_cast<int>(cmd.rows() - 1));
    while (!stk.empty()) {
        int idx = stk.back();
        stk.pop_back();
        ++count;
        uint8_t node = cmd(idx, 0);
        if (!IS_TERMINAL[node]) {
            stk.push_back(static_cast<int>(cmd(idx, 1)));
            if (IS_ARITY_2[node]) {
                stk.push_back(static_cast<int>(cmd(idx, 2)));
            }
        }
    }
    return count;
}

// ================================================================
//  Simplification control
// ================================================================

const std::string& AGraphExpression::simplification() const {
    return simplification_;
}

bool AGraphExpression::propagate_constants() const {
    return propagate_constants_;
}

void AGraphExpression::set_propagate_constants(bool v) {
    propagate_constants_ = v;
}

// ================================================================
//  Evaluation
// ================================================================

RowMatrixXd AGraphExpression::evaluate(const RowMatrixXd& x) {
    if (modified_) update();
    try {
        return cppagraph::evaluate(
            command_array_, x, constants_, integers_);
    } catch (...) {
        return RowMatrixXd::Constant(
            x.rows(), 1, std::numeric_limits<double>::quiet_NaN());
    }
}

std::pair<RowMatrixXd, RowMatrixXd>
AGraphExpression::evaluate_with_x_gradient(const RowMatrixXd& x) {
    if (modified_) update();
    try {
        return evaluate_with_derivative(
            command_array_, x, constants_, integers_, true);
    } catch (...) {
        RowMatrixXd nan_val = RowMatrixXd::Constant(
            x.rows(), x.cols(),
            std::numeric_limits<double>::quiet_NaN());
        return {nan_val, RowMatrixXd(nan_val)};
    }
}

std::pair<RowMatrixXd, RowMatrixXd>
AGraphExpression::evaluate_with_const_gradient(const RowMatrixXd& x) {
    if (modified_) update();
    try {
        return evaluate_with_derivative(
            command_array_, x, constants_, integers_, false);
    } catch (...) {
        Eigen::Index nc =
            static_cast<Eigen::Index>(constants_.size());
        RowMatrixXd nan_val = RowMatrixXd::Constant(
            x.rows(), nc,
            std::numeric_limits<double>::quiet_NaN());
        return {nan_val, RowMatrixXd(nan_val)};
    }
}

// ================================================================
//  sklearn-like interface
// ================================================================

Eigen::VectorXd AGraphExpression::predict(const RowMatrixXd& X) {
    return evaluate(X).col(0);
}

void AGraphExpression::fit(const RowMatrixXd& X,
                           const Eigen::VectorXd& y,
                           int max_iter, double tol) {
    if (modified_) update();
    is_fitted_ = true;

    if (constants_.empty()) return;

    const Eigen::Index m = X.rows();
    const Eigen::Index n =
        static_cast<Eigen::Index>(constants_.size());

    CachedEvaluator cached(command_array_, X, integers_);

    Eigen::VectorXd params(n);
    for (Eigen::Index i = 0; i < n; ++i)
        params(i) = constants_[i];

    double lambda = 1e-3;
    constexpr double lambda_up = 10.0;
    constexpr double lambda_down = 0.1;

    try {
        for (int iter = 0; iter < max_iter; ++iter) {
            std::vector<double> c(
                params.data(), params.data() + params.size());

            auto [f_val, jac] =
                cached.forward_eval_with_const_derivative(c);
            Eigen::VectorXd r = f_val.col(0) - y.head(m);

            if (!r.allFinite() || !jac.allFinite()) break;

            // Normal equations
            Eigen::MatrixXd JtJ = jac.transpose() * jac;
            Eigen::VectorXd Jtr = jac.transpose() * r;

            // Marquardt damping (diagonal scaling)
            Eigen::VectorXd diag_JtJ = JtJ.diagonal();
            for (Eigen::Index i = 0; i < n; ++i)
                if (diag_JtJ(i) < 1e-12)
                    diag_JtJ(i) = 1e-12;

            Eigen::MatrixXd H = JtJ;
            H.diagonal() += lambda * diag_JtJ;

            Eigen::VectorXd dp = H.ldlt().solve(-Jtr);
            if (!dp.allFinite()) break;

            // Trial step
            Eigen::VectorXd new_params = params + dp;
            std::vector<double> c_new(
                new_params.data(),
                new_params.data() + new_params.size());
            RowMatrixXd f_new = cached.forward_eval(c_new);
            Eigen::VectorXd r_new = f_new.col(0) - y.head(m);

            double cost_old = r.squaredNorm();
            double cost_new = r_new.allFinite()
                ? r_new.squaredNorm()
                : std::numeric_limits<double>::infinity();

            if (cost_new < cost_old) {
                params = new_params;
                lambda *= lambda_down;
                // Convergence checks
                if (dp.norm() < tol * (params.norm() + tol))
                    break;
                if (std::abs(cost_old - cost_new) <
                    tol * cost_old)
                    break;
            } else {
                lambda *= lambda_up;
            }
        }
    } catch (...) {
        // Don't crash on bad fits — keep current params.
    }

    constants_ = std::vector<double>(
        params.data(), params.data() + params.size());

    // Propagate if enabled
    if (propagate_constants_ && !constant_mapping_.empty()) {
        for (std::size_t i = 0; i < constant_mapping_.size(); ++i) {
            int raw_idx = constant_mapping_[i];
            if (i < constants_.size() &&
                raw_idx >= 0 &&
                static_cast<std::size_t>(raw_idx) <
                    raw_constants_.size()) {
                raw_constants_[raw_idx] = constants_[i];
            }
        }
    }
}

double AGraphExpression::score(const RowMatrixXd& X,
                               const Eigen::VectorXd& y,
                               const std::string& metric) {
    Eigen::VectorXd residuals = predict(X) - y;
    int nc = static_cast<int>(constants().size());

    if (metric == "bic")
        return bic_score(residuals, nc);
    if (metric == "laplace_nmll")
        return laplace_nmll_score(residuals, nc);
    if (metric == "mae")
        return mean_absolute_error(residuals);
    if (metric == "rmse")
        return root_mean_squared_error(residuals);
    return mean_squared_error(residuals);
}

bool AGraphExpression::is_fitted() {
    if (modified_) update();
    return is_fitted_;
}

// ================================================================
//  Utility
// ================================================================

std::vector<bool> AGraphExpression::get_utilized_commands() const {
    return cppagraph::get_utilized_commands(raw_command_array_);
}

void AGraphExpression::promote_simplification() {
    if (modified_) update();
    raw_command_array_ = command_array_;
    raw_constants_ = constants_;
    raw_integers_ = integers_;
    constant_mapping_.resize(constants_.size());
    std::iota(constant_mapping_.begin(), constant_mapping_.end(), 0);
    modified_ = false;
}

std::map<int, int> AGraphExpression::get_operator_counts(
        bool tree, const std::string& terminals) {
    if (modified_) update();
    if (!tree)
        return dag_operator_counts(command_array_, terminals);
    return tree_operator_counts(command_array_, terminals);
}

int AGraphExpression::distance(const AGraphExpression& other) const {
    const auto& a = raw_command_array_;
    const auto& b = other.raw_command_array_;
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        return static_cast<int>(
            a.rows() * a.cols() + b.rows() * b.cols());
    }
    int count = 0;
    for (Eigen::Index r = 0; r < a.rows(); ++r)
        for (Eigen::Index c = 0; c < a.cols(); ++c)
            if (a(r, c) != b(r, c))
                ++count;
    return count;
}

AGraphExpression AGraphExpression::copy() const {
    AGraphExpression out(simplification_, propagate_constants_);
    out.raw_command_array_ = raw_command_array_;
    out.raw_constants_ = raw_constants_;
    out.raw_integers_ = raw_integers_;
    out.command_array_ = command_array_;
    out.constants_ = constants_;
    out.integers_ = integers_;
    out.constant_mapping_ = constant_mapping_;
    out.is_fitted_ = is_fitted_;
    out.modified_ = modified_;
    out.hash_ = std::nullopt;  // don't carry hash across copies
    return out;
}

// ================================================================
//  Hash / equality
// ================================================================

std::size_t AGraphExpression::hash() {
    if (modified_) {
        update();
    }
    if (!hash_.has_value()) {
        // Hash the simplified command array (matches Python's
        // hash(tuple(map(tuple, command_array)))).
        std::size_t h = 0;
        for (Eigen::Index r = 0; r < command_array_.rows(); ++r) {
            std::size_t row_h = 0;
            for (Eigen::Index c = 0; c < command_array_.cols(); ++c) {
                row_h ^= std::hash<uint8_t>{}(command_array_(r, c)) +
                          0x9e3779b9 + (row_h << 6) + (row_h >> 2);
            }
            h ^= row_h + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        hash_ = h;
    }
    return hash_.value();
}

bool AGraphExpression::equals(AGraphExpression& other) {
    return hash() == other.hash();
}

// ================================================================
//  State
// ================================================================

bool AGraphExpression::modified() const {
    return modified_;
}

// ================================================================
//  Internal
// ================================================================

void AGraphExpression::notify_modification() {
    modified_ = true;
    hash_ = std::nullopt;
    is_fitted_ = false;
}

void AGraphExpression::update() {
    if (simplification_ == "cas") {
        auto result = cas_simplify(
            raw_command_array_, raw_constants_, raw_integers_);
        command_array_ = std::move(result.stack);
        constants_ = std::move(result.constants);
        integers_ = std::move(result.integers);
        constant_mapping_ = std::move(result.constant_mapping);
    } else {
        auto result = reduce(
            raw_command_array_, raw_constants_, raw_integers_);
        command_array_ = std::move(result.stack);
        constants_ = std::move(result.constants);
        integers_ = std::move(result.integers);
        constant_mapping_ = std::move(result.constant_mapping);
    }
    is_fitted_ = constants_.empty();
    modified_ = false;
}

// ================================================================
//  Operator counts — static helpers
// ================================================================

std::map<int, int> AGraphExpression::dag_operator_counts(
        const StackMatrix& cmd, const std::string& terminals) {
    std::map<int, int> counts;
    bool exclude = (terminals == "exclude");
    bool combine = (terminals == "combine");

    for (Eigen::Index i = 0; i < cmd.rows(); ++i) {
        int node = static_cast<int>(cmd(i, 0));
        if (IS_TERMINAL[node]) {
            if (exclude) continue;
            int key = combine
                ? static_cast<int>(Op::VARIABLE)
                : node;
            counts[key]++;
        } else {
            counts[node]++;
        }
    }
    return counts;
}

std::map<int, int> AGraphExpression::tree_operator_counts(
        const StackMatrix& cmd, const std::string& terminals) {
    if (cmd.rows() == 0) return {};

    bool exclude = (terminals == "exclude");
    bool combine = (terminals == "combine");

    std::map<int, int> counts;
    std::vector<int> stk;
    stk.push_back(static_cast<int>(cmd.rows() - 1));

    while (!stk.empty()) {
        int idx = stk.back();
        stk.pop_back();

        uint8_t node   = cmd(idx, 0);
        uint8_t param1 = cmd(idx, 1);
        uint8_t param2 = cmd(idx, 2);

        if (IS_TERMINAL[node]) {
            if (exclude) continue;
            int key = combine
                ? static_cast<int>(Op::VARIABLE)
                : static_cast<int>(node);
            counts[key]++;
            continue;
        }

        counts[static_cast<int>(node)]++;
        stk.push_back(static_cast<int>(param1));
        if (IS_ARITY_2[node]) {
            stk.push_back(static_cast<int>(param2));
        }
    }
    return counts;
}

}  // namespace cppagraph
