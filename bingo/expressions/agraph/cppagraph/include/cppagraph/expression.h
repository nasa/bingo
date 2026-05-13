/**
 * @file expression.h
 * @brief AGraphExpression — dual-layer acyclic-graph expression class.
 *
 * Mirrors bingo.expressions.agraph.pyagraph.expression.AGraphExpression.
 *
 * The class maintains two layers:
 *   - **Raw** (GA-facing): the command stack that genetic operators mutate.
 *   - **Simplified** (evaluation-facing): derived lazily via CAS or
 *     dead-code elimination, cached until the raw layer changes.
 *
 * Provides evaluation, reverse-mode derivatives, Levenberg-Marquardt
 * fitting, scoring, operator counts, hash/equality, and deep copy.
 */

#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "cppagraph/data_container.h"  // RowMatrixXd
#include "cppagraph/evaluation.h"      // StackMatrix

namespace cppagraph {

// ================================================================
//  Scoring metrics
// ================================================================

double mean_absolute_error(const Eigen::VectorXd& residuals);
double mean_squared_error(const Eigen::VectorXd& residuals);
double root_mean_squared_error(const Eigen::VectorXd& residuals);
double bic_score(const Eigen::VectorXd& residuals, int n_constants);
double laplace_nmll_score(const Eigen::VectorXd& residuals, int n_constants);

// ================================================================
//  AGraphExpression
// ================================================================

class AGraphExpression {
public:
    /**
     * Construct an empty expression.
     *
     * @param simplification  "reduce" or "cas" (default "cas").
     * @param propagate_constants  Whether setting simplified constants
     *        also updates raw constants.  Default false.
     */
    explicit AGraphExpression(
        const std::string& simplification = "cas",
        bool propagate_constants = false);

    // ---- Raw (GA-facing) layer ------------------------------------ //

    /** Read-only access to the raw command array. */
    const StackMatrix& raw_command_array() const;

    /** Replace the raw command array and mark as modified. */
    void set_raw_command_array(StackMatrix cmd);

    /** Mutable reference — marks the expression as modified. */
    StackMatrix& mutable_raw_command_array();

    /** Raw constant values. */
    const std::vector<double>& raw_constants() const;

    /** Replace raw constants and mark as modified. */
    void set_raw_constants(std::vector<double> c);

    /** Raw integer values. */
    const std::vector<int>& raw_integers() const;

    /** Replace raw integers and mark as modified. */
    void set_raw_integers(std::vector<int> i);

    // ---- Simplified (evaluation-facing) layer ---------------------- //

    /** Simplified command array (lazy update). */
    const StackMatrix& command_array();

    /** Simplified constants (lazy update). */
    const std::vector<double>& constants();

    /** Set simplified constants — used by fit(), does NOT re-simplify. */
    void set_constants(std::vector<double> c);

    /** Simplified integers (lazy update). */
    const std::vector<int>& integers();

    /** constant_mapping[simplified_idx] → raw_constants index. */
    const std::vector<int>& constant_mapping();

    /** Number of commands in the simplified stack. */
    Eigen::Index complexity();

    /** Tree-based node count (counts shared sub-expressions multiple times). */
    Eigen::Index tree_complexity();

    // ---- Simplification control ------------------------------------ //

    const std::string& simplification() const;
    bool propagate_constants() const;
    void set_propagate_constants(bool v);

    // ---- Evaluation ------------------------------------------------ //

    /** Evaluate f(x).  Returns NaN matrix on arithmetic error. */
    RowMatrixXd evaluate(const RowMatrixXd& x);

    /** (f(x), df/dx).  Returns NaN matrices on error. */
    std::pair<RowMatrixXd, RowMatrixXd>
    evaluate_with_x_gradient(const RowMatrixXd& x);

    /** (f(x), df/dc).  Returns NaN matrices on error. */
    std::pair<RowMatrixXd, RowMatrixXd>
    evaluate_with_const_gradient(const RowMatrixXd& x);

    // ---- sklearn-like interface ------------------------------------ //

    /** Predict target values for X.  Returns (M,) column. */
    Eigen::VectorXd predict(const RowMatrixXd& X);

    /**
     * Optimise constants via Levenberg-Marquardt.
     *
     * @param X         Input data  (M × D).
     * @param y         Target values (M,).
     * @param max_iter  Maximum LM iterations.
     * @param tol       Convergence tolerance.
     */
    void fit(const RowMatrixXd& X, const Eigen::VectorXd& y,
             int max_iter = 100, double tol = 1e-10);

    /** Score the expression.  metric: "mae"|"mse"|"rmse"|"bic"|"laplace_nmll" */
    double score(const RowMatrixXd& X, const Eigen::VectorXd& y,
                 const std::string& metric = "mse");

    /** Whether the expression has been fitted (or has no constants). */
    bool is_fitted();

    // ---- Utility --------------------------------------------------- //

    /** Which raw commands are reachable from the output. */
    std::vector<bool> get_utilized_commands() const;

    /** Replace raw layer with its simplified form. */
    void promote_simplification();

    /** Count operator occurrences.
     *  @param tree       true = depth-first tree, false = DAG.
     *  @param terminals  "include" | "exclude" | "combine".
     */
    std::map<int, int> get_operator_counts(
        bool tree = true,
        const std::string& terminals = "exclude");

    /** Element-wise raw command array distance. */
    int distance(const AGraphExpression& other) const;

    /** Deep copy. */
    AGraphExpression copy() const;

    // ---- Hash / equality ------------------------------------------- //

    /** Hash of the simplified command array. */
    std::size_t hash();

    /** Equality by hash comparison. */
    bool equals(AGraphExpression& other);

    // ---- State (for binding / serialization) ----------------------- //

    bool modified() const;

private:
    void notify_modification();
    void update();

    static std::map<int, int> dag_operator_counts(
        const StackMatrix& cmd, const std::string& terminals);
    static std::map<int, int> tree_operator_counts(
        const StackMatrix& cmd, const std::string& terminals);

    // Simplification mode
    std::string simplification_;
    bool propagate_constants_;

    // Raw (GA-facing) data
    StackMatrix raw_command_array_;
    std::vector<double> raw_constants_;
    std::vector<int> raw_integers_;

    // Simplified (evaluation-facing) data
    StackMatrix command_array_;
    std::vector<double> constants_;
    std::vector<int> integers_;
    std::vector<int> constant_mapping_;

    // State tracking
    bool is_fitted_;
    bool modified_;
    std::optional<std::size_t> hash_;
};

}  // namespace cppagraph
