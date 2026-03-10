/**
 * @file operator_eval.cpp
 * @brief Per-operator forward and reverse evaluation functions.
 *
 * Mirrors bingo.expressions.agraph.pyagraph.evaluation.operator_eval.
 *
 * Protected-math conventions (matching pyagraph):
 *   - sqrt(x)  → sqrt(|x|)
 *   - log(x)   → log(|x|)
 *   - safe_power(a,b) → |a|^b
 *
 * We operate on Eigen matrices (MxC) throughout.
 * Scalars (e.g. integer nodes) are stored as 1x1 matrices so that
 * binary operators can rely on Eigen broadcasting.
 */

#include "cppagraph/operator_eval.h"
#include "cppagraph/operators.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace cppagraph {

// ================================================================
//  Helper: ensure `rev[idx]` has the right shape for accumulation.
//  The reverse buffer may hold a 1x1 "scalar" when the seed is 1.0;
//  we broadcast it to match a target entry if needed.
// ================================================================

static inline void ensure_shape(ReverseBuf& rev, int idx,
                                Eigen::Index rows, Eigen::Index cols) {
    auto& m = rev[idx];
    if (m.rows() == rows && m.cols() == cols) return;
    if (m.size() == 1) {
        double v = m(0, 0);
        m = RowMatrixXd::Constant(rows, cols, v);
    }
}

// Same for the forward buffer entries that may be scalar.
static inline RowMatrixXd as_matrix(const RowMatrixXd& v,
                                     Eigen::Index rows, Eigen::Index cols) {
    if (v.rows() == rows && v.cols() == cols) return v;
    if (v.size() == 1)
        return RowMatrixXd::Constant(rows, cols, v(0, 0));
    return v;  // let Eigen handle broadcasting errors
}

/// Compute the broadcast shape from rev[ri] and one forward entry.
static inline std::pair<Eigen::Index, Eigen::Index>
bcast_shape(const ReverseBuf& rev, int ri,
            const ForwardBuf& fwd, uint8_t p1) {
    return {std::max(rev[ri].rows(), fwd[p1].rows()),
            std::max(rev[ri].cols(), fwd[p1].cols())};
}

/// Compute the broadcast shape from rev[ri] and two forward entries.
static inline std::pair<Eigen::Index, Eigen::Index>
bcast_shape2(const ReverseBuf& rev, int ri,
             const ForwardBuf& fwd, uint8_t p1, uint8_t p2) {
    return {std::max({rev[ri].rows(), fwd[p1].rows(), fwd[p2].rows()}),
            std::max({rev[ri].cols(), fwd[p1].cols(), fwd[p2].cols()})};
}

// ================================================================
//  Terminals
// ================================================================

// VARIABLE
static RowMatrixXd var_fwd(uint8_t p1, uint8_t /*p2*/,
                            const RowMatrixXd& x,
                            const std::vector<double>& /*c*/,
                            const std::vector<int>& /*i*/,
                            const ForwardBuf& /*fwd*/) {
    return x.col(p1);  // Mx1
}

// CONSTANT
static RowMatrixXd const_fwd(uint8_t p1, uint8_t /*p2*/,
                              const RowMatrixXd& /*x*/,
                              const std::vector<double>& c,
                              const std::vector<int>& /*i*/,
                              const ForwardBuf& /*fwd*/) {
    RowMatrixXd m(1, 1);
    m(0, 0) = c[p1];
    return m;
}

// INTEGER
static RowMatrixXd int_fwd(uint8_t p1, uint8_t /*p2*/,
                            const RowMatrixXd& /*x*/,
                            const std::vector<double>& /*c*/,
                            const std::vector<int>& integers,
                            const ForwardBuf& /*fwd*/) {
    RowMatrixXd m(1, 1);
    m(0, 0) = static_cast<double>(integers[p1]);
    return m;
}

// ================================================================
//  Arithmetic
// ================================================================

// ADDITION
static RowMatrixXd add_fwd(uint8_t p1, uint8_t p2,
                            const RowMatrixXd& /*x*/,
                            const std::vector<double>& /*c*/,
                            const std::vector<int>& /*i*/,
                            const ForwardBuf& fwd) {
    const auto& a = fwd[p1];
    const auto& b = fwd[p2];
    Eigen::Index rows = std::max(a.rows(), b.rows());
    Eigen::Index cols = std::max(a.cols(), b.cols());
    return as_matrix(a, rows, cols) + as_matrix(b, rows, cols);
}

static void add_rev(int ri, uint8_t p1, uint8_t p2,
                     const ForwardBuf& /*fwd*/, ReverseBuf& rev) {
    auto rows = rev[ri].rows();
    auto cols = rev[ri].cols();
    ensure_shape(rev, p1, rows, cols);
    ensure_shape(rev, p2, rows, cols);
    rev[p1].array() += rev[ri].array();
    rev[p2].array() += rev[ri].array();
}

// SUBTRACTION
static RowMatrixXd sub_fwd(uint8_t p1, uint8_t p2,
                            const RowMatrixXd& /*x*/,
                            const std::vector<double>& /*c*/,
                            const std::vector<int>& /*i*/,
                            const ForwardBuf& fwd) {
    const auto& a = fwd[p1];
    const auto& b = fwd[p2];
    Eigen::Index rows = std::max(a.rows(), b.rows());
    Eigen::Index cols = std::max(a.cols(), b.cols());
    return as_matrix(a, rows, cols) - as_matrix(b, rows, cols);
}

static void sub_rev(int ri, uint8_t p1, uint8_t p2,
                     const ForwardBuf& /*fwd*/, ReverseBuf& rev) {
    auto rows = rev[ri].rows();
    auto cols = rev[ri].cols();
    ensure_shape(rev, p1, rows, cols);
    ensure_shape(rev, p2, rows, cols);
    rev[p1].array() += rev[ri].array();
    rev[p2].array() -= rev[ri].array();
}

// MULTIPLICATION
static RowMatrixXd mul_fwd(uint8_t p1, uint8_t p2,
                            const RowMatrixXd& /*x*/,
                            const std::vector<double>& /*c*/,
                            const std::vector<int>& /*i*/,
                            const ForwardBuf& fwd) {
    const auto& a = fwd[p1];
    const auto& b = fwd[p2];
    Eigen::Index rows = std::max(a.rows(), b.rows());
    Eigen::Index cols = std::max(a.cols(), b.cols());
    return as_matrix(a, rows, cols).array() * as_matrix(b, rows, cols).array();
}

static void mul_rev(int ri, uint8_t p1, uint8_t p2,
                     const ForwardBuf& fwd, ReverseBuf& rev) {
    auto [rows, cols] = bcast_shape2(rev, ri, fwd, p1, p2);
    ensure_shape(rev, ri, rows, cols);
    ensure_shape(rev, p1, rows, cols);
    ensure_shape(rev, p2, rows, cols);
    auto fwd_p2 = as_matrix(fwd[p2], rows, cols);
    auto fwd_p1 = as_matrix(fwd[p1], rows, cols);
    rev[p1].array() += rev[ri].array() * fwd_p2.array();
    rev[p2].array() += rev[ri].array() * fwd_p1.array();
}

// DIVISION
static RowMatrixXd div_fwd(uint8_t p1, uint8_t p2,
                            const RowMatrixXd& /*x*/,
                            const std::vector<double>& /*c*/,
                            const std::vector<int>& /*i*/,
                            const ForwardBuf& fwd) {
    const auto& a = fwd[p1];
    const auto& b = fwd[p2];
    Eigen::Index rows = std::max(a.rows(), b.rows());
    Eigen::Index cols = std::max(a.cols(), b.cols());
    return as_matrix(a, rows, cols).array() / as_matrix(b, rows, cols).array();
}

static void div_rev(int ri, uint8_t p1, uint8_t p2,
                     const ForwardBuf& fwd, ReverseBuf& rev) {
    auto [rows, cols] = bcast_shape2(rev, ri, fwd, p1, p2);
    ensure_shape(rev, ri, rows, cols);
    ensure_shape(rev, p1, rows, cols);
    ensure_shape(rev, p2, rows, cols);
    auto fwd_p2 = as_matrix(fwd[p2], rows, cols);
    auto fwd_ri = as_matrix(fwd[ri], rows, cols);
    rev[p1].array() += rev[ri].array() / fwd_p2.array();
    rev[p2].array() -= rev[ri].array() * fwd_ri.array() / fwd_p2.array();
}

// ================================================================
//  Power / Root
// ================================================================

// POWER: a^b
static RowMatrixXd pow_fwd(uint8_t p1, uint8_t p2,
                            const RowMatrixXd& /*x*/,
                            const std::vector<double>& /*c*/,
                            const std::vector<int>& /*i*/,
                            const ForwardBuf& fwd) {
    const auto& a = fwd[p1];
    const auto& b = fwd[p2];
    Eigen::Index rows = std::max(a.rows(), b.rows());
    Eigen::Index cols = std::max(a.cols(), b.cols());
    return as_matrix(a, rows, cols).array().pow(as_matrix(b, rows, cols).array());
}

static void pow_rev(int ri, uint8_t p1, uint8_t p2,
                     const ForwardBuf& fwd, ReverseBuf& rev) {
    auto [rows, cols] = bcast_shape2(rev, ri, fwd, p1, p2);
    ensure_shape(rev, ri, rows, cols);
    ensure_shape(rev, p1, rows, cols);
    ensure_shape(rev, p2, rows, cols);
    auto fwd_ri = as_matrix(fwd[ri], rows, cols);
    auto fwd_p1 = as_matrix(fwd[p1], rows, cols);
    auto fwd_p2 = as_matrix(fwd[p2], rows, cols);
    // d/da (a^b) = b * a^(b-1) = f(ri) * b / a
    rev[p1].array() += rev[ri].array() * fwd_ri.array() * fwd_p2.array() / fwd_p1.array();
    // d/db (a^b) = a^b * ln(a) = f(ri) * ln(a)
    rev[p2].array() += rev[ri].array() * fwd_ri.array() * fwd_p1.array().log();
}

// SAFE_POWER: |a|^b
static RowMatrixXd safe_pow_fwd(uint8_t p1, uint8_t p2,
                                 const RowMatrixXd& /*x*/,
                                 const std::vector<double>& /*c*/,
                                 const std::vector<int>& /*i*/,
                                 const ForwardBuf& fwd) {
    const auto& a = fwd[p1];
    const auto& b = fwd[p2];
    Eigen::Index rows = std::max(a.rows(), b.rows());
    Eigen::Index cols = std::max(a.cols(), b.cols());
    return as_matrix(a, rows, cols).array().abs().pow(as_matrix(b, rows, cols).array());
}

static void safe_pow_rev(int ri, uint8_t p1, uint8_t p2,
                          const ForwardBuf& fwd, ReverseBuf& rev) {
    auto [rows, cols] = bcast_shape2(rev, ri, fwd, p1, p2);
    ensure_shape(rev, ri, rows, cols);
    ensure_shape(rev, p1, rows, cols);
    ensure_shape(rev, p2, rows, cols);
    auto fwd_ri = as_matrix(fwd[ri], rows, cols);
    auto fwd_p1 = as_matrix(fwd[p1], rows, cols);
    auto fwd_p2 = as_matrix(fwd[p2], rows, cols);
    // d/da |a|^b  = b * |a|^(b-1) * sign(a) = f(ri) * b / a
    rev[p1].array() += rev[ri].array() * fwd_ri.array() * fwd_p2.array() / fwd_p1.array();
    // d/db |a|^b  = |a|^b * ln|a| = f(ri) * ln|a|
    rev[p2].array() += rev[ri].array() * fwd_ri.array() * fwd_p1.array().abs().log();
}

// SQUARE: a^2
static RowMatrixXd square_fwd(uint8_t p1, uint8_t /*p2*/,
                               const RowMatrixXd& /*x*/,
                               const std::vector<double>& /*c*/,
                               const std::vector<int>& /*i*/,
                               const ForwardBuf& fwd) {
    return fwd[p1].array().square();
}

static void square_rev(int ri, uint8_t p1, uint8_t /*p2*/,
                        const ForwardBuf& fwd, ReverseBuf& rev) {
    auto [rows, cols] = bcast_shape(rev, ri, fwd, p1);
    ensure_shape(rev, ri, rows, cols);
    ensure_shape(rev, p1, rows, cols);
    auto fwd_p1 = as_matrix(fwd[p1], rows, cols);
    rev[p1].array() += rev[ri].array() * 2.0 * fwd_p1.array();
}

// CUBE: a^3
static RowMatrixXd cube_fwd(uint8_t p1, uint8_t /*p2*/,
                             const RowMatrixXd& /*x*/,
                             const std::vector<double>& /*c*/,
                             const std::vector<int>& /*i*/,
                             const ForwardBuf& fwd) {
    return fwd[p1].array().cube();
}

static void cube_rev(int ri, uint8_t p1, uint8_t /*p2*/,
                      const ForwardBuf& fwd, ReverseBuf& rev) {
    auto [rows, cols] = bcast_shape(rev, ri, fwd, p1);
    ensure_shape(rev, ri, rows, cols);
    ensure_shape(rev, p1, rows, cols);
    auto fwd_p1 = as_matrix(fwd[p1], rows, cols);
    rev[p1].array() += rev[ri].array() * 3.0 * fwd_p1.array().square();
}

// SQRT: sqrt(|a|)
static RowMatrixXd sqrt_fwd(uint8_t p1, uint8_t /*p2*/,
                             const RowMatrixXd& /*x*/,
                             const std::vector<double>& /*c*/,
                             const std::vector<int>& /*i*/,
                             const ForwardBuf& fwd) {
    return fwd[p1].array().abs().sqrt();
}

static void sqrt_rev(int ri, uint8_t p1, uint8_t /*p2*/,
                      const ForwardBuf& fwd, ReverseBuf& rev) {
    auto [rows, cols] = bcast_shape(rev, ri, fwd, p1);
    ensure_shape(rev, ri, rows, cols);
    ensure_shape(rev, p1, rows, cols);
    auto fwd_ri = as_matrix(fwd[ri], rows, cols);
    auto fwd_p1 = as_matrix(fwd[p1], rows, cols);
    // d/da sqrt(|a|) = 0.5 / sqrt(|a|) * sign(a)
    rev[p1].array() += 0.5 * rev[ri].array() / fwd_ri.array() * fwd_p1.array().sign();
}

// ================================================================
//  Miscellaneous
// ================================================================

// ABS: |a|
static RowMatrixXd abs_fwd(uint8_t p1, uint8_t /*p2*/,
                            const RowMatrixXd& /*x*/,
                            const std::vector<double>& /*c*/,
                            const std::vector<int>& /*i*/,
                            const ForwardBuf& fwd) {
    return fwd[p1].array().abs();
}

static void abs_rev(int ri, uint8_t p1, uint8_t /*p2*/,
                     const ForwardBuf& fwd, ReverseBuf& rev) {
    auto [rows, cols] = bcast_shape(rev, ri, fwd, p1);
    ensure_shape(rev, ri, rows, cols);
    ensure_shape(rev, p1, rows, cols);
    auto fwd_p1 = as_matrix(fwd[p1], rows, cols);
    rev[p1].array() += rev[ri].array() * fwd_p1.array().sign();
}

// ================================================================
//  Exponential / Logarithmic
// ================================================================

// EXPONENTIAL: exp(a)
static RowMatrixXd exp_fwd(uint8_t p1, uint8_t /*p2*/,
                            const RowMatrixXd& /*x*/,
                            const std::vector<double>& /*c*/,
                            const std::vector<int>& /*i*/,
                            const ForwardBuf& fwd) {
    return fwd[p1].array().exp();
}

static void exp_rev(int ri, uint8_t p1, uint8_t /*p2*/,
                     const ForwardBuf& fwd, ReverseBuf& rev) {
    auto [rows, cols] = bcast_shape(rev, ri, fwd, p1);
    ensure_shape(rev, ri, rows, cols);
    ensure_shape(rev, p1, rows, cols);
    auto fwd_ri = as_matrix(fwd[ri], rows, cols);
    rev[p1].array() += rev[ri].array() * fwd_ri.array();
}

// LOGARITHM: log(|a|)
static RowMatrixXd log_fwd(uint8_t p1, uint8_t /*p2*/,
                            const RowMatrixXd& /*x*/,
                            const std::vector<double>& /*c*/,
                            const std::vector<int>& /*i*/,
                            const ForwardBuf& fwd) {
    return fwd[p1].array().abs().log();
}

static void log_rev(int ri, uint8_t p1, uint8_t /*p2*/,
                     const ForwardBuf& fwd, ReverseBuf& rev) {
    auto [rows, cols] = bcast_shape(rev, ri, fwd, p1);
    ensure_shape(rev, ri, rows, cols);
    ensure_shape(rev, p1, rows, cols);
    auto fwd_p1 = as_matrix(fwd[p1], rows, cols);
    rev[p1].array() += rev[ri].array() / fwd_p1.array();
}

// ================================================================
//  Trigonometric
// ================================================================

// SIN
static RowMatrixXd sin_fwd(uint8_t p1, uint8_t /*p2*/,
                            const RowMatrixXd& /*x*/,
                            const std::vector<double>& /*c*/,
                            const std::vector<int>& /*i*/,
                            const ForwardBuf& fwd) {
    return fwd[p1].array().sin();
}

static void sin_rev(int ri, uint8_t p1, uint8_t /*p2*/,
                     const ForwardBuf& fwd, ReverseBuf& rev) {
    auto [rows, cols] = bcast_shape(rev, ri, fwd, p1);
    ensure_shape(rev, ri, rows, cols);
    ensure_shape(rev, p1, rows, cols);
    auto fwd_p1 = as_matrix(fwd[p1], rows, cols);
    rev[p1].array() += rev[ri].array() * fwd_p1.array().cos();
}

// COS
static RowMatrixXd cos_fwd(uint8_t p1, uint8_t /*p2*/,
                            const RowMatrixXd& /*x*/,
                            const std::vector<double>& /*c*/,
                            const std::vector<int>& /*i*/,
                            const ForwardBuf& fwd) {
    return fwd[p1].array().cos();
}

static void cos_rev(int ri, uint8_t p1, uint8_t /*p2*/,
                     const ForwardBuf& fwd, ReverseBuf& rev) {
    auto [rows, cols] = bcast_shape(rev, ri, fwd, p1);
    ensure_shape(rev, ri, rows, cols);
    ensure_shape(rev, p1, rows, cols);
    auto fwd_p1 = as_matrix(fwd[p1], rows, cols);
    rev[p1].array() -= rev[ri].array() * fwd_p1.array().sin();
}

// TAN
static RowMatrixXd tan_fwd(uint8_t p1, uint8_t /*p2*/,
                            const RowMatrixXd& /*x*/,
                            const std::vector<double>& /*c*/,
                            const std::vector<int>& /*i*/,
                            const ForwardBuf& fwd) {
    return fwd[p1].array().tan();
}

static void tan_rev(int ri, uint8_t p1, uint8_t /*p2*/,
                     const ForwardBuf& fwd, ReverseBuf& rev) {
    auto [rows, cols] = bcast_shape(rev, ri, fwd, p1);
    ensure_shape(rev, ri, rows, cols);
    ensure_shape(rev, p1, rows, cols);
    auto fwd_p1 = as_matrix(fwd[p1], rows, cols);
    rev[p1].array() += rev[ri].array() / fwd_p1.array().cos().square();
}

// ARCSIN
static RowMatrixXd arcsin_fwd(uint8_t p1, uint8_t /*p2*/,
                               const RowMatrixXd& /*x*/,
                               const std::vector<double>& /*c*/,
                               const std::vector<int>& /*i*/,
                               const ForwardBuf& fwd) {
    return fwd[p1].array().asin();
}

static void arcsin_rev(int ri, uint8_t p1, uint8_t /*p2*/,
                        const ForwardBuf& fwd, ReverseBuf& rev) {
    auto [rows, cols] = bcast_shape(rev, ri, fwd, p1);
    ensure_shape(rev, ri, rows, cols);
    ensure_shape(rev, p1, rows, cols);
    auto fwd_p1 = as_matrix(fwd[p1], rows, cols);
    auto xsqr = fwd_p1.array().square();
    rev[p1].array() += rev[ri].array() / (1.0 - xsqr).sqrt();
}

// ARCCOS
static RowMatrixXd arccos_fwd(uint8_t p1, uint8_t /*p2*/,
                               const RowMatrixXd& /*x*/,
                               const std::vector<double>& /*c*/,
                               const std::vector<int>& /*i*/,
                               const ForwardBuf& fwd) {
    return fwd[p1].array().acos();
}

static void arccos_rev(int ri, uint8_t p1, uint8_t /*p2*/,
                        const ForwardBuf& fwd, ReverseBuf& rev) {
    auto [rows, cols] = bcast_shape(rev, ri, fwd, p1);
    ensure_shape(rev, ri, rows, cols);
    ensure_shape(rev, p1, rows, cols);
    auto fwd_p1 = as_matrix(fwd[p1], rows, cols);
    auto xsqr = fwd_p1.array().square();
    rev[p1].array() -= rev[ri].array() / (1.0 - xsqr).sqrt();
}

// ARCTAN
static RowMatrixXd arctan_fwd(uint8_t p1, uint8_t /*p2*/,
                               const RowMatrixXd& /*x*/,
                               const std::vector<double>& /*c*/,
                               const std::vector<int>& /*i*/,
                               const ForwardBuf& fwd) {
    return fwd[p1].array().atan();
}

static void arctan_rev(int ri, uint8_t p1, uint8_t /*p2*/,
                        const ForwardBuf& fwd, ReverseBuf& rev) {
    auto [rows, cols] = bcast_shape(rev, ri, fwd, p1);
    ensure_shape(rev, ri, rows, cols);
    ensure_shape(rev, p1, rows, cols);
    auto fwd_p1 = as_matrix(fwd[p1], rows, cols);
    auto xsqr = fwd_p1.array().square();
    rev[p1].array() += rev[ri].array() / (1.0 + xsqr);
}

// ================================================================
//  Hyperbolic
// ================================================================

// SINH
static RowMatrixXd sinh_fwd(uint8_t p1, uint8_t /*p2*/,
                             const RowMatrixXd& /*x*/,
                             const std::vector<double>& /*c*/,
                             const std::vector<int>& /*i*/,
                             const ForwardBuf& fwd) {
    return fwd[p1].array().sinh();
}

static void sinh_rev(int ri, uint8_t p1, uint8_t /*p2*/,
                      const ForwardBuf& fwd, ReverseBuf& rev) {
    auto [rows, cols] = bcast_shape(rev, ri, fwd, p1);
    ensure_shape(rev, ri, rows, cols);
    ensure_shape(rev, p1, rows, cols);
    auto fwd_p1 = as_matrix(fwd[p1], rows, cols);
    rev[p1].array() += rev[ri].array() * fwd_p1.array().cosh();
}

// COSH
static RowMatrixXd cosh_fwd(uint8_t p1, uint8_t /*p2*/,
                             const RowMatrixXd& /*x*/,
                             const std::vector<double>& /*c*/,
                             const std::vector<int>& /*i*/,
                             const ForwardBuf& fwd) {
    return fwd[p1].array().cosh();
}

static void cosh_rev(int ri, uint8_t p1, uint8_t /*p2*/,
                      const ForwardBuf& fwd, ReverseBuf& rev) {
    auto [rows, cols] = bcast_shape(rev, ri, fwd, p1);
    ensure_shape(rev, ri, rows, cols);
    ensure_shape(rev, p1, rows, cols);
    auto fwd_p1 = as_matrix(fwd[p1], rows, cols);
    rev[p1].array() += rev[ri].array() * fwd_p1.array().sinh();
}

// TANH
static RowMatrixXd tanh_fwd(uint8_t p1, uint8_t /*p2*/,
                             const RowMatrixXd& /*x*/,
                             const std::vector<double>& /*c*/,
                             const std::vector<int>& /*i*/,
                             const ForwardBuf& fwd) {
    return fwd[p1].array().tanh();
}

static void tanh_rev(int ri, uint8_t p1, uint8_t /*p2*/,
                      const ForwardBuf& fwd, ReverseBuf& rev) {
    auto [rows, cols] = bcast_shape(rev, ri, fwd, p1);
    ensure_shape(rev, ri, rows, cols);
    ensure_shape(rev, p1, rows, cols);
    auto fwd_p1 = as_matrix(fwd[p1], rows, cols);
    rev[p1].array() += rev[ri].array() / fwd_p1.array().cosh().square();
}

// ================================================================
//  Dispatch tables
// ================================================================

using FwdFn = RowMatrixXd(*)(uint8_t, uint8_t,
                              const RowMatrixXd&,
                              const std::vector<double>&,
                              const std::vector<int>&,
                              const ForwardBuf&);

using RevFn = void(*)(int, uint8_t, uint8_t,
                       const ForwardBuf&, ReverseBuf&);

// Terminal reverse functions are no-ops.
static void noop_rev(int, uint8_t, uint8_t,
                      const ForwardBuf&, ReverseBuf&) {}

static constexpr std::array<FwdFn, NUM_OPS> FWD_TABLE = {
    var_fwd,        // VARIABLE    0
    const_fwd,      // CONSTANT    1
    int_fwd,        // INTEGER     2
    add_fwd,        // ADDITION    3
    sub_fwd,        // SUBTRACTION 4
    mul_fwd,        // MULTIPLY    5
    div_fwd,        // DIVISION    6
    pow_fwd,        // POWER       7
    safe_pow_fwd,   // SAFE_POWER  8
    square_fwd,     // SQUARE      9
    cube_fwd,       // CUBE       10
    sqrt_fwd,       // SQRT       11
    abs_fwd,        // ABS        12
    exp_fwd,        // EXPONENTIAL 13
    log_fwd,        // LOGARITHM  14
    sin_fwd,        // SIN        15
    cos_fwd,        // COS        16
    tan_fwd,        // TAN        17
    arcsin_fwd,     // ARCSIN     18
    arccos_fwd,     // ARCCOS     19
    arctan_fwd,     // ARCTAN     20
    sinh_fwd,       // SINH       21
    cosh_fwd,       // COSH       22
    tanh_fwd,       // TANH       23
};

static constexpr std::array<RevFn, NUM_OPS> REV_TABLE = {
    noop_rev,       // VARIABLE    0
    noop_rev,       // CONSTANT    1
    noop_rev,       // INTEGER     2
    add_rev,        // ADDITION    3
    sub_rev,        // SUBTRACTION 4
    mul_rev,        // MULTIPLY    5
    div_rev,        // DIVISION    6
    pow_rev,        // POWER       7
    safe_pow_rev,   // SAFE_POWER  8
    square_rev,     // SQUARE      9
    cube_rev,       // CUBE       10
    sqrt_rev,       // SQRT       11
    abs_rev,        // ABS        12
    exp_rev,        // EXPONENTIAL 13
    log_rev,        // LOGARITHM  14
    sin_rev,        // SIN        15
    cos_rev,        // COS        16
    tan_rev,        // TAN        17
    arcsin_rev,     // ARCSIN     18
    arccos_rev,     // ARCCOS     19
    arctan_rev,     // ARCTAN     20
    sinh_rev,       // SINH       21
    cosh_rev,       // COSH       22
    tanh_rev,       // TANH       23
};

// ================================================================
//  Public dispatch
// ================================================================

RowMatrixXd forward_eval_one(
        uint8_t node, uint8_t param1, uint8_t param2,
        const RowMatrixXd& x,
        const std::vector<double>& constants,
        const std::vector<int>& integers,
        const ForwardBuf& fwd) {
    if (node >= NUM_OPS)
        throw std::out_of_range("Unknown operator id " + std::to_string(node));
    return FWD_TABLE[node](param1, param2, x, constants, integers, fwd);
}

void reverse_eval_one(
        uint8_t node, int ri, uint8_t param1, uint8_t param2,
        const ForwardBuf& fwd, ReverseBuf& rev) {
    if (node >= NUM_OPS)
        throw std::out_of_range("Unknown operator id " + std::to_string(node));
    REV_TABLE[node](ri, param1, param2, fwd, rev);
}

}  // namespace cppagraph
