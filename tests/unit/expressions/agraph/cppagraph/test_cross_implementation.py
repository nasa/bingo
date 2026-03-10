"""Cross-implementation validation: cppagraph vs pyagraph.

Generates random AGraph expressions and verifies that the C++ and Python
backends produce bit-exact (or epsilon-close) results for evaluate,
evaluate_with_derivative, simplification, and the full AGraphExpression API.
"""

import copy

import numpy as np
import pytest

from bingo.expressions.agraph.pyagraph.expression import (
    AGraphExpression as PyAGraphExpression,
)
from bingo.expressions.agraph.pyagraph.operators import (
    VARIABLE,
    CONSTANT,
    INTEGER,
    ADDITION,
    SUBTRACTION,
    MULTIPLICATION,
    DIVISION,
    SIN,
    COS,
    EXPONENTIAL,
    SQRT,
    ABS,
    SQUARE,
    CUBE,
    POWER,
    SAFE_POWER,
    LOGARITHM,
    TAN,
    ARCSIN,
    ARCCOS,
    ARCTAN,
    SINH,
    COSH,
    TANH,
)
from bingo.expressions.agraph.pyagraph.evaluation import evaluation as py_eval
from bingo.expressions.agraph.pyagraph.simplification import (
    get_utilized_commands as py_get_utilized,
    reduce as py_reduce,
)

from bingo.expressions.agraph.cppagraph import (
    AGraphExpression as CppAGraphExpression,
    evaluate as cpp_evaluate,
    evaluate_with_derivative as cpp_evaluate_with_derivative,
    get_utilized_commands as cpp_get_utilized,
    reduce as cpp_reduce,
    CachedEvaluator as CppCachedEvaluator,
)
from bingo.expressions.agraph.pyagraph.evaluation.cached_evaluation import (
    CachedEvaluator as PyCachedEvaluator,
)


# ================================================================== #
#  Helpers                                                            #
# ================================================================== #

TERMINAL_OPS = [VARIABLE, CONSTANT, INTEGER]
UNARY_OPS = [
    SIN,
    COS,
    SQRT,
    ABS,
    SQUARE,
    CUBE,
    EXPONENTIAL,
    LOGARITHM,
    TAN,
    ARCSIN,
    ARCCOS,
    ARCTAN,
    SINH,
    COSH,
    TANH,
]
BINARY_OPS = [ADDITION, SUBTRACTION, MULTIPLICATION, DIVISION, POWER, SAFE_POWER]


def _random_stack(rng, n_vars=2, n_rows=8, n_constants=2, n_integers=1):
    """Build a random well-formed command stack.

    Returns (stack, constants, integers).
    """
    stack = np.zeros((n_rows, 3), dtype=np.uint8)
    constants = tuple(rng.standard_normal(n_constants))
    integers = tuple(rng.integers(1, 10, size=n_integers).tolist())

    c_idx = 0
    i_idx = 0

    for row in range(n_rows):
        if row < 2:
            # First rows: terminals
            choice = rng.choice(TERMINAL_OPS)
            if choice == VARIABLE:
                stack[row] = [VARIABLE, rng.integers(0, n_vars), 0]
            elif choice == CONSTANT:
                stack[row] = [CONSTANT, min(c_idx, n_constants - 1), 0]
                c_idx = min(c_idx + 1, n_constants)
            else:
                stack[row] = [INTEGER, min(i_idx, n_integers - 1), 0]
                i_idx = min(i_idx + 1, n_integers)
        else:
            # Pick unary or binary
            if rng.random() < 0.4:
                op = rng.choice(UNARY_OPS)
                p1 = rng.integers(0, row)
                stack[row] = [op, p1, p1]
            else:
                op = rng.choice(BINARY_OPS)
                p1 = rng.integers(0, row)
                p2 = rng.integers(0, row)
                stack[row] = [op, p1, p2]

    return stack, constants, integers


def _make_safe_x(rng, n_samples=10, n_vars=2):
    """Generate x in a range unlikely to cause over/underflow."""
    return rng.uniform(0.1, 2.0, size=(n_samples, n_vars))


# ================================================================== #
#  Cross-implementation: evaluate on 1000 random expressions          #
# ================================================================== #


class TestRandomEvaluateCrossCheck:
    """Generate random stacks and verify cpp evaluate matches py evaluate."""

    N_EXPRESSIONS = 1000

    def test_evaluate_1000_random(self):
        rng = np.random.default_rng(12345)
        x = _make_safe_x(rng, n_samples=20, n_vars=2)
        n_match = 0

        for _ in range(self.N_EXPRESSIONS):
            stack, constants, integers = _random_stack(rng)

            try:
                py_result = py_eval.evaluate(stack, x, constants, integers)
            except Exception:
                continue  # skip ill-formed stacks

            try:
                cpp_result = cpp_evaluate(stack, x, constants, integers)
            except Exception as e:
                pytest.fail(
                    f"cppagraph raised {e} on stack that pyagraph handled:\n"
                    f"stack={stack}, constants={constants}, integers={integers}"
                )

            # Both should produce finite or both NaN at same positions
            py_finite = np.isfinite(py_result)
            cpp_finite = np.isfinite(cpp_result)
            np.testing.assert_array_equal(
                py_finite,
                cpp_finite,
                err_msg="Finite/NaN mismatch",
            )
            # Where both finite, values should be epsilon-close
            mask = py_finite & cpp_finite
            if mask.any():
                np.testing.assert_allclose(
                    cpp_result[mask],
                    py_result[mask],
                    rtol=1e-12,
                    atol=1e-12,
                    err_msg=f"Value mismatch on stack:\n{stack}",
                )
            n_match += 1

        assert n_match >= 900, f"Too few valid expressions: {n_match}"


# ================================================================== #
#  Cross-implementation: derivative on 500 random expressions         #
# ================================================================== #


class TestRandomDerivativeCrossCheck:
    """Generate random stacks and verify derivative matches."""

    N_EXPRESSIONS = 500

    @pytest.mark.parametrize("wrt_x", [True, False], ids=["wrt_x", "wrt_c"])
    def test_derivative_random(self, wrt_x):
        rng = np.random.default_rng(67890)
        x = _make_safe_x(rng, n_samples=15, n_vars=2)
        n_match = 0

        for _ in range(self.N_EXPRESSIONS):
            stack, constants, integers = _random_stack(rng, n_constants=2)

            if not wrt_x and len(constants) == 0:
                continue

            # Reduce stack to remove dead code before derivative comparison.
            # The C++ reverse AD propagates through all rows while pyagraph
            # skips unused ones; reducing makes them agree.
            try:
                stack, constants, integers, _ = py_reduce(stack, constants, integers)
            except Exception:
                continue

            try:
                py_f, py_d = py_eval.evaluate_with_derivative(
                    stack, x, constants, integers, wrt_x
                )
            except Exception:
                continue

            # Skip expressions with large forward values (numerically unstable)
            if not np.all(np.isfinite(py_f)) or np.any(np.abs(py_f) > 1e50):
                continue

            try:
                cpp_f, cpp_d = cpp_evaluate_with_derivative(
                    stack, x, constants, integers, wrt_x
                )
            except Exception as e:
                pytest.fail(
                    f"cppagraph derivative raised {e} on stack pyagraph handled"
                )

            # Function values — filter extreme magnitudes
            mask_f = (
                np.isfinite(py_f)
                & np.isfinite(cpp_f)
                & (np.abs(py_f) < 1e50)
                & (np.abs(cpp_f) < 1e50)
            )
            if mask_f.any():
                np.testing.assert_allclose(
                    cpp_f[mask_f], py_f[mask_f], rtol=1e-10, atol=1e-10
                )

            # Derivative values — filter out extreme magnitudes
            mask_d = (
                np.isfinite(py_d)
                & np.isfinite(cpp_d)
                & (np.abs(py_d) < 1e50)
                & (np.abs(cpp_d) < 1e50)
            )
            if mask_d.any():
                np.testing.assert_allclose(
                    cpp_d[mask_d], py_d[mask_d], rtol=1e-10, atol=1e-10
                )
            n_match += 1

        assert n_match >= 400, f"Too few valid expressions: {n_match}"


# ================================================================== #
#  Cross-implementation: get_utilized_commands and reduce              #
# ================================================================== #


class TestSimplificationCrossCheck:
    N_EXPRESSIONS = 500

    def test_get_utilized_matches(self):
        rng = np.random.default_rng(11111)
        for _ in range(self.N_EXPRESSIONS):
            stack, _, _ = _random_stack(rng)
            py_util = py_get_utilized(stack)
            cpp_util = cpp_get_utilized(stack)
            assert list(py_util) == list(
                cpp_util
            ), f"get_utilized mismatch on stack:\n{stack}"

    def test_reduce_matches(self):
        rng = np.random.default_rng(22222)
        for _ in range(self.N_EXPRESSIONS):
            stack, constants, integers = _random_stack(rng)

            py_stack, py_c, py_i, py_map = py_reduce(stack, constants, integers)
            cpp_stack, cpp_c, cpp_i, cpp_map = cpp_reduce(stack, constants, integers)

            np.testing.assert_array_equal(
                py_stack, cpp_stack, err_msg=f"Reduced stack mismatch on:\n{stack}"
            )
            assert py_c == cpp_c, f"Reduced constants mismatch"
            assert py_i == cpp_i, f"Reduced integers mismatch"
            assert py_map == cpp_map, f"Constant mapping mismatch"


# ================================================================== #
#  Cross-implementation: CachedEvaluator                              #
# ================================================================== #


class TestCachedEvaluatorCrossCheck:
    def test_cached_forward_eval_matches(self):
        rng = np.random.default_rng(33333)
        x = _make_safe_x(rng, n_samples=10, n_vars=2)

        for _ in range(200):
            stack, constants, integers = _random_stack(rng, n_constants=2)

            try:
                py_cached = PyCachedEvaluator(stack, x, integers)
                py_result = py_cached.forward_eval(constants)
            except Exception:
                continue

            cpp_cached = CppCachedEvaluator(stack, x, integers)
            cpp_result = cpp_cached.forward_eval(constants)

            mask = np.isfinite(py_result) & np.isfinite(cpp_result)
            if mask.any():
                np.testing.assert_allclose(
                    cpp_result[mask],
                    py_result[mask],
                    rtol=1e-10,
                    atol=1e-10,
                    err_msg="CachedEvaluator.forward_eval mismatch",
                )

    def test_cached_derivative_matches(self):
        rng = np.random.default_rng(44444)
        x = _make_safe_x(rng, n_samples=10, n_vars=2)

        for _ in range(200):
            stack, constants, integers = _random_stack(rng, n_constants=2)

            # Reduce to remove dead code (see derivative test comment above)
            try:
                stack, constants, integers, _ = py_reduce(stack, constants, integers)
            except Exception:
                continue

            try:
                py_cached = PyCachedEvaluator(stack, x, integers)
                py_f, py_j = py_cached.forward_eval_with_const_derivative(constants)
            except Exception:
                continue

            cpp_cached = CppCachedEvaluator(stack, x, integers)
            cpp_f, cpp_j = cpp_cached.forward_eval_with_const_derivative(constants)

            mask_f = np.isfinite(py_f) & np.isfinite(cpp_f)
            if mask_f.any():
                np.testing.assert_allclose(
                    cpp_f[mask_f], py_f[mask_f], rtol=1e-10, atol=1e-10
                )
            mask_j = np.isfinite(py_j) & np.isfinite(cpp_j)
            if mask_j.any():
                np.testing.assert_allclose(
                    cpp_j[mask_j], py_j[mask_j], rtol=1e-10, atol=1e-10
                )


# ================================================================== #
#  Cross-implementation: AGraphExpression full API                    #
# ================================================================== #


class TestAGraphExpressionCrossCheck:
    """Verify the full AGraphExpression API produces identical results."""

    EQUATIONS = [
        "X_0 + 10.0",
        "sin(X_0)",
        "X_0 * X_1",
        "X_0 * 2.0 + 3.0",
        "X_0 + X_1 + 1.0",
        "X_0 / (X_1 + 0.1)",
    ]

    @pytest.mark.parametrize("eq", EQUATIONS)
    def test_predict_matches(self, eq):
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        py_expr = PyAGraphExpression(equation=eq)
        cpp_expr = CppAGraphExpression(equation=eq)

        py_pred = py_expr.predict(x)
        cpp_pred = cpp_expr.predict(x)
        np.testing.assert_allclose(cpp_pred, py_pred, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("eq", EQUATIONS)
    def test_command_array_matches(self, eq):
        py_expr = PyAGraphExpression(equation=eq)
        cpp_expr = CppAGraphExpression(equation=eq)
        np.testing.assert_array_equal(py_expr.command_array, cpp_expr.command_array)

    @pytest.mark.parametrize("eq", EQUATIONS)
    def test_constants_match(self, eq):
        py_expr = PyAGraphExpression(equation=eq)
        cpp_expr = CppAGraphExpression(equation=eq)
        assert py_expr.constants == pytest.approx(cpp_expr.constants)

    @pytest.mark.parametrize("eq", EQUATIONS)
    def test_complexity_matches(self, eq):
        py_expr = PyAGraphExpression(equation=eq)
        cpp_expr = CppAGraphExpression(equation=eq)
        assert py_expr.complexity == cpp_expr.complexity

    def test_fit_produces_close_constants(self):
        x = np.linspace(0.1, 5.0, 20).reshape(-1, 1)
        y = 3.0 * x.ravel() + 7.0

        py_expr = PyAGraphExpression(equation="X0 * 1.0 + 1.0")
        cpp_expr = CppAGraphExpression(equation="X0 * 1.0 + 1.0")

        py_expr.fit(x, y)
        cpp_expr.fit(x, y)

        for pc, cc in zip(py_expr.constants, cpp_expr.constants):
            assert pc == pytest.approx(cc, abs=0.5)

    def test_score_matches(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = x[:, 0] + 10.0

        py_expr = PyAGraphExpression(equation="X_0 + 10.0")
        cpp_expr = CppAGraphExpression(equation="X_0 + 10.0")

        for metric in ["mse", "mae", "rmse"]:
            py_score = py_expr.score(x, y, metric=metric)
            cpp_score = cpp_expr.score(x, y, metric=metric)
            assert py_score == pytest.approx(
                cpp_score, abs=1e-10
            ), f"score({metric}) mismatch"

    def test_hash_consistent_within_impl(self):
        """Hash is consistent within each implementation (not necessarily across)."""
        cpp_a = CppAGraphExpression(equation="X0 + 1.0")
        cpp_b = CppAGraphExpression(equation="X0 + 1.0")
        assert hash(cpp_a) == hash(cpp_b)

    def test_equality_cross_check(self):
        """Two expressions built the same way should be equal within
        each implementation."""
        py_a = PyAGraphExpression(equation="X0 + 1.0")
        py_b = PyAGraphExpression(equation="X0 + 1.0")
        cpp_a = CppAGraphExpression(equation="X0 + 1.0")
        cpp_b = CppAGraphExpression(equation="X0 + 1.0")
        assert py_a == py_b
        assert cpp_a == cpp_b

    def test_copy_independence(self):
        cpp_expr = CppAGraphExpression(equation="X0 * 2.0")
        clone = cpp_expr.copy()
        clone.constants = (99.0,)
        assert cpp_expr.constants[0] == pytest.approx(2.0)

    def test_get_utilized_commands_matches(self):
        py_expr = PyAGraphExpression(equation="X_0 + 10.0")
        cpp_expr = CppAGraphExpression(equation="X_0 + 10.0")
        assert list(py_expr.get_utilized_commands()) == list(
            cpp_expr.get_utilized_commands()
        )


# ================================================================== #
#  Stress test: 1000 random expressions through full API              #
# ================================================================== #


class TestRandomExpressionCrossCheck:
    """Build 1000 random expressions, compare raw evaluation results."""

    N_EXPRESSIONS = 1000

    def test_predict_1000_random(self):
        rng = np.random.default_rng(99999)
        x = _make_safe_x(rng, n_samples=15, n_vars=2)
        n_match = 0

        for _ in range(self.N_EXPRESSIONS):
            stack, constants, integers = _random_stack(rng)

            # Compare via raw evaluate (bypasses CAS simplification)
            try:
                py_result = py_eval.evaluate(stack, x, constants, integers)
            except Exception:
                continue

            try:
                cpp_result = cpp_evaluate(stack, x, constants, integers)
            except Exception as e:
                pytest.fail(f"cppagraph evaluate raised {e} on valid pyagraph stack")

            py_finite = np.isfinite(py_result)
            cpp_finite = np.isfinite(cpp_result)
            np.testing.assert_array_equal(py_finite, cpp_finite)

            mask = py_finite & cpp_finite & (np.abs(py_result) < 1e100)
            if mask.any():
                np.testing.assert_allclose(
                    cpp_result[mask],
                    py_result[mask],
                    rtol=1e-10,
                    atol=1e-10,
                    err_msg=f"predict mismatch on stack:\n{stack}",
                )
            n_match += 1

        assert n_match >= 900, f"Too few valid expressions: {n_match}"
