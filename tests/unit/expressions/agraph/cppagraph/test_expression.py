"""Tests for cppagraph AGraphExpression — full API parity with pyagraph."""

import copy
import pickle

import numpy as np
import pytest

from bingo.expressions.agraph.cppagraph import (
    AGraphExpression,
    VARIABLE,
    CONSTANT,
    INTEGER,
    ADDITION,
    SUBTRACTION,
    MULTIPLICATION,
    DIVISION,
    SIN,
    SQRT,
)
from bingo.expressions.agraph.pyagraph import AGraphExpression as PyAGraphExpression


# ------------------------------------------------------------------ #
#  Fixtures                                                           #
# ------------------------------------------------------------------ #


@pytest.fixture
def simple_x():
    return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


@pytest.fixture
def x0_plus_c0():
    """Expression: X0 + C0  (C0 = 10.0)"""
    return AGraphExpression(equation="X_0 + 10.0")


@pytest.fixture
def sin_x0():
    """Expression: sin(X0)"""
    return AGraphExpression(equation="sin(X_0)")


@pytest.fixture
def manual_expr():
    """Manually-built expression: X0 * C0  where C0 = 2.0"""
    expr = AGraphExpression()
    expr.raw_command_array = np.array(
        [
            [VARIABLE, 0, 0],
            [CONSTANT, 0, 0],
            [MULTIPLICATION, 0, 1],
        ],
        dtype=np.uint8,
    )
    expr.raw_constants = (2.0,)
    return expr


# ------------------------------------------------------------------ #
#  Construction                                                       #
# ------------------------------------------------------------------ #


class TestConstruction:
    def test_default_constructor(self):
        expr = AGraphExpression()
        assert expr.command_array.shape == (0, 3)
        assert expr.command_array.dtype == np.uint8

    def test_from_equation_string(self, x0_plus_c0):
        assert x0_plus_c0.command_array.dtype == np.uint8
        assert x0_plus_c0.command_array.shape[0] > 0

    def test_from_equation_with_integer(self):
        expr = AGraphExpression(equation="X0 + 7")
        assert len(expr.integers) == 1
        assert expr.integers[0] == 7


# ------------------------------------------------------------------ #
#  Properties                                                         #
# ------------------------------------------------------------------ #


class TestProperties:
    def test_command_array_is_readonly(self, x0_plus_c0):
        with pytest.raises(ValueError):
            x0_plus_c0.command_array[0, 0] = 99

    def test_mutable_raw_command_array_is_writable(self, manual_expr):
        cmd = manual_expr.mutable_raw_command_array
        cmd[0, 1] = 1  # should not raise

    def test_complexity(self, x0_plus_c0):
        assert x0_plus_c0.complexity == 3

    def test_constants_property(self, x0_plus_c0):
        consts = x0_plus_c0.constants
        assert len(consts) == 1
        assert consts[0] == pytest.approx(10.0)

    def test_constants_setter(self):
        expr = AGraphExpression(equation="X0 + 1.0")
        expr.constants = (99.0,)
        assert expr.constants == (99.0,)

    def test_integers_property(self):
        expr = AGraphExpression(equation="X0 + 7")
        ints = expr.integers
        assert len(ints) == 1
        assert ints[0] == 7

    def test_integers_are_fixed_from_source(self):
        expr = AGraphExpression(equation="X0 + 7")
        assert expr.integers == (7,)
        _ = expr.mutable_raw_command_array
        assert expr.integers == (7,)


# ------------------------------------------------------------------ #
#  Evaluation                                                         #
# ------------------------------------------------------------------ #


class TestEvaluation:
    def test_evaluate_x0_plus_c0(self, x0_plus_c0, simple_x):
        result = x0_plus_c0._evaluate(simple_x)
        expected = simple_x[:, 0:1] + 10.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_evaluate_sin_x0(self, sin_x0, simple_x):
        result = sin_x0._evaluate(simple_x)
        expected = np.sin(simple_x[:, 0:1])
        np.testing.assert_array_almost_equal(result, expected)

    def test_evaluate_manual_expr(self, manual_expr, simple_x):
        result = manual_expr._evaluate(simple_x)
        expected = 2.0 * simple_x[:, 0:1]
        np.testing.assert_array_almost_equal(result, expected)

    def test_evaluate_with_x_gradient(self, x0_plus_c0, simple_x):
        f, df_dx = x0_plus_c0._evaluate_with_x_gradient(simple_x)
        np.testing.assert_array_almost_equal(f, simple_x[:, 0:1] + 10.0)
        expected_grad = np.zeros_like(simple_x)
        expected_grad[:, 0] = 1.0
        np.testing.assert_array_almost_equal(df_dx, expected_grad)

    def test_evaluate_with_const_gradient(self, manual_expr, simple_x):
        f, df_dc = manual_expr._evaluate_with_const_gradient(simple_x)
        np.testing.assert_array_almost_equal(f, 2.0 * simple_x[:, 0:1])
        np.testing.assert_array_almost_equal(df_dc, simple_x[:, 0:1])

    def test_evaluate_with_const_hessian(self, manual_expr, simple_x):
        value, gradient, hessian = manual_expr._evaluate_with_const_hessian(
            simple_x
        )
        np.testing.assert_allclose(value, 2.0 * simple_x[:, 0:1])
        np.testing.assert_allclose(gradient, simple_x[:, 0:1])
        np.testing.assert_allclose(hessian, 0.0)


# ------------------------------------------------------------------ #
#  sklearn interface                                                  #
# ------------------------------------------------------------------ #


class TestSklearnInterface:
    def test_predict(self, x0_plus_c0, simple_x):
        pred = x0_plus_c0.predict(simple_x)
        assert pred.ndim == 1
        np.testing.assert_array_almost_equal(pred, simple_x[:, 0] + 10.0)

    def test_loss(self, x0_plus_c0, simple_x):
        y = simple_x[:, 0] + 10.0
        loss = x0_plus_c0.loss(simple_x, y, kind="mse")
        assert loss == pytest.approx(0.0, abs=1e-10)

    def test_loss_mae(self, x0_plus_c0, simple_x):
        y = simple_x[:, 0] + 10.0 + 1.0
        loss = x0_plus_c0.loss(simple_x, y, kind="mae")
        assert loss == pytest.approx(1.0)

    def test_score_r2_perfect_fit(self, x0_plus_c0, simple_x):
        y = simple_x[:, 0] + 10.0
        score = x0_plus_c0.score(simple_x, y)  # default kind="r2"
        assert score == pytest.approx(1.0)

    def test_score_rejects_loss_only_kind(self, x0_plus_c0, simple_x):
        y = simple_x[:, 0] + 10.0
        with pytest.raises(ValueError):
            x0_plus_c0.score(simple_x, y, kind="mse")

    def test_loss_rejects_unknown_kind(self, x0_plus_c0, simple_x):
        y = simple_x[:, 0] + 10.0
        with pytest.raises(ValueError):
            x0_plus_c0.loss(simple_x, y, kind="bogus")

    def test_fit_tolerance_is_keyword_only(self, simple_x):
        expr = AGraphExpression(equation="X0 * 1.0")
        y = 3.0 * simple_x[:, 0]
        with pytest.raises(TypeError):
            expr.fit(simple_x, y, 1e-6)
        expr.fit(simple_x, y, tolerance=1e-6)  # keyword form works

    def test_nonfinite_loss_and_score_normalization(self):
        expr = AGraphExpression(equation="1.0 / X0")
        x = np.array([[0.0]])
        y = np.array([1.0])
        assert expr.loss(x, y, kind="mse") == float("inf")
        assert expr.score(x, y) == float("-inf")

    def test_fit_optimizes_constants(self, simple_x):
        expr = AGraphExpression(equation="X0 * 1.0")
        y = 3.0 * simple_x[:, 0]
        expr.fit(simple_x, y)
        assert expr.constants[0] == pytest.approx(3.0, abs=0.1)

    def test_fit_returns_self(self, simple_x):
        expr = AGraphExpression(equation="X0 + 1.0")
        result = expr.fit(simple_x, simple_x[:, 0])
        assert result is expr


# ------------------------------------------------------------------ #
#  Copy / Pickle                                                      #
# ------------------------------------------------------------------ #


class TestCopyPickle:
    def test_deepcopy(self, x0_plus_c0, simple_x):
        clone = copy.deepcopy(x0_plus_c0)
        np.testing.assert_array_equal(clone.command_array, x0_plus_c0.command_array)
        assert clone.constants == x0_plus_c0.constants
        pred_orig = x0_plus_c0.predict(simple_x)
        pred_clone = clone.predict(simple_x)
        np.testing.assert_array_almost_equal(pred_orig, pred_clone)

    def test_copy_method(self, x0_plus_c0, simple_x):
        clone = x0_plus_c0.copy()
        np.testing.assert_array_equal(clone.command_array, x0_plus_c0.command_array)

    def test_pickle_roundtrip(self, x0_plus_c0, simple_x):
        data = pickle.dumps(x0_plus_c0)
        restored = pickle.loads(data)
        np.testing.assert_array_equal(restored.command_array, x0_plus_c0.command_array)
        np.testing.assert_array_almost_equal(
            restored.predict(simple_x), x0_plus_c0.predict(simple_x)
        )

    def test_deepcopy_is_independent(self, manual_expr):
        clone = copy.deepcopy(manual_expr)
        clone.constants = (99.0,)
        assert manual_expr.constants[0] == pytest.approx(2.0)


# ------------------------------------------------------------------ #
#  Hashing / Equality                                                 #
# ------------------------------------------------------------------ #


class TestHashEquality:
    def test_equal_expressions_have_same_hash(self):
        a = AGraphExpression(equation="X0 + 1.0")
        b = AGraphExpression(equation="X0 + 1.0")
        assert hash(a) == hash(b)

    def test_equal_expressions_are_equal(self):
        a = AGraphExpression(equation="X0 + 1.0")
        b = AGraphExpression(equation="X0 + 1.0")
        assert a == b

    def test_different_expressions_are_not_equal(self):
        a = AGraphExpression(equation="X0 + 1.0")
        b = AGraphExpression(equation="X0 * 1.0")
        assert a != b


# ------------------------------------------------------------------ #
#  String / Formatting                                                #
# ------------------------------------------------------------------ #


class TestFormatting:
    def test_str_not_empty(self, x0_plus_c0):
        s = str(x0_plus_c0)
        assert len(s) > 0

    def test_console_string(self, x0_plus_c0):
        s = x0_plus_c0.console
        assert "X" in s or "x" in s.lower()

    def test_latex_string(self, x0_plus_c0):
        s = x0_plus_c0.latex
        assert len(s) > 0

    def test_constant_folding_string_is_deterministic(self):
        raw_command_array = np.array(
            [
                [1, 0, 0],
                [0, 1, 1],
                [1, 1, 1],
                [0, 4, 4],
                [5, 2, 3],
                [1, 2, 2],
                [0, 0, 0],
                [5, 5, 6],
                [4, 4, 7],
                [5, 1, 8],
                [1, 3, 3],
                [0, 2, 2],
                [5, 10, 11],
                [1, 4, 4],
                [0, 3, 3],
                [5, 13, 14],
                [4, 12, 15],
                [5, 9, 16],
                [3, 0, 17],
            ],
            dtype=np.uint8,
        )
        raw_constants = (-3.4, 0.9, -0.9, -9.0, 9.0)

        py_expr = PyAGraphExpression()
        py_expr.raw_command_array = raw_command_array
        py_expr.raw_constants = raw_constants
        expected = str(py_expr)

        outputs = set()
        for _ in range(10):
            cpp_expr = AGraphExpression()
            cpp_expr.raw_command_array = raw_command_array
            cpp_expr.raw_constants = raw_constants
            outputs.add(str(cpp_expr))

        assert outputs == {expected}

    def test_sympy_string(self, x0_plus_c0):
        s = str(x0_plus_c0.sympy)
        assert len(s) > 0


# ------------------------------------------------------------------ #
#  Simplification / Utilities                                         #
# ------------------------------------------------------------------ #


class TestUtilities:
    def test_get_utilized_commands(self, x0_plus_c0):
        util = x0_plus_c0.get_utilized_commands()
        assert len(util) == x0_plus_c0.command_array.shape[0]
        assert all(u == 1 for u in util)

    def test_promote_simplification(self):
        expr = AGraphExpression(equation="X0 + 0")
        expr.promote_simplification()
        # After promotion, raw should be simplified
        assert expr.raw_command_array.shape[0] <= 1

    def test_get_operator_counts(self, x0_plus_c0):
        counts = x0_plus_c0.get_operator_counts()
        assert isinstance(counts, dict)

    def test_distance(self):
        a = AGraphExpression(equation="X0 + 1.0")
        b = AGraphExpression(equation="X0 * 1.0")
        d = a.distance(b)
        assert isinstance(d, (int, float))
        assert d >= 0


# ------------------------------------------------------------------ #
#  sklearn is_fitted                                                  #
# ------------------------------------------------------------------ #


class TestSklearnIsFitted:
    def test_is_fitted_property(self):
        """After fit, __sklearn_is_fitted__ returns True."""
        expr = AGraphExpression(equation="X0 * 1.0")
        x = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        expr.fit(x, y)
        assert expr.__sklearn_is_fitted__()

    def test_is_fitted_property_matches_hook(self):
        expr = AGraphExpression(equation="X0 + 1.0")
        assert expr.is_fitted == expr.__sklearn_is_fitted__()
        assert not expr.is_fitted
        no_const = AGraphExpression(equation="X0")
        assert no_const.is_fitted

    def test_fit_establishes_and_mutation_unsets(self, simple_x):
        expr = AGraphExpression(equation="X0 * 1.0")
        assert not expr.is_fitted
        expr.fit(simple_x, 2.0 * simple_x[:, 0])
        assert expr.is_fitted
        _ = expr.mutable_raw_command_array
        assert not expr.is_fitted

    def test_non_convergence_still_establishes_fitted(self, simple_x):
        expr = AGraphExpression(equation="X0 * 1.0")
        y = np.full(simple_x.shape[0], np.nan)
        expr.fit(simple_x, y)
        assert expr.is_fitted

    def test_direct_constant_assignment_preserves_but_cannot_establish(
        self, simple_x
    ):
        expr = AGraphExpression(equation="X0 + 1.0")
        expr.constants = (5.0,)
        assert not expr.is_fitted  # cannot establish
        expr.fit(simple_x, simple_x[:, 0] + 3.0)
        assert expr.is_fitted
        expr.constants = (7.0,)
        assert expr.is_fitted  # preserved

    def test_pickle_preserves_fitted_lifecycle(self, simple_x):
        expr = AGraphExpression(equation="X0 * 1.0")
        expr.fit(simple_x, 3.0 * simple_x[:, 0])
        assert expr.is_fitted
        restored = pickle.loads(pickle.dumps(expr))
        assert restored.is_fitted
        assert restored.constants == pytest.approx(expr.constants)

        unfitted = AGraphExpression(equation="X0 + 1.0")
        restored_unfitted = pickle.loads(pickle.dumps(unfitted))
        assert not restored_unfitted.is_fitted

    def test_copy_preserves_fitted_lifecycle(self, simple_x):
        expr = AGraphExpression(equation="X0 * 1.0")
        expr.fit(simple_x, 3.0 * simple_x[:, 0])
        assert expr.copy().is_fitted

        unfitted = AGraphExpression(equation="X0 + 1.0")
        assert not unfitted.copy().is_fitted


# ------------------------------------------------------------------ #
#  Gradient                                                           #
# ------------------------------------------------------------------ #


class TestGradient:
    def test_gradient_returns_predictions_and_input_gradient(
        self, x0_plus_c0, simple_x
    ):
        f, df_dx = x0_plus_c0.gradient(simple_x)
        np.testing.assert_allclose(f, simple_x[:, 0] + 10.0)
        expected = np.zeros_like(simple_x)
        expected[:, 0] = 1.0
        np.testing.assert_allclose(df_dx, expected)


# ------------------------------------------------------------------ #
#  Implicit regression                                               #
# ------------------------------------------------------------------ #


class TestImplicitRegression:
    def test_fit_implicit_establishes_fitted(self, simple_x):
        expr = AGraphExpression(equation="X0 * 1.0 + X1 * 1.0")
        dx_dt = np.ones_like(simple_x)
        assert not expr.is_fitted
        expr.fit_implicit(simple_x, dx_dt)
        assert expr.is_fitted

    def test_fit_implicit_tolerance_is_keyword_only(self, simple_x):
        expr = AGraphExpression(equation="X0 * 1.0 + X1 * 1.0")
        dx_dt = np.ones_like(simple_x)
        with pytest.raises(TypeError):
            expr.fit_implicit(simple_x, dx_dt, 1e-6)
        expr.fit_implicit(simple_x, dx_dt, tolerance=1e-6)

    def test_implicit_score_is_negated_loss(self, simple_x):
        expr = AGraphExpression(equation="X0 + X1")
        dx_dt = np.ones_like(simple_x)
        loss = expr.implicit_loss(simple_x, dx_dt)
        score = expr.implicit_score(simple_x, dx_dt)
        assert np.isfinite(loss)
        assert score == pytest.approx(-loss)

    def test_required_params_guard_yields_inf(self, simple_x):
        expr = AGraphExpression(equation="5.0")  # constant, zero gradient
        dx_dt = np.ones_like(simple_x)
        assert expr.implicit_loss(
            simple_x, dx_dt, required_params=1
        ) == float("inf")
        assert expr.implicit_score(
            simple_x, dx_dt, required_params=1
        ) == float("-inf")
