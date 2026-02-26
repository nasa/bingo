"""Tests for bingo.expressions.agraph.expression"""

import copy
import pickle

import numpy as np
import pytest
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from bingo.expressions.agraph.expression import AGraphExpression
from bingo.expressions.agraph.operators import (
    VARIABLE,
    CONSTANT,
    INTEGER,
    ADDITION,
    MULTIPLICATION,
    SIN,
)


# ------------------------------------------------------------------ #
#  Fixtures                                                           #
# ------------------------------------------------------------------ #


@pytest.fixture
def simple_x():
    return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


@pytest.fixture
def x0_plus_c0():
    """Expression: X0 + C0  (C0 = 10.0)"""
    expr = AGraphExpression(equation="X_0 + 10.0")
    return expr


@pytest.fixture
def sin_x0():
    """Expression: sin(X0)"""
    return AGraphExpression(equation="sin(X_0)")


@pytest.fixture
def manual_expr():
    """Manually-built expression: X0 * C0  where C0 = 2.0"""
    expr = AGraphExpression()
    expr.command_array = np.array(
        [
            [VARIABLE, 0, 0],
            [CONSTANT, 0, 0],
            [MULTIPLICATION, 0, 1],
        ],
        dtype=np.uint8,
    )
    expr.constants = (2.0,)
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

    def test_from_equation_no_underscore(self):
        """Parser accepts X0 (no underscore) as well as X_0."""
        expr = AGraphExpression(equation="X0 + X1")
        assert expr.command_array.shape[0] > 0


# ------------------------------------------------------------------ #
#  Properties                                                         #
# ------------------------------------------------------------------ #


class TestProperties:
    def test_command_array_is_readonly(self, x0_plus_c0):
        with pytest.raises(ValueError):
            x0_plus_c0.command_array[0, 0] = 99

    def test_mutable_command_array_is_writable(self, manual_expr):
        cmd = manual_expr.mutable_command_array
        cmd[0, 1] = 1  # should not raise

    def test_complexity(self, x0_plus_c0):
        # X0 + 10.0 => 3 commands (X0, C0, ADD)
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
        """Integers come from the unsimplified command array, never
        fabricated."""
        expr = AGraphExpression(equation="X0 + 7")
        assert expr.integers == (7,)
        # After modifying command array to add a new INTEGER node
        # that references index 0 of _integers (which is 7), the
        # value should still be 7.
        cmd = expr.mutable_command_array
        # no-op mutation, just verify re-update keeps integer 7
        assert expr.integers == (7,)


# ------------------------------------------------------------------ #
#  Evaluation (private methods)                                       #
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
        # df/dX0 = 1, df/dX1 = 0
        expected_grad = np.zeros_like(simple_x)
        expected_grad[:, 0] = 1.0
        np.testing.assert_array_almost_equal(df_dx, expected_grad)

    def test_evaluate_with_const_gradient(self, manual_expr, simple_x):
        f, df_dc = manual_expr._evaluate_with_const_gradient(simple_x)
        # f = C0 * X0  =>  df/dC0 = X0
        np.testing.assert_array_almost_equal(f, 2.0 * simple_x[:, 0:1])
        np.testing.assert_array_almost_equal(df_dc, simple_x[:, 0:1])


# ------------------------------------------------------------------ #
#  sklearn interface                                                  #
# ------------------------------------------------------------------ #


class TestSklearnInterface:
    def test_predict(self, x0_plus_c0, simple_x):
        pred = x0_plus_c0.predict(simple_x)
        assert pred.ndim == 1
        np.testing.assert_array_almost_equal(pred, simple_x[:, 0] + 10.0)

    def test_score(self, x0_plus_c0, simple_x):
        y = simple_x[:, 0] + 10.0
        score = x0_plus_c0.score(simple_x, y, metric="mse")
        assert score == pytest.approx(0.0, abs=1e-10)

    def test_score_mae(self, x0_plus_c0, simple_x):
        y = simple_x[:, 0] + 10.0 + 1.0  # off by 1
        score = x0_plus_c0.score(simple_x, y, metric="mae")
        assert score == pytest.approx(1.0)

    def test_fit_optimizes_constants(self, simple_x):
        expr = AGraphExpression(equation="X0 * 1.0")
        y = 3.0 * simple_x[:, 0]
        expr.fit(simple_x, y)
        assert expr.constants[0] == pytest.approx(3.0, abs=0.1)

    def test_fit_returns_self(self, simple_x):
        expr = AGraphExpression(equation="X0 + 1.0")
        result = expr.fit(simple_x, simple_x[:, 0])
        assert result is expr

    def test_fit_no_constants_is_noop(self, simple_x):
        expr = AGraphExpression(equation="X0")
        expr.fit(simple_x, simple_x[:, 0])  # should not raise

    def test_score_bic(self, simple_x):
        expr = AGraphExpression(equation="X0 * 1.0")
        y = 3.0 * simple_x[:, 0]
        expr.fit(simple_x, y)
        bic = expr.score(simple_x, y, metric="bic")
        assert np.isfinite(bic)

    def test_score_bic_no_constants(self, simple_x):
        """BIC with 0 explicit constants still has k=1 (noise variance counts)."""
        expr = AGraphExpression(equation="X0")
        y = simple_x[:, 0]
        bic = expr.score(simple_x, y, metric="bic")
        # Perfect fit → MSE ≈ 0 → log_likelihood → +inf → BIC → -inf
        assert bic < 0

    def test_score_laplace_nmll(self, simple_x):
        expr = AGraphExpression(equation="X0 * 1.0")
        y = 3.0 * simple_x[:, 0]
        expr.fit(simple_x, y)
        nmll = expr.score(simple_x, y, metric="laplace_nmll")
        assert np.isfinite(nmll)

    def test_score_laplace_nmll_no_constants(self, simple_x):
        """Laplace NMLL with 0 constants simplifies to Gaussian LL."""
        expr = AGraphExpression(equation="X0")
        y = simple_x[:, 0] + 1.0  # non-zero residuals
        nmll = expr.score(simple_x, y, metric="laplace_nmll")
        assert np.isfinite(nmll)


# ------------------------------------------------------------------ #
#  sklearn is_fitted                                                  #
# ------------------------------------------------------------------ #


class TestSklearnIsFitted:
    def test_expression_with_constants_is_not_fitted(self):
        expr = AGraphExpression(equation="X0 + 1.0")
        assert not expr.__sklearn_is_fitted__()

    def test_expression_without_constants_is_fitted(self):
        expr = AGraphExpression(equation="X0")
        assert expr.__sklearn_is_fitted__()

    def test_fitted_after_fit(self, simple_x):
        expr = AGraphExpression(equation="X0 * 1.0")
        y = 2.0 * simple_x[:, 0]
        expr.fit(simple_x, y)
        assert expr.__sklearn_is_fitted__()

    def test_check_is_fitted_raises_when_not_fitted(self):
        expr = AGraphExpression(equation="X0 + 1.0")
        with pytest.raises(NotFittedError):
            check_is_fitted(expr)

    def test_check_is_fitted_passes_after_fit(self, simple_x):
        expr = AGraphExpression(equation="X0 * 1.0")
        expr.fit(simple_x, 2.0 * simple_x[:, 0])
        check_is_fitted(expr)  # should not raise

    def test_not_fitted_after_modification(self, simple_x):
        expr = AGraphExpression(equation="X0 + 1.0")
        y = 2.0 * simple_x[:, 0]
        expr.fit(simple_x, y)
        assert expr.__sklearn_is_fitted__()
        _ = expr.mutable_command_array
        assert not expr.__sklearn_is_fitted__()


# ------------------------------------------------------------------ #
#  Formatting                                                         #
# ------------------------------------------------------------------ #


class TestFormatting:
    def test_str_uses_console_format(self, x0_plus_c0):
        s = str(x0_plus_c0)
        assert "X0" in s
        # Console format: no Symbol() wrappers
        assert "Symbol" not in s

    def test_console_property(self, x0_plus_c0):
        assert x0_plus_c0.console == str(x0_plus_c0)

    def test_latex_property(self, x0_plus_c0):
        latex = x0_plus_c0.latex
        assert isinstance(latex, str)
        assert "X0" in latex

    def test_sympy_property(self, x0_plus_c0):
        import sympy

        expr = x0_plus_c0.sympy
        assert isinstance(expr, sympy.Basic)

    def test_variables_have_no_underscore(self):
        expr = AGraphExpression(equation="X0 + X1")
        s = str(expr)
        assert "X0" in s
        assert "X1" in s
        assert "X_0" not in s
        assert "X_1" not in s


# ------------------------------------------------------------------ #
#  Hash / equality                                                    #
# ------------------------------------------------------------------ #


class TestHashEquality:
    def test_same_structure_equal(self):
        e1 = AGraphExpression(equation="X0 + X1")
        e2 = AGraphExpression(equation="X0 + X1")
        assert e1 == e2
        assert hash(e1) == hash(e2)

    def test_different_structure_not_equal(self):
        e1 = AGraphExpression(equation="X0 + X1")
        e2 = AGraphExpression(equation="X0 * X1")
        assert e1 != e2

    def test_hash_is_int(self, x0_plus_c0):
        assert isinstance(hash(x0_plus_c0), int)

    def test_not_equal_to_non_expression(self, x0_plus_c0):
        assert x0_plus_c0 != "not an expression"


# ------------------------------------------------------------------ #
#  Copy / distance                                                    #
# ------------------------------------------------------------------ #


class TestCopyDistance:
    def test_copy_produces_independent_object(self, x0_plus_c0, simple_x):
        copied = x0_plus_c0.copy()
        assert copied is not x0_plus_c0
        np.testing.assert_array_equal(copied.command_array, x0_plus_c0.command_array)
        copied.constants = (999.0,)
        assert x0_plus_c0.constants[0] != 999.0

    def test_deepcopy(self, x0_plus_c0):
        copied = copy.deepcopy(x0_plus_c0)
        assert copied is not x0_plus_c0
        np.testing.assert_array_equal(copied.command_array, x0_plus_c0.command_array)

    def test_distance_same(self, x0_plus_c0):
        assert x0_plus_c0.distance(x0_plus_c0) == 0

    def test_distance_different(self):
        e1 = AGraphExpression(equation="X0 + X1")
        e2 = AGraphExpression(equation="X0 * X1")
        assert e1.distance(e2) > 0


# ------------------------------------------------------------------ #
#  Serialization                                                      #
# ------------------------------------------------------------------ #


class TestSerialization:
    def test_pickle_roundtrip(self, x0_plus_c0, simple_x):
        data = pickle.dumps(x0_plus_c0)
        restored = pickle.loads(data)
        np.testing.assert_array_almost_equal(
            restored._evaluate(simple_x),
            x0_plus_c0._evaluate(simple_x),
        )


# ------------------------------------------------------------------ #
#  Modification tracking                                              #
# ------------------------------------------------------------------ #


class TestModificationTracking:
    def test_mutable_access_marks_not_fitted(self, simple_x):
        expr = AGraphExpression(equation="X0 + 1.0")
        y = 2.0 * simple_x[:, 0]
        expr.fit(simple_x, y)
        assert expr.__sklearn_is_fitted__()
        _ = expr.mutable_command_array
        assert not expr.__sklearn_is_fitted__()

    def test_command_array_setter_marks_not_fitted(self, simple_x):
        expr = AGraphExpression(equation="X0 + 1.0")
        y = 2.0 * simple_x[:, 0]
        expr.fit(simple_x, y)
        assert expr.__sklearn_is_fitted__()
        # Replace with an array that still contains a constant
        new_cmd = np.array(
            [[VARIABLE, 0, 0], [CONSTANT, 0, 0], [ADDITION, 0, 1]],
            dtype=np.uint8,
        )
        expr.command_array = new_cmd
        assert not expr.__sklearn_is_fitted__()

    def test_modification_resets_hash(self, x0_plus_c0):
        h1 = hash(x0_plus_c0)
        _ = x0_plus_c0.mutable_command_array
        assert x0_plus_c0._hash is None
