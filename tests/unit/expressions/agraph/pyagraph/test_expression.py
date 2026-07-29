"""Tests for bingo.expressions.agraph.expression"""

import copy
import pickle

import numpy as np
import pytest
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from bingo.expressions.agraph.pyagraph.expression import AGraphExpression
from bingo.expressions.agraph.pyagraph.operators import (
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
        # X0 + 10.0 => 3 commands (X0, C0, ADD)
        assert x0_plus_c0.complexity == 3

    def test_tree_complexity_equals_complexity_no_dag_reuse(self, x0_plus_c0):
        # No shared sub-expressions → tree_complexity == complexity
        assert x0_plus_c0.tree_complexity == x0_plus_c0.complexity

    def test_tree_complexity_greater_than_complexity_with_dag_reuse(
        self, dag_with_shared_subgraph
    ):
        # (X0 + C0) * (X0 + C0) reuses row 2
        # DAG complexity = 4, tree complexity = 7
        assert dag_with_shared_subgraph.tree_complexity > dag_with_shared_subgraph.complexity

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
        # that references index 0 of _raw_integers (which is 7), the
        # value should still be 7.  This verifies that integers are fixed
        # from the source, not fabricated from the simplified command array.
        _ = expr.mutable_raw_command_array
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

    def test_gradient_returns_predictions_and_input_derivative(
        self, x0_plus_c0, simple_x
    ):
        f, df_dx = x0_plus_c0.gradient(simple_x)
        np.testing.assert_array_almost_equal(f, simple_x[:, 0] + 10.0)
        assert f.ndim == 1
        # d(X0 + 10)/dX0 = 1, d/dX1 = 0
        expected = np.zeros_like(simple_x)
        expected[:, 0] = 1.0
        np.testing.assert_array_almost_equal(df_dx, expected)

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

    def test_fit_tolerance_is_keyword_only(self, simple_x):
        """``tolerance`` cannot be passed positionally."""
        expr = AGraphExpression(equation="X0 * 1.0")
        y = 3.0 * simple_x[:, 0]
        with pytest.raises(TypeError):
            expr.fit(simple_x, y, 1e-6)

    def test_fit_rejects_legacy_metric_kwarg(self, simple_x):
        """The legacy ``metric`` fit option is absent."""
        expr = AGraphExpression(equation="X0 * 1.0")
        y = 3.0 * simple_x[:, 0]
        with pytest.raises(TypeError):
            expr.fit(simple_x, y, metric="mse")

    def test_fit_rejects_arbitrary_solver_options(self, simple_x):
        """Arbitrary public solver options are not accepted."""
        expr = AGraphExpression(equation="X0 * 1.0")
        y = 3.0 * simple_x[:, 0]
        with pytest.raises(TypeError):
            expr.fit(simple_x, y, options={"maxiter": 5})


class TestExplicitScore:
    """Higher-is-better score with the selected vocabulary."""

    def test_r2_is_default_and_perfect_fit_is_one(self, x0_plus_c0, simple_x):
        y = simple_x[:, 0] + 10.0
        assert x0_plus_c0.score(simple_x, y) == pytest.approx(1.0)

    def test_r2_worse_fit_is_lower(self, x0_plus_c0, simple_x):
        perfect = simple_x[:, 0] + 10.0
        noisy = perfect + np.array([1.0, -2.0, 3.0])
        assert x0_plus_c0.score(simple_x, noisy) < x0_plus_c0.score(simple_x, perfect)

    def test_laplace_nmll_score(self, simple_x):
        expr = AGraphExpression(equation="X0 * 1.0")
        y = 3.0 * simple_x[:, 0]
        expr.fit(simple_x, y)
        assert np.isfinite(expr.score(simple_x, y, kind="laplace_nmll"))

    def test_score_rejects_loss_only_kind(self, x0_plus_c0, simple_x):
        y = simple_x[:, 0] + 10.0
        with pytest.raises(ValueError):
            x0_plus_c0.score(simple_x, y, kind="mse")


class TestExplicitLoss:
    """Lower-is-better loss with the selected vocabulary."""

    def test_mse_is_default_and_zero_for_perfect_fit(self, x0_plus_c0, simple_x):
        y = simple_x[:, 0] + 10.0
        assert x0_plus_c0.loss(simple_x, y) == pytest.approx(0.0, abs=1e-10)

    def test_mae(self, x0_plus_c0, simple_x):
        y = simple_x[:, 0] + 10.0 + 1.0  # off by 1
        assert x0_plus_c0.loss(simple_x, y, kind="mae") == pytest.approx(1.0)

    def test_rmse(self, x0_plus_c0, simple_x):
        y = simple_x[:, 0] + 10.0 + 2.0  # off by 2
        assert x0_plus_c0.loss(simple_x, y, kind="rmse") == pytest.approx(2.0)

    def test_relative_mse_zero_for_perfect_fit(self, x0_plus_c0, simple_x):
        y = simple_x[:, 0] + 10.0
        assert x0_plus_c0.loss(simple_x, y, kind="relative_mse") == pytest.approx(0.0)

    def test_relative_mse_rejects_zero_targets(self, simple_x):
        expr = AGraphExpression(equation="X0")
        y = simple_x[:, 0].copy()
        y[0] = 0.0
        with pytest.raises(ValueError):
            expr.loss(simple_x, y, kind="relative_mse")

    def test_correlation_loss_zero_for_perfectly_correlated(self, x0_plus_c0, simple_x):
        y = 5.0 * (simple_x[:, 0] + 10.0)  # perfectly correlated, different scale
        assert x0_plus_c0.loss(simple_x, y, kind="correlation") == pytest.approx(
            0.0, abs=1e-10
        )

    def test_laplace_nmll_loss_is_negated_score(self, simple_x):
        expr = AGraphExpression(equation="X0 * 1.0")
        y = 3.0 * simple_x[:, 0] + 0.5  # nonzero residuals
        expr.fit(simple_x, y)
        loss = expr.loss(simple_x, y, kind="laplace_nmll")
        score = expr.score(simple_x, y, kind="laplace_nmll")
        assert loss == pytest.approx(-score)

    def test_loss_rejects_unknown_kind(self, x0_plus_c0, simple_x):
        y = simple_x[:, 0] + 10.0
        with pytest.raises(ValueError):
            x0_plus_c0.loss(simple_x, y, kind="bogus")


# ------------------------------------------------------------------ #
#  sklearn is_fitted                                                  #
# ------------------------------------------------------------------ #


class TestFittedLifecycle:
    """Structure-only ``is_fitted`` lifecycle (CONTEXT.md ``Fitted expression``)."""

    def test_expression_with_constants_is_not_fitted(self):
        expr = AGraphExpression(equation="X0 + 1.0")
        assert not expr.is_fitted

    def test_expression_without_constants_is_fitted(self):
        expr = AGraphExpression(equation="X0")
        assert expr.is_fitted

    def test_sklearn_hook_matches_property(self):
        expr = AGraphExpression(equation="X0 + 1.0")
        assert expr.__sklearn_is_fitted__() == expr.is_fitted
        expr = AGraphExpression(equation="X0")
        assert expr.__sklearn_is_fitted__() == expr.is_fitted

    def test_fit_establishes_fitted(self, simple_x):
        expr = AGraphExpression(equation="X0 * 1.0")
        y = 2.0 * simple_x[:, 0]
        expr.fit(simple_x, y)
        assert expr.is_fitted

    def test_fit_implicit_establishes_fitted(self, simple_x):
        expr = AGraphExpression(equation="X0 * 1.0 + X1 * 1.0")
        dx_dt = np.ones_like(simple_x)
        assert not expr.is_fitted
        expr.fit_implicit(simple_x, dx_dt)
        assert expr.is_fitted

    def test_non_convergence_does_not_unset_fitted(self, simple_x):
        """A fitting attempt establishes fitted even if the solver fails."""
        expr = AGraphExpression(equation="X0 * 1.0")
        # NaN targets guarantee the solver cannot converge.
        y = np.full(simple_x.shape[0], np.nan)
        expr.fit(simple_x, y)
        assert expr.is_fitted

    def test_check_is_fitted_raises_when_not_fitted(self):
        expr = AGraphExpression(equation="X0 + 1.0")
        with pytest.raises(NotFittedError):
            check_is_fitted(expr)

    def test_check_is_fitted_passes_after_fit(self, simple_x):
        expr = AGraphExpression(equation="X0 * 1.0")
        expr.fit(simple_x, 2.0 * simple_x[:, 0])
        check_is_fitted(expr)  # should not raise

    def test_raw_command_mutation_unsets_fitted(self, simple_x):
        expr = AGraphExpression(equation="X0 + 1.0")
        expr.fit(simple_x, 2.0 * simple_x[:, 0])
        assert expr.is_fitted
        _ = expr.mutable_raw_command_array
        assert not expr.is_fitted

    def test_raw_command_setter_unsets_fitted(self, simple_x):
        expr = AGraphExpression(equation="X0 + 1.0")
        expr.fit(simple_x, 2.0 * simple_x[:, 0])
        assert expr.is_fitted
        new_cmd = np.array(
            [[VARIABLE, 0, 0], [CONSTANT, 0, 0], [ADDITION, 0, 1]],
            dtype=np.uint8,
        )
        expr.raw_command_array = new_cmd
        assert not expr.is_fitted

    def test_raw_constants_setter_unsets_fitted(self, simple_x):
        expr = AGraphExpression(equation="X0 + 1.0")
        expr.fit(simple_x, 2.0 * simple_x[:, 0])
        assert expr.is_fitted
        expr.raw_constants = (999.0,)
        assert not expr.is_fitted

    def test_direct_constant_assignment_does_not_establish_fitted(self):
        """Direct simplified-constant assignment cannot establish fitted."""
        expr = AGraphExpression(equation="X0 + 1.0")
        assert not expr.is_fitted
        expr.constants = (5.0,)
        assert not expr.is_fitted

    def test_direct_constant_assignment_preserves_fitted(self, simple_x):
        """Direct assignment on a fitted expression leaves it fitted."""
        expr = AGraphExpression(equation="X0 + 1.0")
        expr.fit(simple_x, simple_x[:, 0] + 3.0)
        assert expr.is_fitted
        expr.constants = (7.0,)
        assert expr.is_fitted


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

    def test_pickle_preserves_unfitted_state(self):
        expr = AGraphExpression(equation="X0 + 1.0")
        assert not expr.is_fitted
        restored = pickle.loads(pickle.dumps(expr))
        assert not restored.is_fitted

    def test_pickle_preserves_fitted_state_and_constants(self, simple_x):
        expr = AGraphExpression(equation="X0 * 1.0")
        y = 3.0 * simple_x[:, 0]
        expr.fit(simple_x, y)
        assert expr.is_fitted
        restored = pickle.loads(pickle.dumps(expr))
        assert restored.is_fitted
        assert restored.constants == pytest.approx(expr.constants)
        np.testing.assert_array_almost_equal(
            restored.predict(simple_x), expr.predict(simple_x)
        )

    def test_pickle_preserves_fitted_state_when_constants_unchanged(self, simple_x):
        """A no-constant expression that was 'fit' stays fitted (trivially)."""
        expr = AGraphExpression(equation="X0")
        expr.fit(simple_x, simple_x[:, 0])
        restored = pickle.loads(pickle.dumps(expr))
        assert restored.is_fitted

    def test_deepcopy_preserves_fitted_state(self, simple_x):
        expr = AGraphExpression(equation="X0 * 1.0")
        expr.fit(simple_x, 3.0 * simple_x[:, 0])
        copied = copy.deepcopy(expr)
        assert copied.is_fitted
        assert copied.constants == pytest.approx(expr.constants)

    def test_deepcopy_preserves_unfitted_state(self):
        expr = AGraphExpression(equation="X0 + 1.0")
        copied = copy.deepcopy(expr)
        assert not copied.is_fitted


# ------------------------------------------------------------------ #
#  Implicit regression contract                                       #
# ------------------------------------------------------------------ #


@pytest.fixture
def circle_data():
    """Points on the unit circle with their tangent trajectory derivatives.

    For f = X0^2 + X1^2, the input gradient (2*X0, 2*X1) is orthogonal to the
    tangent (-X1, X0), so the implicit residual is ~0.
    """
    theta = np.array([0.3, 0.9, 1.7, 2.5, 3.3, 4.1, 5.0])
    x = np.c_[np.cos(theta), np.sin(theta)]
    dx_dt = np.c_[-np.sin(theta), np.cos(theta)]
    return x, dx_dt


class TestImplicitContract:
    def test_implicit_loss_low_for_true_relationship(self, circle_data):
        x, dx_dt = circle_data
        expr = AGraphExpression(equation="X0 * X0 + X1 * X1")
        assert expr.implicit_loss(x, dx_dt) == pytest.approx(0.0, abs=1e-8)

    def test_implicit_loss_higher_for_wrong_relationship(self, circle_data):
        x, dx_dt = circle_data
        good = AGraphExpression(equation="X0 * X0 + X1 * X1")
        bad = AGraphExpression(equation="X0 + X1")
        assert bad.implicit_loss(x, dx_dt) > good.implicit_loss(x, dx_dt)

    def test_implicit_score_is_negated_loss(self, circle_data):
        x, dx_dt = circle_data
        expr = AGraphExpression(equation="X0 + X1")
        loss = expr.implicit_loss(x, dx_dt)
        assert expr.implicit_score(x, dx_dt) == pytest.approx(-loss)

    def test_required_params_guard_infinite_loss(self, circle_data):
        """An expression using too few derivative components fails the guard."""
        x, dx_dt = circle_data
        expr = AGraphExpression(equation="X0")  # only one component nonzero
        assert expr.implicit_loss(x, dx_dt, required_params=2) == float("inf")

    def test_required_params_guard_passes_when_enough(self, circle_data):
        x, dx_dt = circle_data
        expr = AGraphExpression(equation="X0 * X0 + X1 * X1")
        assert np.isfinite(expr.implicit_loss(x, dx_dt, required_params=2))

    def test_required_params_guard_infinite_score(self, circle_data):
        x, dx_dt = circle_data
        expr = AGraphExpression(equation="X0")
        assert expr.implicit_score(x, dx_dt, required_params=2) == float("-inf")

    def test_fit_implicit_returns_self(self, circle_data):
        x, dx_dt = circle_data
        expr = AGraphExpression(equation="X0 * X0 + 1.0 * X1 * X1")
        assert expr.fit_implicit(x, dx_dt) is expr

    def test_fit_implicit_improves_loss(self, circle_data):
        x, dx_dt = circle_data
        expr = AGraphExpression(equation="X0 * X0 + 5.0 * X1 * X1")
        before = expr.implicit_loss(x, dx_dt)
        expr.fit_implicit(x, dx_dt)
        after = expr.implicit_loss(x, dx_dt)
        assert after <= before + 1e-12

    def test_fit_implicit_tolerance_is_keyword_only(self, circle_data):
        x, dx_dt = circle_data
        expr = AGraphExpression(equation="X0 * X0 + 1.0 * X1 * X1")
        with pytest.raises(TypeError):
            expr.fit_implicit(x, dx_dt, 1e-6)


# ------------------------------------------------------------------ #
#  Non-finite normalization                                          #
# ------------------------------------------------------------------ #


class TestNonFiniteNormalization:
    """Non-finite public evaluation → +inf loss, -inf score."""

    def test_explicit_loss_infinite_on_nonfinite_prediction(self):
        expr = AGraphExpression(equation="1.0 / X0")
        x = np.array([[0.0]])
        assert expr.loss(x, [1.0]) == float("inf")

    def test_explicit_score_negative_infinite_on_nonfinite_prediction(self):
        expr = AGraphExpression(equation="1.0 / X0")
        x = np.array([[0.0]])
        assert expr.score(x, [1.0]) == float("-inf")

    def test_explicit_loss_infinite_for_all_kinds(self):
        expr = AGraphExpression(equation="1.0 / X0")
        x = np.array([[0.0]])
        for kind in ("mse", "mae", "rmse", "correlation", "laplace_nmll"):
            assert expr.loss(x, [1.0], kind=kind) == float("inf")

    def test_implicit_loss_infinite_on_nonfinite_gradient(self):
        # sqrt gradient diverges at 0 -> non-finite df/dx
        expr = AGraphExpression(equation="sqrt(X0)")
        x = np.array([[0.0]])
        dx_dt = np.array([[1.0]])
        assert expr.implicit_loss(x, dx_dt) == float("inf")

    def test_implicit_score_negative_infinite_on_nonfinite_gradient(self):
        expr = AGraphExpression(equation="sqrt(X0)")
        x = np.array([[0.0]])
        dx_dt = np.array([[1.0]])
        assert expr.implicit_score(x, dx_dt) == float("-inf")


# ------------------------------------------------------------------ #
#  Modification tracking                                              #
# ------------------------------------------------------------------ #


class TestModificationTracking:
    def test_modification_resets_hash(self, x0_plus_c0):
        hash(x0_plus_c0)
        _ = x0_plus_c0.mutable_raw_command_array
        assert x0_plus_c0._hash is None


# ------------------------------------------------------------------ #
#  Constant mapping & propagation                                     #
# ------------------------------------------------------------------ #


class TestConstantMapping:
    """Tests for constant_mapping property."""

    def test_mapping_exists_after_access(self):
        expr = AGraphExpression(equation="X0 + 1.0")
        mapping = expr.constant_mapping
        assert isinstance(mapping, tuple)
        assert len(mapping) == len(expr.constants)

    def test_mapping_identity_for_simple_expression(self):
        """For X0 + C0, reduce produces a trivial 1:1 mapping."""
        expr = AGraphExpression(equation="X0 + 1.0")
        mapping = expr.constant_mapping
        assert mapping == (0,)

    def test_mapping_for_multiple_constants(self):
        """X0 * C0 + C1 should have two entries in constant_mapping."""
        expr = AGraphExpression()
        expr.raw_command_array = np.array(
            [
                [VARIABLE, 0, 0],
                [CONSTANT, 0, 0],
                [MULTIPLICATION, 0, 1],
                [CONSTANT, 1, 1],
                [ADDITION, 2, 3],
            ],
            dtype=np.uint8,
        )
        expr.raw_constants = (2.0, 3.0)
        mapping = expr.constant_mapping
        assert len(mapping) == 2

    def test_mapping_skips_dead_constant(self):
        """If a raw constant is unused, it should not appear in mapping."""
        expr = AGraphExpression(simplification="reduce")
        # C0 used, C1 dead (unused), C2 used
        expr.raw_command_array = np.array(
            [
                [CONSTANT, 0, 0],
                [CONSTANT, 1, 1],  # dead
                [CONSTANT, 2, 2],
                [ADDITION, 0, 2],
            ],
            dtype=np.uint8,
        )
        expr.raw_constants = (10.0, 99.0, 20.0)
        mapping = expr.constant_mapping
        # reduced[0] → raw[0], reduced[1] → raw[2]; raw[1] is dead
        assert 1 not in mapping
        assert len(mapping) == 2

    def test_mapping_values_index_into_raw(self):
        """constant_mapping[i] should be a valid raw_constants index."""
        expr = AGraphExpression(simplification="reduce")
        expr.raw_command_array = np.array(
            [
                [CONSTANT, 0, 0],
                [CONSTANT, 1, 1],  # dead
                [CONSTANT, 2, 2],
                [ADDITION, 0, 2],
            ],
            dtype=np.uint8,
        )
        expr.raw_constants = (10.0, 99.0, 20.0)
        mapping = expr.constant_mapping
        for i, raw_idx in enumerate(mapping):
            assert expr.constants[i] == expr.raw_constants[raw_idx]


class TestPropagateConstants:
    """Tests for propagate_constants toggle."""

    def test_default_is_false(self):
        expr = AGraphExpression(equation="X0 + 1.0")
        assert expr.propagate_constants is False

    def test_can_enable_via_constructor(self):
        expr = AGraphExpression(equation="X0 + 1.0", propagate_constants=True)
        assert expr.propagate_constants is True

    def test_can_toggle_via_setter(self):
        expr = AGraphExpression(equation="X0 + 1.0")
        expr.propagate_constants = True
        assert expr.propagate_constants is True

    def test_disabled_does_not_propagate(self):
        """With propagate_constants=False, setting constants leaves raw untouched."""
        expr = AGraphExpression(equation="X0 + 1.0")
        original_raw = expr.raw_constants
        expr.constants = (99.0,)
        assert expr.raw_constants == original_raw

    def test_enabled_propagates_to_raw(self):
        """With propagate_constants=True, setting constants updates raw."""
        expr = AGraphExpression(equation="X0 + 1.0", propagate_constants=True)
        _ = expr.constants  # trigger initial simplification
        expr.constants = (42.0,)
        # The mapped raw constant should now be 42.0
        mapping = expr.constant_mapping
        assert expr.raw_constants[mapping[0]] == 42.0

    def test_propagation_with_dead_constants(self):
        """Propagation correctly targets the mapped raw index,
        leaving dead constants untouched."""
        expr = AGraphExpression(propagate_constants=True, simplification="reduce")
        # C0 used, C1 dead, C2 used
        expr.raw_command_array = np.array(
            [
                [CONSTANT, 0, 0],
                [CONSTANT, 1, 1],  # dead
                [CONSTANT, 2, 2],
                [ADDITION, 0, 2],
            ],
            dtype=np.uint8,
        )
        expr.raw_constants = (10.0, 99.0, 20.0)
        # Force update
        _ = expr.constants
        # Set new simplified constants
        expr.constants = (100.0, 200.0)
        # raw[0] should be 100.0 (from simplified[0])
        # raw[1] should still be 99.0 (dead, untouched)
        # raw[2] should be 200.0 (from simplified[1])
        assert expr.raw_constants[0] == 100.0
        assert expr.raw_constants[1] == 99.0
        assert expr.raw_constants[2] == 200.0

    def test_propagation_does_not_retrigger_simplification(self):
        """Propagation back to raw should NOT mark the expression modified."""
        expr = AGraphExpression(equation="X0 + 1.0", propagate_constants=True)
        _ = expr.constants  # trigger update, clear _modified
        assert expr._modified is False
        expr.constants = (42.0,)
        assert expr._modified is False

    def test_fit_propagates_when_enabled(self):
        """After fit() with propagation enabled, raw_constants match fitted."""
        expr = AGraphExpression(equation="X0 * 1.0", propagate_constants=True)
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = 3.0 * X[:, 0]
        expr.fit(X, y)
        # fitted constant should be ~3.0
        assert expr.constants[0] == pytest.approx(3.0, abs=0.1)
        # raw should also reflect the fitted value
        mapping = expr.constant_mapping
        assert expr.raw_constants[mapping[0]] == pytest.approx(3.0, abs=0.1)

    def test_fit_does_not_propagate_when_disabled(self):
        """After fit() with propagation disabled, raw_constants are unchanged."""
        expr = AGraphExpression(equation="X0 * 1.0", propagate_constants=False)
        original_raw = expr.raw_constants
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = 3.0 * X[:, 0]
        expr.fit(X, y)
        assert expr.raw_constants == original_raw

    def test_deepcopy_preserves_propagation_flag(self):
        """deepcopy should carry over the propagate_constants flag."""
        expr = AGraphExpression(equation="X0 + 1.0", propagate_constants=True)
        copied = copy.deepcopy(expr)
        assert copied.propagate_constants is True

    def test_pickle_preserves_propagation_flag(self):
        """pickle roundtrip should preserve propagate_constants flag."""
        expr = AGraphExpression(equation="X0 + 1.0", propagate_constants=True)
        data = pickle.dumps(expr)
        restored = pickle.loads(data)
        assert restored.propagate_constants is True

    def test_pickle_backward_compat_defaults_false(self):
        """Unpickling old data (no _propagate_constants) defaults to False."""
        expr = AGraphExpression(equation="X0 + 1.0")
        # Simulate an old-format state dict (only essential keys).
        state = expr.__getstate__()
        state.pop("_propagate_constants", None)
        state.pop("_constant_mapping", None)
        new = AGraphExpression.__new__(AGraphExpression)
        new.__setstate__(state)
        assert new.propagate_constants is False
        assert new._constant_mapping == ()


class TestPromoteSimplificationMapping:
    """Tests for constant_mapping after promote_simplification()."""

    def test_promote_resets_mapping_to_identity(self):
        """After promotion, raw == simplified, so mapping should be identity."""
        expr = AGraphExpression(equation="X0 + 1.0")
        _ = expr.constants  # ensure update has run
        expr.promote_simplification()
        mapping = expr.constant_mapping
        assert mapping == tuple(range(len(expr.constants)))

    def test_promote_then_propagate(self):
        """After promotion with propagation enabled, constants = raw constants."""
        expr = AGraphExpression(equation="X0 + 1.0", propagate_constants=True)
        _ = expr.constants
        expr.promote_simplification()
        expr.constants = (77.0,)
        assert expr.raw_constants[0] == 77.0


# ------------------------------------------------------------------ #
#  Operator counts                                                    #
# ------------------------------------------------------------------ #


@pytest.fixture
def dag_with_shared_subgraph():
    """Expression where a node is shared: (X0 + C0) * (X0 + C0).

    DAG rows:
        0: X0
        1: C0
        2: X0 + C0
        3: (row2) * (row2)   <-- reuses row 2

    After simplification the DAG may be reduced, so we use
    ``simplification="reduce"`` to keep the structure predictable.
    """
    expr = AGraphExpression(simplification="reduce")
    expr.raw_command_array = np.array(
        [
            [VARIABLE, 0, 0],
            [CONSTANT, 0, 0],
            [ADDITION, 0, 1],
            [MULTIPLICATION, 2, 2],
        ],
        dtype=np.uint8,
    )
    expr.raw_constants = (3.0,)
    return expr


class TestGetOperatorCounts:
    """Tests for AGraphExpression.get_operator_counts."""

    # -- basic counting ------------------------------------------------- #

    def test_tree_include_simple(self, x0_plus_c0):
        counts = x0_plus_c0.get_operator_counts(tree=True, terminals="include")
        assert counts[ADDITION] == 1
        assert VARIABLE in counts
        assert CONSTANT in counts

    def test_tree_exclude_terminals(self, x0_plus_c0):
        counts = x0_plus_c0.get_operator_counts(tree=True, terminals="exclude")
        assert VARIABLE not in counts
        assert CONSTANT not in counts
        assert counts.get(ADDITION, 0) == 1

    def test_tree_combine_terminals(self, x0_plus_c0):
        counts = x0_plus_c0.get_operator_counts(tree=True, terminals="combine")
        # All terminals folded under VARIABLE key
        assert CONSTANT not in counts
        assert INTEGER not in counts
        # There should be at least 2 terminal nodes (X0, C0)
        assert counts[VARIABLE] >= 2

    def test_dag_include_simple(self, x0_plus_c0):
        counts = x0_plus_c0.get_operator_counts(tree=False, terminals="include")
        assert counts[ADDITION] == 1
        assert VARIABLE in counts

    def test_dag_exclude_terminals(self, x0_plus_c0):
        counts = x0_plus_c0.get_operator_counts(tree=False, terminals="exclude")
        assert VARIABLE not in counts
        assert CONSTANT not in counts

    def test_dag_combine_terminals(self, x0_plus_c0):
        counts = x0_plus_c0.get_operator_counts(tree=False, terminals="combine")
        assert CONSTANT not in counts
        assert VARIABLE in counts

    # -- tree vs DAG difference with shared sub-graph -------------------- #

    def test_tree_counts_shared_subgraph_twice(self, dag_with_shared_subgraph):
        """Tree traversal should count the shared sub-tree twice."""
        counts = dag_with_shared_subgraph.get_operator_counts(
            tree=True, terminals="include"
        )
        # The ADDITION node appears in both branches of MULTIPLICATION
        assert counts[ADDITION] == 2
        assert counts[VARIABLE] == 2
        assert counts[CONSTANT] == 2
        assert counts[MULTIPLICATION] == 1

    def test_dag_counts_shared_subgraph_once(self, dag_with_shared_subgraph):
        """DAG traversal counts each row exactly once."""
        counts = dag_with_shared_subgraph.get_operator_counts(
            tree=False, terminals="include"
        )
        assert counts[ADDITION] == 1
        assert counts[VARIABLE] == 1
        assert counts[CONSTANT] == 1
        assert counts[MULTIPLICATION] == 1

    # -- empty expression ------------------------------------------------ #

    def test_empty_expression(self):
        expr = AGraphExpression()
        assert expr.get_operator_counts(tree=True) == {}
        assert expr.get_operator_counts(tree=False) == {}

    # -- unary operator -------------------------------------------------- #

    def test_unary_operator(self, sin_x0):
        counts = sin_x0.get_operator_counts(tree=True, terminals="include")
        assert counts[SIN] == 1
        assert counts[VARIABLE] == 1

    # -- manual multi-operator expression -------------------------------- #

    def test_manual_multi_op(self):
        """X0 + sin(X1) - sqrt(C0)  →  SUB(ADD(X0, SIN(X1)), SQRT(C0))"""
        expr = AGraphExpression(simplification="reduce")
        expr.raw_command_array = np.array(
            [
                [VARIABLE, 0, 0],  # row 0: X0
                [VARIABLE, 1, 0],  # row 1: X1
                [SIN, 1, 0],  # row 2: sin(X1)
                [ADDITION, 0, 2],  # row 3: X0 + sin(X1)
                [CONSTANT, 0, 0],  # row 4: C0
                [SQRT, 4, 0],  # row 5: sqrt(C0)
                [SUBTRACTION, 3, 5],  # row 6: (row3) - (row5)
            ],
            dtype=np.uint8,
        )
        expr.raw_constants = (4.0,)

        tree_counts = expr.get_operator_counts(tree=True, terminals="include")
        assert tree_counts[SUBTRACTION] == 1
        assert tree_counts[ADDITION] == 1
        assert tree_counts[SIN] == 1
        assert tree_counts[SQRT] == 1
        assert tree_counts[VARIABLE] == 2
        assert tree_counts[CONSTANT] == 1

        dag_counts = expr.get_operator_counts(tree=False, terminals="include")
        assert dag_counts[SUBTRACTION] == 1
        assert dag_counts[ADDITION] == 1
        assert dag_counts[SIN] == 1
        assert dag_counts[SQRT] == 1
        assert dag_counts[VARIABLE] == 2
        assert dag_counts[CONSTANT] == 1

    # -- defaults -------------------------------------------------------- #

    def test_default_is_tree_exclude(self, x0_plus_c0):
        default = x0_plus_c0.get_operator_counts()
        explicit = x0_plus_c0.get_operator_counts(tree=True, terminals="exclude")
        assert default == explicit

    # -- return type ----------------------------------------------------- #

    def test_returns_plain_dict(self, x0_plus_c0):
        counts = x0_plus_c0.get_operator_counts()
        assert type(counts) is dict
