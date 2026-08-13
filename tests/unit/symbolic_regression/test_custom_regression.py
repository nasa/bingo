# pylint: disable=missing-docstring
import numpy as np

from bingo.expressions.agraph.evolvable import EvolvableExpression
from bingo.expressions.agraph.pyagraph.expression import AGraphExpression
from bingo.expressions.agraph.pyagraph.operators import CONSTANT, VARIABLE
from bingo.symbolic_regression import (
    CustomRegression,
    FitResult,
    ObjectiveData,
    ScipyFitter,
    explicit_residuals,
    expression_loss,
)


def _constant_expression(value=0.0):
    expression = AGraphExpression()
    expression.raw_command_array = np.array([[CONSTANT, 0, 0]], dtype=np.uint8)
    expression.raw_constants = (value,)
    return expression


def test_custom_regression_commits_fitter_constants_then_ranks_expression():
    expression = _constant_expression()
    data = ObjectiveData(np.arange(3.0), np.full(3, 2.0))

    def fitter(fitted_expression, objective_data, fitting_measure):
        assert fitted_expression is expression
        assert objective_data is data
        assert fitting_measure(fitted_expression, objective_data, [2.0]) == 0.0
        return FitResult([2.0], success=True)

    def measure(fitted_expression, objective_data, constants):
        return constants[0] - objective_data.arrays[1][0]

    def loss(fitted_expression, objective_data):
        return np.mean((fitted_expression.predict(objective_data.arrays[0]) - 2.0) ** 2)

    objective = CustomRegression(data, fitter, measure, loss)

    assert objective(EvolvableExpression(expression)) == 0.0
    assert expression.constants == (2.0,)
    assert expression.is_fitted


def test_explicit_measure_evaluates_candidate_constants_without_mutating_expression():
    expression = _constant_expression(1.0)
    data = ObjectiveData(np.arange(3.0).reshape(-1, 1), np.full(3, 2.0))

    residuals = explicit_residuals()(expression, data, [2.0])

    np.testing.assert_array_equal(residuals, np.zeros(3))
    assert expression.constants == (1.0,)
    assert not expression.is_fitted


def test_scipy_root_fitter_uses_residual_measure_to_fit_constants():
    expression = _constant_expression()
    data = ObjectiveData(np.arange(3.0).reshape(-1, 1), np.full(3, 2.0))
    objective = CustomRegression(
        data,
        ScipyFitter("lm"),
        explicit_residuals(),
        lambda fitted_expression, objective_data: fitted_expression.loss(
            objective_data.arrays[0], objective_data.arrays[1]
        ),
    )

    assert objective(EvolvableExpression(expression)) == 0.0
    assert expression.constants == (2.0,)


def test_scipy_minimize_fitter_uses_scalar_measure_to_fit_constants():
    expression = _constant_expression()
    data = ObjectiveData(np.arange(3.0).reshape(-1, 1), np.full(3, 2.0))
    objective = CustomRegression(
        data,
        ScipyFitter("BFGS"),
        expression_loss(),
        lambda fitted_expression, objective_data: fitted_expression.loss(
            objective_data.arrays[0], objective_data.arrays[1]
        ),
    )

    assert objective(EvolvableExpression(expression)) < 1e-10
    assert expression.is_fitted


def test_invalid_fit_result_leaves_expression_unfitted():
    expression = _constant_expression()
    data = ObjectiveData(np.arange(3.0), np.full(3, 2.0))
    objective = CustomRegression(
        data,
        lambda *_: FitResult([np.nan], success=False),
        lambda *_: 0.0,
        lambda *_: 0.0,
    )

    with np.testing.assert_raises_regex(ValueError, "fitted constants must be finite"):
        objective(EvolvableExpression(expression))
    assert expression.constants == (0.0,)
    assert not expression.is_fitted


def test_invalid_fit_result_shape_leaves_expression_unfitted():
    expression = _constant_expression()
    objective = CustomRegression(
        ObjectiveData(np.arange(3.0)),
        lambda *_: FitResult([[2.0]], success=True),
        lambda *_: 0.0,
        lambda *_: 0.0,
    )

    with np.testing.assert_raises_regex(ValueError, "one-dimensional"):
        objective(EvolvableExpression(expression))
    assert expression.constants == (0.0,)
    assert not expression.is_fitted


def test_loss_exception_leaves_newly_fitted_expression_unfitted():
    expression = _constant_expression()
    objective = CustomRegression(
        ObjectiveData(np.arange(3.0)),
        lambda *_: FitResult([2.0], success=True),
        lambda *_: 0.0,
        lambda *_: (_ for _ in ()).throw(RuntimeError("loss failed")),
    )

    with np.testing.assert_raises_regex(RuntimeError, "loss failed"):
        objective(EvolvableExpression(expression))
    assert expression.constants == (0.0,)
    assert not expression.is_fitted


def test_nonconverged_finite_fit_result_commits_best_constants():
    expression = _constant_expression()
    objective = CustomRegression(
        ObjectiveData(np.arange(3.0)),
        lambda *_: FitResult([3.0], success=False, message="maximum iterations"),
        lambda *_: 0.0,
        lambda fitted_expression, _: fitted_expression.constants[0],
    )

    assert objective(EvolvableExpression(expression)) == 3.0
    assert expression.constants == (3.0,)
    assert expression.is_fitted


def test_constant_free_expression_skips_fitter_and_ranks_directly():
    expression = AGraphExpression()
    expression.raw_command_array = np.array([[VARIABLE, 0, 0]], dtype=np.uint8)
    objective = CustomRegression(
        ObjectiveData(np.arange(3.0).reshape(-1, 1)),
        lambda *_: (_ for _ in ()).throw(AssertionError("fitter should not run")),
        lambda *_: 0.0,
        lambda fitted_expression, data: fitted_expression.predict(data.arrays[0]).mean(),
    )

    assert objective(EvolvableExpression(expression)) == 1.0


def test_scipy_fitter_rejects_incompatible_measure_contract():
    expression = _constant_expression()
    data = ObjectiveData(np.arange(3.0).reshape(-1, 1), np.full(3, 2.0))

    with np.testing.assert_raises_regex(TypeError, "residual fitting measure"):
        ScipyFitter("lm")(expression, data, expression_loss())
