# pylint: disable=missing-docstring
import numpy as np

from bingo.expressions.agraph.evolvable import EvolvableExpression
from bingo.expressions.agraph.pyagraph.expression import AGraphExpression
from bingo.expressions.agraph.pyagraph.operators import (
    ADDITION,
    CONSTANT,
    MULTIPLICATION,
    VARIABLE,
)
from bingo.symbolic_regression.implicit_regression import ImplicitRegression


def _linear_expression():
    expression = AGraphExpression()
    expression.raw_command_array = np.array(
        [
            [VARIABLE, 0, 0],
            [CONSTANT, 0, 0],
            [MULTIPLICATION, 0, 1],
        ],
        dtype=np.uint8,
    )
    expression.raw_constants = (1.0,)
    return expression


def test_implicit_objective_fits_and_returns_expression_loss(mocker):
    expression = _linear_expression()
    individual = EvolvableExpression(expression)
    x = np.arange(4.0).reshape(-1, 1)
    dx_dt = np.ones((4, 1))
    objective = ImplicitRegression(x, dx_dt)
    fit_implicit = mocker.spy(expression, "fit_implicit")

    loss = objective(individual)
    fit_implicit.assert_called_once()
    fit_x, fit_dx_dt = fit_implicit.call_args.args
    np.testing.assert_array_equal(fit_x, x)
    np.testing.assert_array_equal(fit_dx_dt, dx_dt)
    assert loss == expression.implicit_loss(x, dx_dt)
    assert expression.is_fitted


def test_implicit_objective_does_not_refit_an_unchanged_expression(mocker):
    expression = _linear_expression()
    individual = EvolvableExpression(expression)
    x = np.arange(4.0).reshape(-1, 1)
    objective = ImplicitRegression(x, np.ones((4, 1)))
    objective(individual)
    fit = mocker.spy(expression, "fit_implicit")

    objective(individual)

    fit.assert_not_called()


def test_implicit_objective_returns_infinite_loss_when_guard_fails():
    expression = _linear_expression()
    individual = EvolvableExpression(expression)
    x = np.arange(4.0).reshape(-1, 1)
    objective = ImplicitRegression(x, np.zeros((4, 1)), required_params=1)

    assert objective(individual) == np.inf


def test_implicit_objective_prefers_lower_loss():
    x = np.arange(8.0).reshape(-1, 2)
    low_loss = ImplicitRegression(x, np.tile([1.0, -1.0], (4, 1)))
    high_loss = ImplicitRegression(x, np.ones((4, 2)))
    expression = AGraphExpression()
    expression.raw_command_array = np.array(
        [[VARIABLE, 0, 0], [VARIABLE, 1, 1], [ADDITION, 0, 1]], dtype=np.uint8
    )

    assert low_loss(EvolvableExpression(expression)) < high_loss(
        EvolvableExpression(expression.copy())
    )
