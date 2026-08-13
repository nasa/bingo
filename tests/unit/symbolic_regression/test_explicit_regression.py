# pylint: disable=missing-docstring
import numpy as np

from bingo.expressions.agraph.evolvable import EvolvableExpression
from bingo.expressions.agraph.pyagraph.expression import AGraphExpression
from bingo.expressions.agraph.pyagraph.operators import CONSTANT, VARIABLE
from bingo.symbolic_regression.explicit_regression import ExplicitRegression


def _constant_expression(value=0.0):
    expression = AGraphExpression()
    expression.raw_command_array = np.array([[CONSTANT, 0, 0]], dtype=np.uint8)
    expression.raw_constants = (value,)
    return expression


def test_explicit_objective_fits_an_unfitted_expression():
    expression = _constant_expression()
    individual = EvolvableExpression(expression)
    objective = ExplicitRegression(np.arange(4.0).reshape(-1, 1), np.full(4, 2.0))

    assert objective(individual) == 0.0
    assert expression.constants == (2.0,)
    assert expression.is_fitted
    assert objective.eval_count == 1


def test_explicit_objective_does_not_refit_an_unchanged_expression(mocker):
    expression = _constant_expression()
    individual = EvolvableExpression(expression)
    objective = ExplicitRegression(np.arange(4.0).reshape(-1, 1), np.full(4, 2.0))
    objective(individual)
    fit = mocker.spy(expression, "fit")

    assert objective(individual) == 0.0
    fit.assert_not_called()


def test_explicit_objective_returns_expression_loss():
    expression = AGraphExpression()
    expression.raw_command_array = np.array([[VARIABLE, 0, 0]], dtype=np.uint8)
    individual = EvolvableExpression(expression)
    x = np.arange(4.0).reshape(-1, 1)
    y = np.full(4, 3.0)
    objective = ExplicitRegression(x, y, loss="mae")

    assert objective(individual) == 1.5


def test_explicit_objective_prefers_lower_loss():
    x = np.arange(4.0).reshape(-1, 1)
    objective = ExplicitRegression(x, x.ravel(), loss="mse")
    matching = AGraphExpression()
    matching.raw_command_array = np.array([[VARIABLE, 0, 0]], dtype=np.uint8)
    mismatching = _constant_expression(1.0)
    mismatching.fit(x, np.ones(4))

    assert objective(EvolvableExpression(matching)) < objective(
        EvolvableExpression(mismatching)
    )
