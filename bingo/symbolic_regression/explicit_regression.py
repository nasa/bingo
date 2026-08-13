"""Expression-backed explicit symbolic-regression objective."""

import numpy as np

from ._expression_regression_objective import _ExpressionRegressionObjective
from .objective_data import ObjectiveData


class _ExplicitObjectiveData(ObjectiveData):
    """Aligned explicit-regression arrays kept private by the objective."""

    def __init__(self, X, y):
        self.X = np.asarray(X, dtype=float)
        if self.X.ndim == 1:
            self.X = self.X.reshape(-1, 1)
        if self.X.ndim != 2:
            raise TypeError("Explicit regression X must be a 2D array")
        self.y = np.asarray(y, dtype=float).ravel()
        super().__init__(self.X, self.y)


class ExplicitRegression(_ExpressionRegressionObjective):
    """Lower-is-better explicit-regression loss for evolvable Expressions."""

    def __init__(self, X, y, loss="mse", fit_tolerance=1e-5):
        super().__init__(_ExplicitObjectiveData(X, y))
        self._loss = loss
        self._fit_tolerance = fit_tolerance

    def _fit_expression(self, expression, data):
        expression.fit(data.X, data.y, tolerance=self._fit_tolerance)

    def _expression_loss(self, expression, data):
        return expression.loss(data.X, data.y, kind=self._loss)
