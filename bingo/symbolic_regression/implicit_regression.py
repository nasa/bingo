"""Expression-backed implicit symbolic-regression objective."""

import numpy as np

from ._expression_regression_objective import _ExpressionRegressionObjective
from .objective_data import ObjectiveData


class _ImplicitObjectiveData(ObjectiveData):
    """Aligned implicit-regression arrays kept private by the objective."""

    def __init__(self, X, dx_dt):
        self.X = np.asarray(X, dtype=float)
        if self.X.ndim == 1:
            self.X = self.X.reshape(-1, 1)
        if self.X.ndim != 2:
            raise TypeError("Implicit regression X must be a 2D array")
        self.dx_dt = np.asarray(dx_dt, dtype=float)
        if self.dx_dt.ndim == 1:
            self.dx_dt = self.dx_dt.reshape(-1, 1)
        if self.dx_dt.ndim != 2:
            raise TypeError("Implicit regression dx_dt must be a 2D array")
        if self.X.shape != self.dx_dt.shape:
            raise ValueError("Implicit regression X and dx_dt must have equal shape")
        super().__init__(self.X, self.dx_dt)


class ImplicitRegression(_ExpressionRegressionObjective):
    """Lower-is-better implicit-regression loss for evolvable Expressions."""

    def __init__(self, X, dx_dt, required_params=None):
        super().__init__(_ImplicitObjectiveData(X, dx_dt))
        self._required_params = required_params

    def _fit_expression(self, expression, data):
        expression.fit_implicit(data.X, data.dx_dt, tolerance=1e-5)

    def _expression_loss(self, expression, data):
        return expression.implicit_loss(
            data.X, data.dx_dt, required_params=self._required_params
        )
