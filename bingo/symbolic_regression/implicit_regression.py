"""Expression-backed implicit symbolic-regression objective."""

import numpy as np

from .custom_regression import CustomRegression
from .fitting import ScipyFitter, implicit_residuals
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


class ImplicitRegression(CustomRegression):
    """Lower-is-better implicit-regression loss for evolvable Expressions."""

    def __init__(self, X, dx_dt, required_params=None):
        data = _ImplicitObjectiveData(X, dx_dt)
        super().__init__(
            data,
            ScipyFitter("least_squares"),
            implicit_residuals(),
            lambda expression, objective_data: expression.implicit_loss(
                objective_data.X,
                objective_data.dx_dt,
                required_params=required_params,
            ),
        )
        self._required_params = required_params
