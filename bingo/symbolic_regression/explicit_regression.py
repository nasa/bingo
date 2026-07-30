"""Expression-backed explicit symbolic-regression objective."""

import numpy as np

from ..evaluation.fitness_function import FitnessFunction


class _ExplicitObjectiveData:
    """Aligned explicit-regression arrays kept private by the objective."""

    def __init__(self, X, y):
        self.X = np.asarray(X, dtype=float)
        if self.X.ndim == 1:
            self.X = self.X.reshape(-1, 1)
        if self.X.ndim != 2:
            raise TypeError("Explicit regression X must be a 2D array")
        self.y = np.asarray(y, dtype=float).ravel()
        if len(self.X) != len(self.y):
            raise ValueError("Explicit regression X and y must have equal length")

    def __getitem__(self, items):
        return _ExplicitObjectiveData(self.X[items], self.y[items])

    def __len__(self):
        return len(self.X)


class ExplicitRegression(FitnessFunction):
    """Lower-is-better explicit-regression loss for evolvable Expressions."""

    def __init__(self, X, y, loss="mse", fit_tolerance=1e-5):
        super().__init__()
        self._objective_data = _ExplicitObjectiveData(X, y)
        self._loss = loss
        self._fit_tolerance = fit_tolerance

    def __call__(self, individual):
        """Fit an Expression once, then return its loss on active objective data."""
        expression = individual.expression
        data = self._objective_data
        if not expression.is_fitted:
            expression.fit(data.X, data.y, tolerance=self._fit_tolerance)
        self.eval_count += 1
        return expression.loss(data.X, data.y, kind=self._loss)
