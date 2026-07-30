"""Expression-backed explicit symbolic-regression objective."""

import numpy as np

from ..evaluation.fitness_function import FitnessFunction
from ..evaluation.training_data import TrainingData


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

    def __init__(self, X, y, loss="mse"):
        super().__init__()
        self._objective_data = _ExplicitObjectiveData(X, y)
        self._loss = loss

    def __call__(self, individual):
        """Fit an Expression once, then return its loss on active objective data."""
        expression = individual.expression
        data = self._objective_data
        if not expression.is_fitted:
            expression.fit(data.X, data.y, tolerance=1e-5)
        self.eval_count += 1
        return expression.loss(data.X, data.y, kind=self._loss)


class ExplicitTrainingData(TrainingData):
    """Legacy explicit training-data container retained until the API cutover."""

    def __init__(self, x, y):
        if x.ndim == 1:
            x = x.reshape([-1, 1])
        if x.ndim > 2:
            raise TypeError("Explicit training x should be 2 dim array")
        if y.ndim == 1:
            y = y.reshape([-1, 1])
        if y.ndim > 2:
            raise TypeError("Explicit training y should be 2 dim array")
        self._x = x
        self._y = y

    @property
    def x(self):
        """Independent data."""
        return self._x

    @property
    def y(self):
        """Dependent data."""
        return self._y

    def __getitem__(self, items):
        return ExplicitTrainingData(self._x[items, :], self._y[items, :])

    def __len__(self):
        return self._x.shape[0]
