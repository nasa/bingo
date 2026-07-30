"""Expression-backed implicit symbolic-regression objective."""

import numpy as np

from ..evaluation.fitness_function import FitnessFunction


class _ImplicitObjectiveData:
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

    def __getitem__(self, items):
        return _ImplicitObjectiveData(self.X[items], self.dx_dt[items])

    def __len__(self):
        return len(self.X)


class ImplicitRegression(FitnessFunction):
    """Lower-is-better implicit-regression loss for evolvable Expressions."""

    def __init__(self, X, dx_dt, required_params=None):
        super().__init__()
        self._objective_data = _ImplicitObjectiveData(X, dx_dt)
        self._required_params = required_params

    def __call__(self, individual):
        """Fit an Expression once, then return its loss on active objective data."""
        expression = individual.expression
        data = self._objective_data
        if not expression.is_fitted:
            expression.fit_implicit(data.X, data.dx_dt, tolerance=1e-5)
        self.eval_count += 1
        return expression.implicit_loss(
            data.X, data.dx_dt, required_params=self._required_params
        )
