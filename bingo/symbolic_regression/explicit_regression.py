"""Expression-backed explicit symbolic-regression objective."""

import numpy as np

from .custom_regression import CustomRegression
from .fitting import ScipyFitter, explicit_residuals
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


class ExplicitRegression(CustomRegression):
    """Lower-is-better explicit-regression loss for evolvable Expressions.

    Parameters
    ----------
    X : array-like
        Predictor values with samples along the first axis.
    y : array-like
        Target values aligned with ``X``.
    loss : str, optional
        Named Expression loss used for ranking.
    fit_tolerance : float, optional
        Convergence tolerance for Levenberg-Marquardt fitting.

    Raises
    ------
    TypeError
        If ``X`` cannot be represented as a two-dimensional array.
    ValueError
        If ``X`` and ``y`` have unequal sample counts.
    """

    def __init__(self, X, y, loss="mse", fit_tolerance=1e-5):
        data = _ExplicitObjectiveData(X, y)
        super().__init__(
            data,
            ScipyFitter("lm", tolerance=fit_tolerance),
            explicit_residuals(),
            lambda expression, objective_data: expression.loss(
                objective_data.X, objective_data.y, kind=loss
            ),
        )
        self._loss_kind = loss
        self._fit_tolerance = fit_tolerance
