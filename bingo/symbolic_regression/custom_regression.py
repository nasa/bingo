"""Advanced independently configurable expression-regression objective."""

import numpy as np

from ._expression_regression_objective import _ExpressionRegressionObjective
from .fitting import FitResult
from .objective_data import ObjectiveData


class CustomRegression(_ExpressionRegressionObjective):
    """Fit Expressions with a selected fitter and rank them with a scalar loss.

    Fitter, measure, and loss callables may be any compatible Python callable
    during serial evaluation. Multiprocessing evaluation and checkpointing
    require those callables to be pickleable.

    Parameters
    ----------
    data : ObjectiveData
        Aligned arrays provided to the fitter, fitting measure, and loss.
    fitter : callable
        Called as ``fitter(expression, data, fitting_measure)`` and returns a
        :class:`FitResult`.
    fitting_measure : callable
        Called as ``fitting_measure(expression, data, constants)`` by fitter.
    loss : callable
        Lower-is-better scalar called as ``loss(expression, data)``.
    """

    def __init__(self, data, fitter, fitting_measure, loss):
        if not isinstance(data, ObjectiveData):
            raise TypeError("CustomRegression data must be an ObjectiveData instance")
        if not callable(fitter):
            raise TypeError("CustomRegression fitter must be callable")
        if not callable(fitting_measure):
            raise TypeError("CustomRegression fitting_measure must be callable")
        if not callable(loss):
            raise TypeError("CustomRegression loss must be callable")
        super().__init__(data)
        self._fitter = fitter
        self._fitting_measure = fitting_measure
        self._loss = loss

    def __call__(self, individual):
        """Fit and rank an individual, committing only after successful ranking."""
        expression = individual.expression
        data = self._objective_data
        if expression.is_fitted:
            loss = self._expression_loss(expression, data)
        elif not expression.constants:
            loss = self._expression_loss(expression, data)
        else:
            constants = self._fit_constants(expression, data)
            trial = expression.copy()
            trial.commit_fit(constants)
            loss = self._expression_loss(trial, data)
            expression.commit_fit(constants)
        self.eval_count += 1
        return loss

    def _fit_expression(self, expression, data):
        expression.commit_fit(self._fit_constants(expression, data))

    def _fit_constants(self, expression, data):
        if not expression.constants:
            return np.empty(0)
        result = self._fitter(expression, data, self._fitting_measure)
        if not isinstance(result, FitResult):
            raise TypeError("fitters must return a FitResult")
        constants = np.asarray(result.constants, dtype=float)
        if constants.ndim != 1:
            raise ValueError("FitResult constants must be a one-dimensional array")
        return constants

    def _expression_loss(self, expression, data):
        value = np.asarray(self._loss(expression, data), dtype=float)
        if value.ndim != 0:
            raise ValueError("CustomRegression loss must return a scalar")
        return float(value)
