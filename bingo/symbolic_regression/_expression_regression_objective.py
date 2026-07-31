"""Shared lifecycle for Expression-backed regression objectives."""

from abc import ABCMeta, abstractmethod

from ..evaluation.fitness_function import FitnessFunction


class _ExpressionRegressionObjective(FitnessFunction, metaclass=ABCMeta):
    """Fit an Expression once, then calculate a regression loss."""

    def __init__(self, objective_data):
        super().__init__()
        self._objective_data = objective_data

    def __call__(self, individual):
        """Fit an Expression once, then return its loss on objective data."""
        expression = individual.expression
        data = self._objective_data
        if not expression.is_fitted:
            self._fit_expression(expression, data)
        self.eval_count += 1
        return self._expression_loss(expression, data)

    @abstractmethod
    def _fit_expression(self, expression, data):
        """Fit an unfitted Expression to objective data."""
        raise NotImplementedError

    @abstractmethod
    def _expression_loss(self, expression, data):
        """Calculate an Expression's loss on objective data."""
        raise NotImplementedError