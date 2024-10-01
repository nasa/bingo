"""
This module contains a wrapper around the bingo Equation object to match the 
regressor interface in scikit-learn. Calling `fit` performs a fitting of the 
numerical constants in the equation.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.symbolic_regression.explicit_regression import (
    ExplicitRegression,
    ExplicitTrainingData,
)  # this forces use of python fit funcs

INF_REPLACEMENT = 1e100


class EquationRegressor(RegressorMixin, BaseEstimator):
    """A thin scikit learn wrapper around bingo equations

    Parameters
    ----------
    equation : `Equation`
        equation that wiull be wrapped
    metric : str, optional
        metric used for local optimization on parameters during fit, by default "mse"
    algo : str, optional
        algorithm used for local optimization on parameters during fit, by default "lm"
    tol : _type_, optional
        tolerance used for local optimization on parameters during fit, by default 1e-6
    fit_retries : int, optional
        number of times to attempt to fit parameters. This is a hedge against the
        variability of selecting a random starting point for the local optimization,
        by default 5
    """

    def __init__(self, equation, metric="mse", algo="lm", tol=1e-6, fit_retries=5):
        self.equation = equation
        self.tol = tol
        self.fit_retries = fit_retries
        self.metric = metric
        self.algo = algo
        self.is_fitted_ = True

    @property
    def fitness(self):
        """Fitness of equation"""
        return self.equation.fitness

    @property
    def complexity(self):
        """Complexity of equation"""
        return self.equation.get_complexity()

    def fit(self, X, y, sample_weight=None):
        """Fit constants in equation to the given data.

        Parameters
        ----------
        X: MxD numpy array of numeric
            Input values. D is the number of dimensions and
            M is the number of data points.
        y: Mx1 numpy array of numeric
            Target/output values. M is the number of data points.
        sample_weight: Mx1 numpy array of numeric, optional
            Weights per sample/data point. M is the number of data points.
            Not currently supported
        """
        if sample_weight is not None:
            print("sample weight not None, TODO")
            raise NotImplementedError

        if self.equation.get_number_local_optimization_params() == 0:
            return

        fit_func = self._get_local_opt(X, y)
        best_fitness = fit_func(self.equation)
        best_constants = tuple(self.equation.constants)
        for _ in range(self.fit_retries):
            self.equation._needs_opt = True
            fitness = fit_func(self.equation)
            if fitness < best_fitness:
                best_fitness = fitness
                best_constants = tuple(self.equation.constants)
        self.equation.fitness = best_fitness
        self.equation.set_local_optimization_params(best_constants)

    def _get_local_opt(self, X, y):
        training_data = ExplicitTrainingData(X, y)
        fitness = ExplicitRegression(training_data=training_data, metric=self.metric)
        optimizer = ScipyOptimizer(fitness, method=self.algo, tol=self.tol)
        local_opt_fitness = LocalOptFitnessFunction(fitness, optimizer)
        return local_opt_fitness

    def predict(self, X):
        """Evaluate the equation to predict the outputs of `X`.

        Parameters
        ----------
        X: MxD numpy array of numeric
            Input values. D is the number of dimensions and
            M is the number of data points.

        Returns
        -------
        pred_y: Mx1 numpy array of numeric
            Predicted target/output values. M is the number of data points.
        """
        output = self.equation.evaluate_equation_at(X)
        return np.nan_to_num(output, posinf=INF_REPLACEMENT, neginf=-INF_REPLACEMENT)

    def __str__(self):
        return str(self.equation)
