"""Local optimization of continuous, real-valued parameters

This module contains the implementation of local optimization or continuous,
real-valued parameters in a chromosome.  The local optimization algorithm is
defined as well as the interface that must implemented by chromosomes wishing
to use the functionality.
"""
from abc import ABCMeta, abstractmethod

import numpy as np
import scipy.optimize as optimize

from .FitnessEvaluator import FitnessEvaluator, VectorBasedEvaluator

class ContinuousLocalOptimization(FitnessEvaluator):

    def __init__(self, fitness_function, algorithm='Nelder-Mead'):
        super().__init__()
        self._check_lm_alg_returns_vector(fitness_function, algorithm)
        self._fitness_function = fitness_function
        self._algorithm = algorithm

    def __call__(self, individual):
        if individual.needs_local_optimization():
            self._optimize_params(individual)
        return self._evaluate_fitness(individual)

    def _check_lm_alg_returns_vector(self, fitness_function, algorithm):
        if algorithm == 'lm' and not isinstance(fitness_function,
                                                VectorBasedEvaluator):
            raise TypeError("Levenberg-Marquart requires VectorBasedFunction\
                            as a fitness function")

    def _optimize_params(self, individual):
        num_params = individual.get_number_local_optimization_params()
        c_0 = np.random.uniform(-10000, 10000, num_params)
        params = self._run_algorithm_for_optimization(
            self._sub_routine_for_fit_function, individual, c_0)
        individual.set_local_optimization_params(params)

    def _sub_routine_for_fit_function(self, params, individual):
        individual.set_local_optimization_params(params)
        if self._algorithm == 'lm':
            return self._fitness_function._evaluate_fitness_vector(individual)
        return self._fitness_function(individual)

    def _run_algorithm_for_optimization(self, sub_routine, individual, params):
        if self._algorithm == 'lm':
            optimize_result = optimize.root(sub_routine, params,
                                            args=(individual),
                                            method=self._algorithm,
                                            tol=1e-6)
        else:
            optimize_result = optimize.minimize(sub_routine, params,
                                                args=(individual),
                                                method=self._algorithm,
                                                tol=1e-6)
        return optimize_result.x

    def _evaluate_fitness(self, individual):
        return self._fitness_function(individual)


class ChromosomeInterface(metaclass=ABCMeta):
    """For chromosomes with continuous local optimization

    An interface to be used on Chromosomes that will be using continuous local
    optimization.
    """
    @abstractmethod
    def needs_local_optimization(self):
        """Does the individual need local optimization

        Returns
        -------
        bool
            Individual needs optimization
        """
        raise NotImplementedError

    @abstractmethod
    def get_number_local_optimization_params(self):
        """Get number of parameters in local optimization

        Returns
        -------
        int
            number of paramneters to be optimized
        """
        raise NotImplementedError

    @abstractmethod
    def set_local_optimization_params(self, params):
        """Set local optimization parameters

        Parameters
        ----------
        params : list-like of numeric
                 Values to set the parameters
        """
        raise NotImplementedError
