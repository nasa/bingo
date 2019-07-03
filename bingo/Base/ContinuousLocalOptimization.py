"""Local optimization of continuous, real-valued parameters

This module contains the implementation of local optimization or continuous,
real-valued parameters in a chromosome.  The local optimization algorithm is
defined as well as the interface that must implemented by chromosomes wishing
to use the functionality.
"""
from abc import ABCMeta, abstractmethod

import numpy as np
import scipy.optimize as optimize

from .FitnessFunction import FitnessFunction, VectorBasedFunction

ROOT_SET = {
    # 'hybr',
    'lm'
    # 'broyden1',
    # 'broyden2',
    # 'anderson',
    # 'linearmixing',
    # 'diagbroyden',
    # 'excitingmixing',
    # 'krylov',
    # 'df-sane'
}

MINIMIZE_SET = {
    'Nelder-Mead',
    'Powell',
    'CG',
    'BFGS',
    # 'Newton-CG',
    'L-BFGS-B',
    # 'TNC',
    # 'COBYLA',
    'SLSQP'
    # 'trust-constr'
    # 'dogleg',
    # 'trust-ncg',
    # 'trust-exact',
    # 'trust-krylov'
}

class ContinuousLocalOptimization(FitnessFunction):
    """Fitness evaluation metric for individuals.

    A class for the fitness evaluation metric of genetic individuals that may or
    may not need local optimization before evaluation.

    Parameters
    ----------
    fitness_function : `FitnessFunction`
        A FitnessFunction that evaluates the fitness of a `Chromosome` in bingo.
        For certain algorithms, `VectorBasedFunction` is required. Please see
        algorithm listing for details.
    algorithm : string
        An algorithm that is used in the local optimization of a
        `Chromosome`. The default algorithm is *Nelder-Mead*. The
        other options are:
            1. FitnessFunction
                - Nelder-Mead
                - Powell
                - CG
                - BFGS
                - Newton-CG (not available yet)
                - L-BFGS-B
                - TNC (not available yet)
                - COBYLA (not available yet)
                - SLSQP
                - trust-constr (not available yet)
                - dogleg (not available yet)
                - trust-ncg (not available yet)
                - trust-exact (not available yet)
                - trust-krylov (not available yet)
            2. VectorBasedFunction
                - hybr (not available yet)
                - lm
                - broyden1 (not available yet)
                - broyden2 (not available yet)
                - anderson (not available yet)
                - linearmixing (not available yet)
                - diagbroyden (not available yet)
                - excitingmixing (not available yet)
                - krylov (not available yet)
                - df-sane (not available yet)

    Attributes
    ----------
    eval_count : int
                 the number of evaluations that have been performed by the
                 wrapped fitness function
    training_data :
                   (Optional) data that can be used in the wrapped fitness
                   function

    Raises
    ------
    KeyError:
        `algorithm` must be an algorithm provided by the interface
    TypeError :
        `fitness_function` must Be a valid `FitnessFunction` for the specified
        algorithm
    """
    def __init__(self, fitness_function, algorithm='Nelder-Mead'):
        self._check_algorithm_is_valid(algorithm)
        self._check_root_alg_returns_vector(fitness_function, algorithm)
        self._fitness_function = fitness_function
        self._algorithm = algorithm

    @property
    def training_data(self):
        """TrainingData : data that can be used in fitness evaluations"""
        return self._fitness_function.training_data

    @training_data.setter
    def training_data(self, value):
        self._fitness_function.training_data = value

    @property
    def eval_count(self):
        """int : the number of evaluations that have been performed"""
        return self._fitness_function.eval_count

    @eval_count.setter
    def eval_count(self, value):
        self._fitness_function.eval_count = value

    def __call__(self, individual):
        """Evaluates the fitness of the individual. Provides local optimization
        on `MultipleFloatChromosome` individual if necessary.

        Parameters
        ----------
        individual : `MultipleValueChromosome`
            Individual to which to calculate the fitness. If the individual is
            an instance of `MultipleFloatChromosome`, then local optimization
            may be performed if necessary and the correct `FitnessFunction` is
            provided.

        Returns
        -------
        float :
            The fitness of the invdividual
        """
        if individual.needs_local_optimization():
            self._optimize_params(individual)
        return self._evaluate_fitness(individual)

    @staticmethod
    def _check_algorithm_is_valid(algorithm):
        if algorithm not in ROOT_SET and algorithm not in MINIMIZE_SET:
            raise KeyError("{} is not a listed algorithm".format(algorithm))

    @staticmethod
    def _check_root_alg_returns_vector(fitness_function, algorithm):
        if algorithm in ROOT_SET and not isinstance(fitness_function,
                                                    VectorBasedFunction):
            raise TypeError("{} requires VectorBasedFunction\
                            as a fitness function".format(algorithm))

    def _optimize_params(self, individual):
        num_params = individual.get_number_local_optimization_params()
        c_0 = np.random.uniform(-10000, 10000, num_params)
        params = self._run_algorithm_for_optimization(
            self._sub_routine_for_fit_function, individual, c_0)
        individual.set_local_optimization_params(params)

    def _sub_routine_for_fit_function(self, params, individual):
        individual.set_local_optimization_params(params)
        if self._algorithm in ROOT_SET:
            return self._fitness_function.evaluate_fitness_vector(individual)
        return self._fitness_function(individual)

    def _run_algorithm_for_optimization(self, sub_routine, individual, params):
        if self._algorithm in ROOT_SET:
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
