"""Normalized marginal likelihood calculation using SMCPy

This module contains the implementation of a fitness function wrapper
that will perform probabilistic local optimization of a `Chromosome` using 
SMCPy.  The normaized marginal likelihood from the SMC optimization is returned.
"""

import numpy as np
from bingo.local_optimizers.smcpy_optimizer import SmcpyOptimizer
from ..evaluation.fitness_function import FitnessFunction


class NormalizedMarginalLikelihood(FitnessFunction):
    """Normalized marginal likelihood calculation using SMCPy

    A class for fitness evaluation of individuals that have local optimization
    parameters

    Parameters
    ----------
    fitness_function : `FitnessFunction`
        A `FitnessFunction` for evaluating the fitness of a `Chromosome`.
    deterministic_optimizer : `LocalOptimizer`
        An optimizer that will perform deterministic local optimization on a
        `Chromosome`. Used in proposals of `SmcpyOptimizer`
    **kwargs:
        other keyword arguments are passed to the SmcpyOptimizer initialization

    Attributes
    ----------
    eval_count : int
        the number of evaluations that have been performed by the wrapped
        fitness function
    training_data : `TrainingData`
        data that can be used in the wrapped fitness function
    """

    def __init__(
        self, fitness_function, deterministic_optimizer, log_scale=True, **kwargs
    ):
        # pylint: disable=super-init-not-called
        self._log_scale = log_scale
        self.optimizer = SmcpyOptimizer(
            fitness_function, deterministic_optimizer, **kwargs
        )

    @property
    def training_data(self):
        """TrainingData : data that can be used in fitness evaluations"""
        return self.optimizer.training_data

    @training_data.setter
    def training_data(self, value):
        self.optimizer.training_data = value

    @property
    def eval_count(self):
        """int : the number of evaluations that have been performed"""
        return self.optimizer.eval_count

    @eval_count.setter
    def eval_count(self, value):
        self.optimizer.eval_count = value

    def __call__(self, individual):
        """Evaluates the normalized marginal likelihood of the individual.

        Parameters
        ----------
        individual : `Chromosome`
            Individual to calculate the normalized marginal likelihood of.
            Probabilistic local optimization is performed during evaluation.

        Returns
        -------
        float
            The *negative* normalized marginal likelihood
        """
        log_nml = self.optimizer(individual)[0]
        if self._log_scale:
            return -log_nml
        return -np.exp(log_nml)
