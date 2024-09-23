"""Fitness evaluation with local optimization

This module contains the implementation of a fitness function wrapper
that will perform local optimization of a `Chromosome` as necessary
using a `LocalOptimizer` before evaluating it.
"""

from ..evaluation.fitness_function import FitnessFunction


class LocalOptFitnessFunction(FitnessFunction):
    """Fitness function wrapper for individuals that want local optimization

    A class for fitness evaluation of individuals that may or may
    not need local optimization before evaluation.

    Parameters
    ----------
    fitness_function : `FitnessFunction`
        A `FitnessFunction` for evaluating the fitness of a `Chromosome`.
    optimizer : `LocalOptimizer`
        An optimizer that will perform local optimization on a
        `Chromosome` before evaluation as needed.

    Attributes
    ----------
    eval_count : int
        the number of evaluations that have been performed by the wrapped
        fitness function
    training_data : `TrainingData`
        data that can be used in the wrapped fitness function
    """

    def __init__(self, fitness_function, optimizer):
        # pylint: disable=super-init-not-called
        self._fitness_function = fitness_function
        self.optimizer = optimizer

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
        on the individual if necessary.

        Parameters
        ----------
        individual : `Chromosome`
            Individual to calculate the fitness of. Local optimization
            is performed if necessary before evaluation.

        Returns
        -------
        float
            The fitness of the individual
        """
        if individual.needs_local_optimization():
            self.optimizer(individual)
        return self._fitness_function(individual)
