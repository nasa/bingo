"""Fitness evaluation with local optimization of continuous,
real-valued parameters

This module contains the implementation of a fitness function wrapper
that will perform optimization of a `Chromosome`'s constants as necessary
before evaluating it.  An interface, `ChromosomeInterface`, is also
defined and must implemented by `Chromosome`s wishing to use the
optimization wrapper.
"""
from abc import ABCMeta, abstractmethod

from ..evaluation.fitness_function import FitnessFunction


class ContinuousLocalOptimization(FitnessFunction):
    """Fitness function wrapper for individuals that want local optimization

    A class for fitness evaluation of individuals that may or may
    not need local optimization before evaluation.

    Parameters
    ----------
    fitness_function : `FitnessFunction`
        A `FitnessFunction` for evaluating the fitness of a `Chromosome`.
    optimizer : `Optimizer`
        An optimizer that will perform local optimization on a
        `Chromosome`'s constants before evaluation as needed.

    Attributes
    ----------
    eval_count : int
        the number of evaluations that have been performed by the wrapped
        fitness function
    training_data : `TrainingData`
        data that can be used in the wrapped fitness function
    """
    def __init__(self, fitness_function, optimizer):
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
            Individual to calculate the fitness of. If the individual is
            an instance of `Chromosome`, then local optimization
            is performed if necessary before evaluation.

        Returns
        -------
        float
            The fitness of the individual
        """
        if individual.needs_local_optimization():
            self.optimizer(individual)
        return self._fitness_function(individual)


class ChromosomeInterface(metaclass=ABCMeta):
    """For chromosomes with continuous local optimization

    An interface to be used on chromosomes that will be using continuous local
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
            Number of parameters to be optimized
        """
        raise NotImplementedError

    @abstractmethod
    def set_local_optimization_params(self, params):
        """Set local optimization parameters

        Parameters
        ----------
        params : list-like of numeric
            Values to set the parameters to
        """
        raise NotImplementedError
