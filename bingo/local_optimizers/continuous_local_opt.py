"""Local optimization of continuous, real-valued parameters

This module contains the implementation of local optimization or continuous,
real-valued parameters in a chromosome.  The local optimization algorithm is
defined as well as the interface that must implemented by chromosomes wishing
to use the functionality.
"""
from abc import ABCMeta, abstractmethod

from ..evaluation.fitness_function import FitnessFunction


class ContinuousLocalOptimization(FitnessFunction):
    def __init__(self, optimizer):
        self.optimizer = optimizer

    @property
    def training_data(self):
        """TrainingData : data that can be used in fitness evaluations"""
        return self.optimizer.objective_fn.training_data

    @training_data.setter
    def training_data(self, value):
        self.optimizer.objective_fn.training_data = value

    @property
    def eval_count(self):
        """int : the number of evaluations that have been performed"""
        return self.optimizer.objective_fn.eval_count

    @eval_count.setter
    def eval_count(self, value):
        self.optimizer.objective_fn.eval_count = value

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
            The fitness of the individual
        """
        if individual.needs_local_optimization():
            self.optimizer(individual)
        return self.optimizer.objective_fn(individual)


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
