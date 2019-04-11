"""The definition of fitness evalutions for individuals.

This module defines the basis of fitness evaluation in bingo evolutionary
analyses.
"""
from abc import ABCMeta, abstractmethod

import numpy as np


class FitnessFunction(metaclass=ABCMeta):
    """Fitness evaluation metric for individuals.

    An abstract base class for the fitness evaluation of genetic individuals
    (Chromosomes) in bingo.

    Parameters
    ----------
    training_data :
                   (Optional) data that can be used in fitness evaluation

    Attributes
    ----------
    eval_count : int
                 the number of evaluations that have been performed
    training_data :
                   (Optional) data that can be used in fitness evaluation
    """
    def __init__(self, training_data=None):
        self.eval_count = 0
        self.training_data = training_data

    @abstractmethod
    def __call__(self, individual):
        """Evaluates the fitness of an individual

        Parameters
        ----------
        individual : Chromosome
                     individual for which fitness will be calculated

        Notes
        -----
        The eval_count should be incremented in a subclass' __call__ definition
        for accurate evaluation counting

        Returns
        -------
         :
            fitness of the individual
        """
        raise NotImplementedError


class VectorBasedFunction(FitnessFunction, metaclass=ABCMeta):
    """Fitness evaluation based on vectorized fitness
    """
    def __call__(self, individual):
        """Vector based fitness evaluation

        Evaluate the fitness of an individual as the total absolute error of
        vectorized fitness values.

        Parameters
        ----------
        individual : Chromosome
                     individual for which fitness will be calculated

        Returns
        -------
         :
           fitness of the individual
        """
        fitness_vector = self._evaluate_fitness_vector(individual)
        return np.mean(np.abs(fitness_vector))

    @abstractmethod
    def _evaluate_fitness_vector(self, individual):
        raise NotImplementedError
