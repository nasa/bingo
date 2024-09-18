"""The definition of fitness evaluations for individuals.

This module defines the basis of fitness evaluation in bingo evolutionary
analyses.
"""

from abc import ABCMeta, abstractmethod

import numpy as np


# Fitness metric functions, outside of FitnessFunction for use in GradientMixin
def mean_absolute_error(vector, individual=None):  # pylint: disable=unused-argument
    """Calculate the mean absolute error of an error vector"""
    return np.mean(np.abs(vector))


def root_mean_squared_error(vector, individual=None):  # pylint: disable=unused-argument
    """Calculate the root mean squared error of an error vector"""
    return np.sqrt(np.mean(np.square(vector)))


def mean_squared_error(vector, individual=None):  # pylint: disable=unused-argument
    """Calculate the mean squared error of an error vector"""
    return np.mean(np.square(vector))


def negative_nmll_laplace(vector, individual):
    """Calculate the nmll squared error of an error vector"""
    n = len(vector)
    k = individual.get_number_local_optimization_params() + 1
    b = 1 / np.sqrt(n)
    mse = np.mean(np.square(vector))
    log_like = -n / 2 * np.log(mse) - n / 2 - n / 2 * np.log(2 * np.pi)
    nmll_laplace = (1 - b) * log_like + np.log(b) / 2 * k
    return -nmll_laplace


class FitnessFunction(metaclass=ABCMeta):
    """Fitness evaluation metric for individuals.

    An abstract base class for the fitness evaluation of genetic individuals
    (chromosomes) in bingo.

    Parameters
    ----------
    training_data : TrainingData
        (Optional) data that can be used in fitness evaluation

    Attributes
    ----------
    eval_count : int
        the number of evaluations that have been performed
    training_data : TrainingData
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
        fitness : numeric
            fitness of the individual
        """
        raise NotImplementedError


class VectorBasedFunction(FitnessFunction, metaclass=ABCMeta):
    """Fitness evaluation based on vectorized fitness

    Parameters
    ----------
    training_data : TrainingData
        data that is used in fitness evaluation.
    metric : str
        String defining the measure of error to use. Available options are:
        'mean absolute error'/'mae', 'mean squared error'/'mse', and
        'root mean squared error'/'rmse'
    """

    def __init__(self, training_data=None, metric="mae"):
        super().__init__(training_data)

        if metric in ["mean absolute error", "mae"]:
            self._metric = mean_absolute_error
        elif metric in ["mean squared error", "mse"]:
            self._metric = mean_squared_error
        elif metric in ["root mean squared error", "rmse"]:
            self._metric = root_mean_squared_error
        elif metric in ["negative nmll laplace"]:
            self._metric = negative_nmll_laplace
        else:
            raise ValueError("Invalid metric for Fitness Function")

    def __call__(self, individual):
        """Vector based fitness evaluation

        Evaluate the fitness of an individual as based on a vector of fitness
        (error) values.  The metric defined in the constructor is used to
        aggregate the vector fitness into a single fitness value

        Parameters
        ----------
        individual : Chromosome
            individual for which fitness will be calculated

        Returns
        -------
        fitness : numeric
            fitness of the individual
        """
        fitness_vector = self.evaluate_fitness_vector(individual)
        return self._metric(fitness_vector, individual)

    @abstractmethod
    def evaluate_fitness_vector(self, individual):
        """Calculate a vector of fitness values for the passed in individual

        Parameters
        ----------
        individual : Chromosome
            individual for which fitness will be calculated

        Returns
        -------
        vector_fitness : array of numeric
            a vector of fitness values for the passed in individual
        """
        raise NotImplementedError
