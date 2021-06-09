"""Mixin classes used to extend fitness functions to be able to use
gradient- and jacobian-based continuous local optimization methods.

This module defines the basis of gradient and jacobian partial derivatives
of fitness functions used in bingo evolutionary analyses.
"""
from abc import ABCMeta, abstractmethod

import numpy as np

from .fitness_function import VectorBasedFunction


class GradientMixin(metaclass=ABCMeta):
    """Mixin for using gradients for fitness functions

    An abstract base class/mixin used to implement the gradients
    of fitness functions.
    """
    @abstractmethod
    def get_gradient(self, individual):
        """Fitness function gradient

        Get the gradient of this function with respect to the
        passed in individual's constants.

        Parameters
        ----------
        individual : chromosomes
            individual for which the gradient will be calculated for

        Returns
        -------
        gradient :
            the gradient of this function with respect to each of the individual's constants
        """
        raise NotImplementedError


class VectorGradientMixin(GradientMixin):
    """Mixin for using gradients and jacobians for vector based fitness functions

    An abstract base class/mixin used to implement the gradients and jacobians
    of vector based fitness functions.
    """
    def __init__(self, *args, **kwargs):
        if not isinstance(self, VectorBasedFunction):
            raise TypeError("VectorGradientMixin should be used with a VectorBasedFunction")
        super().__init__(*args, **kwargs)

        if self._metric_string == "mae":
            self._metric_derivative = VectorGradientMixin._mean_absolute_error_derivative
        elif self._metric_string == "mse":
            self._metric_derivative = VectorGradientMixin._mean_squared_error_derivative
        elif self._metric_string == "rmse":
            self._metric_derivative = VectorGradientMixin._root_mean_squared_error_derivative

    def get_gradient(self, individual):
        """Gradient of vector based fitness function with metric
        (i.e. the fitness function originally returns a vector
        that is converted into a scalar using some metric)

        Get the gradient of this function with respect to the
        passed in individual's constants.

        Parameters
        ----------
        individual : chromosomes
            individual for which the gradient will be calculated for

        Returns
        -------
        gradient :
            the gradient of this function with respect to each of the individual's constants
        """
        fitness_vector = self.evaluate_fitness_vector(individual)
        fitness_partials = self.get_jacobian(individual).transpose()
        return self._metric_derivative(fitness_vector, fitness_partials)

    @abstractmethod
    def get_jacobian(self, individual):
        """Returns the jacobian of this vector fitness function with
        respect to the passed in individual's constants

        jacobian = [[:math:`df1/dc1`, :math:`df1/dc2`, ...],
                    [:math:`df2/dc1`, :math:`df2/dc2`, ...],
                    ...]
            where :math:`f` # is the fitness function corresponding with the
            #th fitness vector entry and :math:`c` # is the corresponding
            constant of the individual

        Parameters
        ----------
        individual : chromosomes
            individual whose constants are used for the jacobian calculation

        Returns
        -------
        jacobian :
            the partial derivatives of each fitness function with respect
            to each of the individual's constants
        """
        raise NotImplementedError

    @staticmethod
    def _mean_absolute_error_derivative(fitness_vector, fitness_partials):
        return np.mean(np.sign(fitness_vector) * fitness_partials, axis=1)

    @staticmethod
    def _mean_squared_error_derivative(fitness_vector, fitness_partials):
        return 2 * np.mean(fitness_vector * fitness_partials, axis=1)

    @staticmethod
    def _root_mean_squared_error_derivative(fitness_vector, fitness_partials):
        return 1/np.sqrt(np.mean(np.square(fitness_vector))) * np.mean(fitness_vector * fitness_partials, axis=1)
