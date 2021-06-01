# TODO documentation
from abc import ABCMeta, abstractmethod

import numpy as np


class GradientMixin(metaclass=ABCMeta):
    @abstractmethod
    def get_gradient(self, individual):
        raise NotImplementedError


class VectorGradientMixin(GradientMixin):
    def get_gradient(self, individual):
        """Vector based fitness gradient

        Get the gradient of this function with respect to an individual's constants.

        Parameters
        ----------
        individual : chromosomes
            individual for which the gradient will be calculated for

        Returns
        -------
        gradient :
            the gradient of this function with respect to each of the individual's constants
        """
        # TODO elegant way to get fitness vector and metric derivative?
        fitness_vector = self.evaluate_fitness_vector(individual)
        fitness_derivatives = self.get_jacobian(individual).transpose()

        gradient = np.zeros(len(fitness_derivatives))
        for i in range(len(fitness_derivatives)):
            gradient[i] = self._metric_derivative(fitness_vector, fitness_derivatives[i])
        return gradient

    @abstractmethod
    def get_jacobian(self, individual):
        raise NotImplementedError
