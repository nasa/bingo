# TODO documentation
from abc import ABCMeta, abstractmethod


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
        fitness_partials = self.get_jacobian(individual).transpose()
        return self._metric_derivative(fitness_vector, fitness_partials)

    @abstractmethod
    def get_jacobian(self, individual):
        raise NotImplementedError
