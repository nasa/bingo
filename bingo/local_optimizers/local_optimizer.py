"""This module contains the abstract definition of an optimizer
that can be used for local optimization of a `Chromosome`.
"""

from abc import ABCMeta, abstractmethod


class LocalOptimizer(metaclass=ABCMeta):
    """An abstract base class for optimizing a `Chromosome`.
    """
    @property
    @abstractmethod
    def objective_fn(self):
        """function to minimize, must take a `Chromosome` as input
        and return a number"""
        raise NotImplementedError

    @objective_fn.setter
    @abstractmethod
    def objective_fn(self, value):
        raise NotImplementedError

    @property
    @abstractmethod
    def options(self):
        """dict : optimizer's options"""
        raise NotImplementedError

    @options.setter
    @abstractmethod
    def options(self, value):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, individual):
        """Performs local optimization of the individual
        based on minimizing this object's objective_fn.

        Parameters
        ----------
        individual : `Chromosome`
            The individual who will be optimized.
        """
        raise NotImplementedError
