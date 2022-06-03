"""This module contains the abstract definition of an interface
for `Chromosome`s that want to use local optimization.
"""
from abc import ABCMeta, abstractmethod


class LocalOptimizationInterface(metaclass=ABCMeta):
    """For `Chromosome`s with local optimization

    An interface to be used on `Chromosome`s that will be using local
    optimization.
    """
    @abstractmethod
    def needs_local_optimization(self):
        """Does the `Chromosome` need local optimization

        Returns
        -------
        bool
            Whether `Chromosome` needs optimization
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