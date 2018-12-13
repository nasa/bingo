"""Local optimization of continuous, real-valued parameters

This module contains the implementation of local optimization or continuous,
real-valued parameters in a chromosome.  The local optimization algorithm is
defined as well as the interface that must implemented by chromosomes wishing
to use the functionality.
"""
from abc import ABCMeta, abstractmethod


class ChromosomeInterface(metaclass=ABCMeta):
    """For chromosomes with continuous local optimization

    An interface to be used on Chromosomes that will be using continuous local
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
