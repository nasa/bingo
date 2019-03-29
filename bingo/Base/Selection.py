"""The genetic operation of Selection.

This module defines the basis of selection in bingo evolutionary analyses.
"""

from abc import ABCMeta, abstractmethod


class Selection(metaclass=ABCMeta):
    """A selector of fit individuals.

    An abstract base class for the selection of genetic individuals
    (Chromosomes) in bingo.
    """
    @abstractmethod
    def __call__(self, population, target_population_size):
        """Performs selection on individuals.

        Parameters
        ----------
        population : list of Chromosome
                     The population on which to perform selection
        target_population_size : int
                                 Target size of the population after selection

        Returns
        -------
        list of Chromosome :
            A subset of the input population
        """
        raise NotImplementedError
