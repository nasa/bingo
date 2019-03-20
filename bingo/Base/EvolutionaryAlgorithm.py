"""The base of evolutionary algorithm definition

This module defines the basis of evolutionary algorithms in bingo analyses.
"""

from abc import ABCMeta, abstractmethod


class EvolutionaryAlgorithm(metaclass=ABCMeta):
    """The algorithm used to perform generational steps.

    An abstract base class for evolutionary algorithms in bingo in bingo.
    """
    @abstractmethod
    def generational_step(self, population):
        """Performs selection on individuals.

        Parameters
        ----------
        population : list of Chromosome
                     The population at the start of the generational step

        Returns
        -------
        list of Chromosome :
            The next generation of the population
        """
        raise NotImplementedError
