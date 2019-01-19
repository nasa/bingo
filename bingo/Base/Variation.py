"""The genetic operation of Variation.

This module defines the basis of variation in bingo evolutionary analyses.
Generally this consists of crossover, mutation and replication.
"""

from abc import ABCMeta, abstractmethod


class Variation(metaclass=ABCMeta):
    """A variator of individuals.

    An abstract base class for the variation of genetic populations
    (list Chromosomes) in bingo.

    Attributes
    ----------
    stats : list of int
            A coded integer for each member of the last offspring indicating
            the reproduction method for that individual
            0 = Replication
            1 = Crossover
            2 = Mutation
            3 = Crossover and Mutation
    """
    def __init__(self):
        self.stats = []

    @abstractmethod
    def __call__(self, population):
        """Performs variation on a population.

        Parameters
        ----------
        population : list of Chromosome
                     The population on which to perform selection

        Returns
        -------
        list of Chromosome :
            The offspring of the population
        """
        raise NotImplementedError