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
    crossover_offspring : list of bool
                          list indicating whether the corresponding member of
                          the last offspring was a result of crossover
    mutation_offspring : list of bool
                         list indicating whether the corresponding member of
                         the last offspring was a result of mutation
    """
    def __init__(self):
        self.crossover_offspring = []
        self.mutation_offspring = []

    @abstractmethod
    def __call__(self, population, number_offspring):
        """Performs variation on a population.

        Parameters
        ----------
        population : list of Chromosome
                     The population on which to perform selection
        number_offspring : int
                           number of offspring to produce

        Returns
        -------
        list of Chromosome :
            The offspring of the population
        """
        raise NotImplementedError
