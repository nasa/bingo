"""The genetic operation of variation.

This module defines the basis of variation in bingo evolutionary analyses.
Generally this consists of crossover, mutation and replication.
"""

from abc import ABCMeta, abstractmethod
import numpy as np


class Variation(metaclass=ABCMeta):
    """A variator of individuals.

    An abstract base class for the variation of genetic populations
    (list chromosomes) in bingo.

    Parameters
    ----------
    crossover_types : iterable of str, optional
        possible types of crossover (excluding None)
    mutation_types : iterable of str, optional
        possible types of mutation (excluding None)

    Attributes
    ----------
    crossover_offspring_type : numpy array of str
        numpy array indicating the crossover type that the
        corresponding offspring underwent (or None)
    crossover_types : iterable of str
        possible types of crossover (excluding None)
    mutation_offspring_type : numpy array of str
        numpy array indicating the mutation type that the
        corresponding offspring underwent (or None)
    mutation_types : iterable of str
        possible types of mutation (excluding None)
    offspring_parents : list of list of int
        list indicating the parents (by index in the population) of the
        corresponding member of the last offspring
    """

    def __init__(self, crossover_types=None, mutation_types=None):
        self.crossover_offspring_type = np.zeros(shape=(0,), dtype=object)
        self._crossover_types = [] if crossover_types is None else crossover_types

        self.mutation_offspring_type = np.zeros(shape=(0,), dtype=object)
        self._mutation_types = [] if mutation_types is None else mutation_types
        self.offspring_parents = []

    @property
    def crossover_types(self):
        """Types of crossover allowed in variation"""
        return self._crossover_types

    @property
    def mutation_types(self):
        """Types of mutation allowed in variation"""
        return self._mutation_types

    @abstractmethod
    def __call__(self, population, number_offspring):
        """Performs variation on a population.

        Parameters
        ----------
        population : list of chromosomes
                     The population on which to perform selection
        number_offspring : int
                           number of offspring to produce

        Returns
        -------
        list of chromosomes :
            The offspring of the population
        """
        raise NotImplementedError
