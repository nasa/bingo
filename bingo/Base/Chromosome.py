"""Chromosome: the base class for all genetic individuals in bingo

This module defines a Chromosome, the base class for all genetic individuals in
bingo evolutionary analyses.  Extension of bingo can be performed by developing
subclasses of Chromosome.
"""

import copy
from abc import ABCMeta, abstractmethod


class Chromosome(metaclass=ABCMeta):
    """A genetic individual

    This class is the base of a genetic individual in bingo evolutionary
    analyses.

    Attributes
    ----------
    fitness
    genetic_age : int
                  age of the oldest component of the genetic material in the
                  individual
    fit_set : bool
              Whether the fitness has been calculated for the individual
    """
    def __init__(self):
        self.genetic_age = 0
        self._fitness = None
        self.fit_set = False

    @property
    def fitness(self):
        """numeric or tuple of numeric: The fitness of the individual"""
        return self._fitness

    @fitness.setter
    def fitness(self, fitness):
        self._fitness = fitness
        self.fit_set = True

    def copy(self):
        """copy

        Returns
        -------
            A deep copy of self
        """
        return copy.deepcopy(self)

    @abstractmethod
    def __str__(self):
        """String conversion of individual

        Returns
        -------
        str
            individual string form
        """
        raise NotImplementedError

    @abstractmethod
    def distance(self, chromosome):
        """Distance from self to chromosome

        Returns
        -------
        float
            distance from self to chromosome
        """
        raise NotImplementedError
