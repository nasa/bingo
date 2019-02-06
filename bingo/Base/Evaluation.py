"""The genetic operation of Evaluation.

This module defines the basis of the evaluation phase of bingo evolutionary
algorithms.
"""

from abc import ABCMeta, abstractmethod


class Evaluation(metaclass=ABCMeta):
    """EA phase for calculating fitness of a population.

    An abstract base class for the fitness evaluation of populations of
    genetic individuals (list of Chromosomes) in bingo.
    """
    @abstractmethod
    def __call__(self, population):
        """Evaluates the fitness of an individual

        Parameters
        ----------
        population : list of Chromosome
                     population for which fitness should be calculated
        """
        raise NotImplementedError