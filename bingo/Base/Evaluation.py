"""The genetic operation of Evaluation.

This module defines the basis of evaluation in bingo evolutionary analyses.
"""

from abc import ABCMeta, abstractmethod


class Evaluation(metaclass=ABCMeta):
    """Fitness evaluation metric for individuals.

    An abstract base class for the fitness evaluation of genetic individuals
    (Chromosomes) in bingo.
    """
    @abstractmethod
    def __call__(self, individual):
        """Evaluates the fitness of an individual

        Parameters
        ----------
        individual : Chromosome
                     individual for which fitness should be calculated

        Returns
        -------
         :
            fitness of the individual
        """
        raise NotImplementedError