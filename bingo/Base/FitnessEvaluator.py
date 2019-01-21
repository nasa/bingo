"""The definition of fitness evalutions for individuals.

This module defines the basis of fitness evaluation in bingo evolutionary
analyses.
"""

from abc import ABCMeta, abstractmethod


class FitnessEvaluator(metaclass=ABCMeta):
    """Fitness evaluation metric for individuals.

    An abstract base class for the fitness evaluation of genetic individuals
    (Chromosomes) in bingo.

    Attributes
    ----------
    eval_count : int
                 the number of evaluations that have been performed
    """
    def __init__(self):
        self.eval_count = 0

    @abstractmethod
    def __call__(self, individual):
        """Evaluates the fitness of an individual

        Parameters
        ----------
        individual : Chromosome
                     individual for which fitness should be calculated


        Notes
        -----
        The eval_count should be incremented in a subclass' __call__ definition
        for accurate evaluation counting

        Returns
        -------
         :
            fitness of the individual
        """
        raise NotImplementedError