"""Generator of individuals

This module covers the template for generation of individuals in the genetic
evolution in bingo
"""
from abc import ABCMeta, abstractmethod


class IndividualGenerator(metaclass=ABCMeta):
    """A generator of individuals.

    An abstract base class for the generation of evolutionary individuals in
    bingo.
    """
    @abstractmethod
    def generate(self):
        """Generates individuals for evolutionary optimization

        Returns
        -------
        GeneticIndividual :
            A newly generated individual
        """
        raise NotImplementedError
