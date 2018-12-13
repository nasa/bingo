"""The genetic operation of Mutation.

This module defines the basis of mutation in bingo evolutionary analyses.
"""

from abc import ABCMeta, abstractmethod


class Mutation(metaclass=ABCMeta):
    """A mutator of individuals.

    An abstract base class for the mutation of genetic individuals in
    bingo.
    """
    @abstractmethod
    def __call__(self, parent):
        """Mutates individuals

        Parameters
        ----------
        parent : GeneticIndividual
                     The individual to be mutated.

        Returns
        -------
        GeneticIndividual :
            A mutated generated individual
        """
        raise NotImplementedError
