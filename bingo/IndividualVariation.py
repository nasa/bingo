"""Generator of individuals

This module covers the template for generation of individuals in the genetic
evolution in bingo
"""
from abc import ABCMeta, abstractmethod


class Generation(metaclass=ABCMeta):
    """A generator of individuals.

    An abstract base class for the generation of genetic individuals in
    bingo.
    """
    @abstractmethod
    def __call__(self):
        """Generates individuals

        Returns
        -------
        GeneticIndividual :
            A newly generated individual
        """
        raise NotImplementedError


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


class Crossover(metaclass=ABCMeta):
    """Crossover for individuals.

    An abstract base class for the crossovor between two genetic
    individuals in bingo.
    """
    @abstractmethod
    def __call__(self, parent_1, parent_2):
        """Crossover between two individuals

        Parameters
        ----------
        parent_1 : GeneticIndividual
                   The first parent individual
        parent_2 : GeneticIndividual
                   The second parent individual

        Returns
        -------
        tuple(GeneticIndividual, GeneticIndividual) :
            The two children from the crossover.
        """
        raise NotImplementedError
