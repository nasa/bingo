"""Generator of individuals

This module covers the template for generation of individuals in the genetic
evolution in bingo
"""
from abc import ABCMeta, abstractmethod


class IndividualGeneration(metaclass=ABCMeta):
    """A generator of individuals.

    An abstract base class for the generation of genetic individuals in
    bingo.
    """
    @abstractmethod
    def __call__(self):
        """Generates individuals for evolutionary optimization

        Returns
        -------
        GeneticIndividual :
            A newly generated individual
        """
        raise NotImplementedError


class IndividualMutation(metaclass=ABCMeta):
    """A mutator of individuals.

    An abstract base class for the mutation of genetic individuals in
    bingo.
    """
    @abstractmethod
    def __call__(self, individual):
        """Generates individuals for evolutionary optimization

        Parameters
        ----------
        individual : GeneticIndividual
                     The individual to be mutated.

        Returns
        -------
        GeneticIndividual :
            A mutated generated individual
        """
        raise NotImplementedError


class IndividualCrossover(metaclass=ABCMeta):
    """Crossover for individuals.

    An abstract base class for the crossovor between two genetic
    individuals in bingo.
    """
    @abstractmethod
    def __call__(self, individual):
        """Generates individuals for evolutionary optimization

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
