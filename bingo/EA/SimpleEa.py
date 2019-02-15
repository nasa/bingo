"""The base of evolutionary algorithm definition

This module defines the basis of evolutionary algorithms in bingo analyses.
"""
from ..Base.EvolutionaryAlgorithm import EvolutionaryAlgorithm


class SimpleEa(EvolutionaryAlgorithm):
    """The algorithm used to perform generational steps.

    An abstract base class for evolutionary algorithms in bingo in bingo.
    """
    def __init__(self, variation, evaluation, selection):
        self._variation = variation
        self._evaluation = evaluation
        self._selection = selection

    def generational_step(self, population):
        """Performs selection on individuals.

        Parameters
        ----------
        population : list of Chromosome
                     The population at the start of the generational step

        Returns
        -------
        list of Chromosome :
            The next generation of the population
        """
        population_size = len(population)
        offspring = self._variation(population, population_size)
        self._evaluation(offspring)
        next_generation = self._selection(offspring, population_size)
        return next_generation
