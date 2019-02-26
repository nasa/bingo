"""The "Mu + Lambda"

This module defines the basis of the "mu plus lambda"
evolutionary algorithm in bingo analyses.
"""
from bingo.Base.EvolutionaryAlgorithm import EvolutionaryAlgorithm

class MuPlusLambda(EvolutionaryAlgorithm):
    """The algorithm used to perform generational steps.

    A class for the "mu plus lambda" evolutionary algorithm in bingo.
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
        self._evaluation(population)
        number_offspring = 20
        offspring = self._variation(population, len(population))
        self._evaluation(offspring)
        return self._selection(population + offspring, len(population))
