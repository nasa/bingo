"""The "Mu + Lambda"

This module defines the basis of the "mu plus lambda"
evolutionary algorithm in bingo analyses. The next generation
is evaluated and selected from both the parent and offspring
populations.
"""
from bingo.Base.EvolutionaryAlgorithm import EvolutionaryAlgorithm
from bingo.EA.VarOr import VarOr


class MuPlusLambda(EvolutionaryAlgorithm):
    """The algorithm used to perform generational steps.

    A class for the "mu plus lambda" evolutionary algorithm in bingo.

    Attributes
    ----------
    variation : VarOr
                VarOr variation to perform variation on a population
    evaluation : Evaluation
                 Evaluation instance to perform evaluation on a population
    selection : Selection
                Selection instance to perform selection on a population
    number_offspring : int
                       The desired size of the offspring population

    """
    def __init__(self, evaluation, selection, crossover, mutation,
                 crossover_probability, mutation_probability,
                 number_offspring):
        self._variation = VarOr(crossover, mutation, crossover_probability,
                                mutation_probability)
        self._evaluation = evaluation
        self._selection = selection
        self._number_offspring = number_offspring

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
        offspring = self._variation(population, self._number_offspring)
        self._evaluation(population + offspring)
        return self._selection(population + offspring, len(population))
