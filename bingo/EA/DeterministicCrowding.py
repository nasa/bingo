"""The "Deterministic Crowding" evolutionary algorithm

This module defines the basis of the "deterministic crowding"
evolutionary algorithm in bingo analyses. The next generation
is selected by pairing parents with their offspring and
advancing the most fit of the two.
"""
from ..Base.EvolutionaryAlgorithm import EvolutionaryAlgorithm
from bingo.EA.VarAnd import VarAnd
from bingo.EA.DeterministicCrowdingSelection import DeterministicCrowdingSelection

class DeterministicCrowdingEA(EvolutionaryAlgorithm):
    """The algorithm used to perform generational steps.

    A class for the "mu comma lambda" evolutionary algorithm in bingo.

    Attributes
    ----------
    evaluation : Evaluation
                 Evaluation instance to perform evaluation on a population
    selection : DeterministicCrowdingSelection
                Performs selection on a population via deterministic crowding
    variation : VarAnd
                Performs VarAnd variation on a population
    """
    def __init__(self, evaluation, crossover, mutation, crossover_probability,
                 mutation_probability):
        self._evaluation = evaluation
        self._selection = DeterministicCrowdingSelection()
        self._variation = VarAnd(crossover, mutation, crossover_probability,
                                 mutation_probability)

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
        offspring = self._variation(population, len(population))
        self._evaluation(population + offspring)
        return self._selection(population + offspring, len(population))
