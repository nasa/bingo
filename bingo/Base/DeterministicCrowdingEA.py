"""The "Deterministic Crowding" evolutionary algorithm

This module defines the basis of the "deterministic crowding"
evolutionary algorithm in bingo analyses. The next generation
is selected by pairing parents with their offspring and
advancing the most fit of the two.
"""
from .EvolutionaryAlgorithm import EvolutionaryAlgorithm
from .VarAnd import VarAnd
from .DeterministicCrowdingSelection import DeterministicCrowdingSelection


class DeterministicCrowdingEA(EvolutionaryAlgorithm):
    """The algorithm used to perform generational steps.

    A class for the deterministic crowding evolutionary algorithm in bingo.

    Parameters
    ----------
    evaluation : Evaluation
        The evaluation algorithm that sets the fitness on the population.
    crossover : Crossover
        The algorithm that performs crossover during variation.
    mutation : Mutation
        The algorithm that performs mutation during variation.
    crossover_probability : float
        Probability that crossover will occur on an individual.
    mutation_probability : float
        Probability that mutation will occur on an individual.

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
        super().__init__(variation=VarAnd(crossover, mutation,
                                          crossover_probability,
                                          mutation_probability),
                         evaluation=evaluation,
                         selection=DeterministicCrowdingSelection())

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
        offspring = self.variation(population, len(population))
        self.evaluation(population + offspring)
        return self.selection(population + offspring, len(population))
