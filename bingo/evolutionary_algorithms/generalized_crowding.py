"""The generalized crowding evolutionary algorithm

This module defines the basis of the generalized crowding evolutionary
algorithm in bingo analyses. The next generation is selected by pairing parents
with their offspring and advancing the most fit of the two.
"""
import numpy as np

from .evolutionary_algorithm import EvolutionaryAlgorithm
from ..variation.var_and import VarAnd
from ..selection.deterministic_crowding import DeterministicCrowding


class GeneralizedCrowdingEA(EvolutionaryAlgorithm):
    """The algorithm used to perform generational steps.

    A class for the generalized crowding evolutionary algorithm in bingo.

    Parameters
    ----------
    evaluation : evaluation
        The evaluation algorithm that sets the fitness on the population.
    crossover : Crossover
        The algorithm that performs crossover during variation.
    mutation : Mutation
        The algorithm that performs mutation during variation.
    crossover_probability : float
        Probability that crossover will occur on an individual.
    mutation_probability : float
        Probability that mutation will occur on an individual.
    selection : CrowdingSelection
        Selection phase. Default DeterministicCrowding.

    Attributes
    ----------
    evaluation : Evaluation
        evaluation instance to perform evaluation on a population
    selection : CrowdingSelection
        Performs selection on a population via deterministic crowding
    variation : VarAnd
        Performs VarAnd variation on a population
    diagnostics : `bingo.evolutionary_algorithms.ea_diagnostics.EaDiagnostics`
        Public to the EA diagnostics
    """

    def __init__(
        self,
        evaluation,
        crossover,
        mutation,
        crossover_probability,
        mutation_probability,
        selection=None,
    ):
        if selection is None:
            selection = DeterministicCrowding()

        super().__init__(
            variation=VarAnd(
                crossover,
                mutation,
                crossover_probability,
                mutation_probability,
            ),
            evaluation=evaluation,
            selection=selection,
        )

    def generational_step(self, population):
        """Performs selection on individuals.

        Parameters
        ----------
        population : list of chromosomes
            The population at the start of the generational step

        Returns
        -------
        list of chromosomes :
            The next generation of the population
        """
        offspring = self.variation(population, len(population))
        self.evaluation(population)
        self.evaluation(offspring)
        self.update_diagnostics(population, offspring)
        next_gen = self.selection(population + offspring, len(population))
        np.random.shuffle(next_gen)
        return next_gen
