"""The base of evolutionary algorithm definition

This module defines the basis of evolutionary algorithms in bingo analyses.
An Base in bingo is defined by three phases: variation, evaluation, and
selection.  These phases, when repeated, define the evolution of a population.
"""
from collections import namedtuple

import numpy as np

EaDiagnosticsSummary = namedtuple("EaDiagnosticsSummary",
                                  ["beneficial_crossover_rate",
                                   "detrimental_crossover_rate",
                                   "beneficial_mutation_rate",
                                   "detrimental_mutation_rate",
                                   "beneficial_crossover_mutation_rate",
                                   "detrimental_crossover_mutation_rate"])

class EvolutionaryAlgorithm():
    """The algorithm used to perform generational steps.

    The basic implementation used in this base Base implementation is a simple
    steady-state Base (akin to simpleEA in DEAP)

    Parameters
    ----------
    variation : `variation`
                The phase of bingo EAs that is responsible for varying the
                population (usually through some form of crossover and/or
                mutation).
    evaluation : `evaluation`
                 The phase in bingo EAs responsible for the evaluation of the
                 fitness of the individuals in the population.
    selection : `selection`
                The phase of bingo EAs responsible for selecting the
                individuals in a population which survive into the next
                generation.

    Attributes
    ----------
    variation : `variation`
        Public to the variation phase of the Base
    evaluation : `evaluation`
        Public to the evaluation phase of the Base
    selection : `selection`
        Public to the selection phase of the Base
    """
    def __init__(self, variation, evaluation, selection):
        self.variation = variation
        self.evaluation = evaluation
        self.selection = selection
        self.diagnostics = EaDiagnostics()

    def generational_step(self, population):
        """Performs a generational step on population.

        Parameters
        ----------
        population : list of chromosomes
            The population at the start of the generational step

        Returns
        -------
        list of chromosomes :
            The next generation of the population
        """
        population_size = len(population)
        offspring = self.variation(population, population_size)
        self.evaluation(offspring)
        next_generation = self.selection(offspring, population_size)
        self.update_diagnostics(population, offspring)
        return next_generation

    def update_diagnostics(self, population, offspring):
        self.diagnostics.update(population, offspring,
                                self.variation.offspring_parents,
                                self.variation.crossover_offspring,
                                self.variation.mutation_offspring)


class EaDiagnostics:
    def __init__(self):
        self._crossover_stats = np.zeros(3)
        self._mutation_stats = np.zeros(3)
        self._cross_mut_stats = np.zeros(3)

    @property
    def summary(self):
        return EaDiagnosticsSummary(
                self._crossover_stats[1] / self._crossover_stats[0],
                self._crossover_stats[2] / self._crossover_stats[0],
                self._mutation_stats[1] / self._mutation_stats[0],
                self._mutation_stats[2] / self._mutation_stats[0],
                self._cross_mut_stats[1] / self._cross_mut_stats[0],
                self._cross_mut_stats[2] / self._cross_mut_stats[0])

    def update(self, population, offspring, offspring_parents,
               offspring_crossover, offspring_mutation):
        beneficial_var = np.zeros(len(offspring), dtype=bool)
        detrimental_var = np.zeros(len(offspring), dtype=bool)
        for i, (child, parent_indices) in \
                enumerate(zip(offspring, offspring_parents)):
            if len(parent_indices) == 0:
                continue
            beneficial_var[i] = \
                all([child.fitness < population[p].fitness
                     for p in parent_indices])
            detrimental_var[i] = \
                all([child.fitness > population[p].fitness
                     for p in parent_indices])

        just_cross = offspring_crossover * ~offspring_mutation
        just_mut = ~offspring_crossover * offspring_mutation
        cross_mut = offspring_crossover * offspring_mutation
        self._crossover_stats += (sum(just_cross),
                                  sum(beneficial_var * just_cross),
                                  sum(detrimental_var * just_cross))
        self._mutation_stats += (sum(just_mut),
                                 sum(beneficial_var * just_mut),
                                 sum(detrimental_var * just_mut))
        self._cross_mut_stats += (sum(cross_mut),
                                  sum(beneficial_var * cross_mut),
                                  sum(detrimental_var * cross_mut))

    def __add__(self, other):
        sum_ = EaDiagnostics()
        sum_._crossover_stats = self._crossover_stats + other._crossover_stats
        sum_._mutation_stats = self._mutation_stats + other._mutation_stats
        sum_._cross_mut_stats = self._cross_mut_stats + other._cross_mut_stats
        return sum_

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)
