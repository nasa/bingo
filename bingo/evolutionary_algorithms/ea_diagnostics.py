"""Evolutionary algorithm diagnostics

EA diagnostics are tracked to allow for investigating convergence properties,
etc.  Currently ony diagnostics associated with the variation phase of a EA are
tracked.
"""
from collections import namedtuple
from itertools import product

import numpy as np

EaDiagnosticsSummary = namedtuple("EaDiagnosticsSummary",
                                  ["beneficial_crossover_rate",
                                   "detrimental_crossover_rate",
                                   "beneficial_mutation_rate",
                                   "detrimental_mutation_rate",
                                   "beneficial_crossover_mutation_rate",
                                   "detrimental_crossover_mutation_rate"])
GeneticOperatorSummary = namedtuple("GeneticOperatorSummary",
                                    ["beneficial_rate", "detrimental_rate"])


class EaDiagnostics:
    """Evolutionary Algorithm Diagnostic Information

    EA diagnostics are tracked to allow for investigating convergence
    properties, etc.  Currently ony diagnostics associated with the variation
    phase of a EA are tracked.

    Attributes
    ----------
    summary : `EaDiagnosticsSummary`
        namedtuple describing the summary of the diagnostic information

    """
    def __init__(self, crossover_types, mutation_types):
        self._crossover_stats = np.zeros(3)
        self._mutation_stats = np.zeros(3)
        self._cross_mut_stats = np.zeros(3)
        self._crossover_types = crossover_types
        self._mutation_types = mutation_types

        self._crossover_type_stats = \
            {type_: np.zeros(3) for type_ in crossover_types}
        self._mutation_type_stats = \
            {type_: np.zeros(3) for type_ in mutation_types}
        self._crossover_mutation_type_stats = \
            {type_pair: np.zeros(3) for type_pair in product(crossover_types,
                                                             mutation_types)}

    @property
    def summary(self):
        """Summary statistics of the diagnostic data"""
        return EaDiagnosticsSummary(
            self._crossover_stats[1] / self._crossover_stats[0],
            self._crossover_stats[2] / self._crossover_stats[0],
            self._mutation_stats[1] / self._mutation_stats[0],
            self._mutation_stats[2] / self._mutation_stats[0],
            self._cross_mut_stats[1] / self._cross_mut_stats[0],
            self._cross_mut_stats[2] / self._cross_mut_stats[0])

    @property
    def crossover_type_summary(self):
        summary = {}
        for crossover_type in self._crossover_types:
            type_stats = self._crossover_type_stats[crossover_type]
            summary[crossover_type] = GeneticOperatorSummary(
                type_stats[1] / type_stats[0],
                type_stats[2] / type_stats[0])
        return summary

    @property
    def mutation_type_summary(self):
        summary = {}
        for mutation_type in self._mutation_types:
            type_stats = self._mutation_type_stats[mutation_type]
            summary[mutation_type] = GeneticOperatorSummary(
                type_stats[1] / type_stats[0],
                type_stats[2] / type_stats[0])
        return summary

    @property
    def crossover_mutation_type_summary(self):
        summary = {}
        for type_pairing in product(self._crossover_types,
                                    self._mutation_types):
            pair_stats = self._crossover_mutation_type_stats[type_pairing]
            summary[type_pairing] = GeneticOperatorSummary(
                pair_stats[1] / pair_stats[0],
                pair_stats[2] / pair_stats[0])
        return summary
    
    def _get_stats(self, idx, beneficial_var, detrimental_var):
        return sum(idx), sum(beneficial_var * idx), sum(detrimental_var * idx)

    def update(self, population, offspring, offspring_parents,
               offspring_crossover_type, offspring_mutation_type):
        """Updates the diagnostic information associated with a single step in
        an EA

        Parameters
        ----------
        population : list of `Chromosome`
            population at the beginning of the generational step
        offspring : list of `Chromosome`
            the result of the EAs variation phase
        offspring_parents : list of list of int
            list indicating the parents (by index in population) of the
            corresponding member of offspring
        offspring_crossover_type : numpy array of object
            numpy array indicating the crossover type that the
            corresponding offspring underwent
        offspring_mutation_type : numpy array of object
            numpy array indicating the mutation type that the
            corresponding offspring underwent
        """
        offspring_crossover = offspring_crossover_type.astype(bool)
        offspring_mutation = offspring_mutation_type.astype(bool)

        beneficial_var = np.zeros(len(offspring), dtype=bool)
        detrimental_var = np.zeros(len(offspring), dtype=bool)
        for i, (child, parent_indices) in \
                enumerate(zip(offspring, offspring_parents)):
            if len(parent_indices) == 0:
                continue
            beneficial_var[i] = \
                all(child.fitness < population[p].fitness
                    for p in parent_indices)
            detrimental_var[i] = \
                all(child.fitness > population[p].fitness
                    for p in parent_indices)

        just_cross = offspring_crossover * ~offspring_mutation
        just_mut = ~offspring_crossover * offspring_mutation
        cross_mut = offspring_crossover * offspring_mutation
        self._crossover_stats += self._get_stats(just_cross, beneficial_var,
                                                 detrimental_var)
        self._mutation_stats += self._get_stats(just_mut, beneficial_var,
                                                 detrimental_var)
        self._cross_mut_stats += self._get_stats(cross_mut, beneficial_var,
                                                 detrimental_var)

        for crossover_type in self._crossover_types:
            cross_idx = offspring_crossover_type == crossover_type
            self._crossover_type_stats[crossover_type] += self._get_stats(
                just_cross[cross_idx], beneficial_var[cross_idx],
                detrimental_var[cross_idx])

            for mutation_type in self._mutation_types:
                mut_idx = offspring_mutation_type == mutation_type
                self._mutation_type_stats[mutation_type] += self._get_stats(
                    just_mut[mut_idx], beneficial_var[mut_idx],
                    detrimental_var[mut_idx])
                
                cross_mut_idx = np.logical_and(cross_idx, mut_idx)
                self._crossover_mutation_type_stats[
                    (crossover_type, mutation_type)] += self._get_stats(
                    cross_mut[cross_mut_idx], beneficial_var[cross_mut_idx],
                    detrimental_var[cross_mut_idx])

    def __add__(self, other):
        sum_ = EaDiagnostics(self._crossover_types, self._mutation_types)
        sum_._crossover_stats = self._crossover_stats + other._crossover_stats
        sum_._mutation_stats = self._mutation_stats + other._mutation_stats
        sum_._cross_mut_stats = self._cross_mut_stats + other._cross_mut_stats
        return sum_

    def __radd__(self, other):
        if other == 0:
            return self
        raise NotImplementedError
