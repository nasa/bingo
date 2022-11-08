"""Evolutionary algorithm diagnostics

EA diagnostics are tracked to allow for investigating convergence properties,
etc.  Currently ony diagnostics associated with the variation phase of a EA are
tracked.
"""
from collections import namedtuple
from itertools import product

import numpy as np

EaDiagnosticsSummary = namedtuple(
    "EaDiagnosticsSummary",
    [
        "beneficial_crossover_rate",
        "detrimental_crossover_rate",
        "beneficial_mutation_rate",
        "detrimental_mutation_rate",
        "beneficial_crossover_mutation_rate",
        "detrimental_crossover_mutation_rate",
    ],
)
GeneticOperatorSummary = namedtuple(
    "GeneticOperatorSummary", ["beneficial_rate", "detrimental_rate"]
)


class EaDiagnostics:
    """Evolutionary Algorithm Diagnostic Information

    EA diagnostics are tracked to allow for investigating convergence
    properties, etc.  Currently ony diagnostics associated with the variation
    phase of a EA are tracked.

    Parameters
    ----------
    crossover_types : iterable of str, optional
        possible crossover types (excluding None)
    mutation_types : iterable of str, optional
        possible mutation types (excluding None)

    Attributes
    ----------
    summary : `EaDiagnosticsSummary`
        namedtuple describing the summary of the diagnostic information
    crossover_type_summary : dict(str: `GeneticOperatorSummary`)
        dict mapping crossover types to `GeneticOperatorSummary`, describing
        the diagnostic information of cases when only a particular crossover 
        type was applied
    mutation_type_summary : dict(str: `GeneticOperatorSummary`)
        dict mapping mutation types to `GeneticOperatorSummary`, describing
        the diagnostic information of cases when only a particular mutation 
        type was applied
    crossover_mutation_type_summary : dict(tuple(str, str): `GeneticOperatorSummary`)  # pylint: disable=line-too-long
        dict mapping a tuple of crossover type and mutation type (in that order)
        to the diagnostic information of cases when both the crossover
        and mutation type were applied
    """

    def __init__(self, crossover_types=None, mutation_types=None):
        self._crossover_types = (
            [] if crossover_types is None else crossover_types
        )
        self._mutation_types = [] if mutation_types is None else mutation_types

        self._crossover_stats = np.zeros(
            (len(self._crossover_types), 3), dtype=int
        )
        self._mutation_stats = np.zeros(
            (len(self._mutation_types), 3), dtype=int
        )
        self._cross_mut_stats = np.zeros(
            (len(self._crossover_types) * len(self._mutation_types), 3),
            dtype=int,
        )
        #

    @property
    def summary(self):  # TODO
        """Summary statistics of the diagnostic data"""

        cross_tots = self._crossover_stats.sum(axis=0)
        mut_tots = self._mutation_stats.sum(axis=0)
        cross_mut_tots = self._cross_mut_stats.sum(axis=0)
        return EaDiagnosticsSummary(
            cross_tots[1] / cross_tots[0],
            cross_tots[2] / cross_tots[0],
            mut_tots[1] / mut_tots[0],
            mut_tots[2] / mut_tots[0],
            cross_mut_tots[1] / cross_mut_tots[0],
            cross_mut_tots[2] / cross_mut_tots[0],
        )

    @property
    def crossover_type_summary(self):
        """Summary of diagnostic data when only crossover happened"""
        summary = {}
        for crossover_type, type_stats in zip(
            self._crossover_types, self._crossover_stats
        ):
            summary[crossover_type] = GeneticOperatorSummary(
                type_stats[1] / type_stats[0], type_stats[2] / type_stats[0]
            )
        return summary

    @property
    def mutation_type_summary(self):
        """Summary of diagnostic data when only mutation happened"""
        summary = {}
        for mutation_type, type_stats in zip(
            self._mutation_types, self._mutation_stats
        ):
            summary[mutation_type] = GeneticOperatorSummary(
                type_stats[1] / type_stats[0], type_stats[2] / type_stats[0]
            )
        return summary

    @property
    def crossover_mutation_type_summary(self):
        """Summary of diagnostic data when both crossover and
        mutation happened"""
        summary = {}
        for type_pairing, pair_stats in zip(
            product(self._crossover_types, self._mutation_types),
            self._cross_mut_stats,
        ):
            summary[type_pairing] = GeneticOperatorSummary(
                pair_stats[1] / pair_stats[0], pair_stats[2] / pair_stats[0]
            )
        return summary

    def get_log_header(self):
        """Gets a comma separated header for use in diagnstics logging

        Returns
        -------
        str :
            A comma separated desription of stats that will be logged
        """
        header = (
            "crossover_number, crossover_beneficial, crossover_detrimental, "
            "mutation_number, mutation_beneficial, mutation_detrimental, "
            "crossover_mutation_number, crossover_mutation_beneficial, "
            "crossover_mutation_detrimental"
        )
        for c in self._crossover_types:
            header += f", {c}_number, {c}_beneficial, {c}_detrimental"
        for m in self._mutation_types:
            header += f", {m}_number, {m}_beneficial, {m}_detrimental"
        for c, m in product(self._crossover_types, self._mutation_types):
            header += (
                f", {c}_{m}_number, {c}_{m}_beneficial, {c}_{m}_detrimental"
            )
        return header

    def get_log_stats(self):
        """Gets a list of statistics that describe the effects of each type of 
        genetic variation

        Return:
        array :
            the number, beneficial (improved firness compared to parents), and 
            detrimental (worsened compared to parents) for each possible 
            variation
        """

        stats = np.hstack(
            (
                self._crossover_stats.sum(axis=0),
                self._mutation_stats.sum(axis=0),
                self._cross_mut_stats.sum(axis=0),
                self._crossover_stats.flatten(),
                self._mutation_stats.flatten(),
                self._cross_mut_stats.flatten(),
            )
        ).flatten()
        return stats

    def update(
        self,
        population,
        offspring,
        offspring_parents,
        crossover_offspring_type,
        mutation_offspring_type,
    ):
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
        crossover_offspring_type : numpy array of str
            numpy array indicating the crossover type that the
            corresponding offspring underwent (or None)
        mutation_offspring_type : numpy array of str
            numpy array indicating the mutation type that the
            corresponding offspring underwent (or None)
        """
        beneficial_var = np.zeros(len(offspring), dtype=bool)
        detrimental_var = np.zeros(len(offspring), dtype=bool)
        for i, (child, parent_indices) in enumerate(
            zip(offspring, offspring_parents)
        ):
            if len(parent_indices) == 0:
                continue
            beneficial_var[i] = all(
                child.fitness < population[p].fitness for p in parent_indices
            )
            detrimental_var[i] = all(
                child.fitness > population[p].fitness for p in parent_indices
            )

        crossover_idx = np.array(
            [crossover_offspring_type == i for i in self._crossover_types]
        )
        mutation_idx = np.array(
            [mutation_offspring_type == i for i in self._mutation_types]
        )
        cross_only_idx = crossover_idx & ~mutation_idx.any(axis=0)
        mut_only_idx = mutation_idx & ~crossover_idx.any(axis=0)
        cross_mut_idx = (
            crossover_idx[:, np.newaxis, :] * mutation_idx[np.newaxis, :, :]
        ).reshape(-1, len(offspring))

        self._crossover_stats += self._get_stats(
            cross_only_idx, beneficial_var, detrimental_var
        )
        self._mutation_stats += self._get_stats(
            mut_only_idx, beneficial_var, detrimental_var
        )
        self._cross_mut_stats += self._get_stats(
            cross_mut_idx, beneficial_var, detrimental_var
        )

    def _get_stats(self, idx, beneficial_var, detrimental_var):
        return np.count_nonzero(
            [idx, beneficial_var & idx, detrimental_var & idx], axis=2
        ).T

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
