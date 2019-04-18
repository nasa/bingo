"""Age-Fitness selection

This module implements the Age-Fitness selection algorithm that defines
the selection used in the Age-Fitness evolutionary algorithm module.
This module expects to be used in conjunction with the
``RandomIndividualVariation`` module that wraps the ``VarOr`` module.
"""
import numpy as np
import random

from .Selection import Selection
from ..Util.ArgumentValidation import argument_validation


class AgeFitness(Selection):
    """Age-Fitness selection

    Parameters
    ----------
    selection_size : int
        The size of the group of individuals to be randomly
        compared. The size must be an integer greater than 1.
    """
    WORST_CASE_FACTOR = 50

    @argument_validation(selection_size={">=": 2})
    def __init__(self, selection_size=2):
        self._selection_size = selection_size
        self._selected_indices = []
        self._population_index_array = np.array([])
        self._selection_attempts = 0

    def __call__(self, population, target_population_size):
        """Performs Age-Fitness selection on a population. If ``selection_size``
        is larger than the population, the population size is used as the
        ``selection_size``.

        Parameters
        ----------
        population : list of Chromosome
            The population on which to perform selection
        target_population_size : int
            The size of the new population after selection. It will never be the
            case that the new population will have a size smaller than the
            target population. However, it *is* possible to for the new
            population to be larger than ``target_population_size``.

        Returns
        -------
        list of Chromosome :
            The chromosomes not selected for removal

        Raises
        ------
        ValueError
            If the ``target_population_size`` is larger than the intial
            `population`
        """
        if target_population_size > len(population):
            raise ValueError("Target population size should\
                              be less than initial population")

        num_removed = 0
        start_pop_size = len(population)
        self._population_index_array = np.random.permutation(len(population))

        self._selection_attempts = 0
        while (start_pop_size - num_removed) > target_population_size and \
              self._selection_attempts < \
              start_pop_size * self.WORST_CASE_FACTOR:

            self._get_unique_random_individuals(population,
                                                self._selection_size,
                                                num_removed)
            removed_indv_indexs = self._get_individuals_for_removal(
                population, target_population_size, num_removed)
            num_removed = self._remove_indviduals(removed_indv_indexs,
                                                  num_removed)
            self._selection_attempts += 1

        return self._update_population(population, num_removed)

    def select_pareto_front(self, population):
        """Selects the pareto front for the `population`

        Parameters
        ----------
            population: list of Chromosomes
                The population to which the pareto front individuals will be 
                selected from.

        Returns
        -------
            list of Chromosomes:
                The Chromosomes in the pareto front.
        """
        num_removed = 0
        self._population_index_array = np.random.permutation(len(population))

        self._get_unique_random_individuals(population,
                                            len(population),
                                            num_removed)
        removed_indv_indexs = self._get_individuals_for_removal(
            population, 1, num_removed)
        num_removed = self._remove_indviduals(removed_indv_indexs, num_removed)

        return self._update_population(population, num_removed)

    def _get_unique_random_individuals(self,
                                       population,
                                       selection_size,
                                       num_removed):
        selection_size = min(selection_size, len(population) - num_removed)
        for i in range(num_removed, num_removed + selection_size):
            random_index = np.random.randint(i, len(population))
            if i != random_index:
                self._swap(self._population_index_array, i, random_index)
        self._selected_indices = range(num_removed,
                                       num_removed + selection_size)

    # TODO look into optimizing. Possibly greedy approach
    def _get_individuals_for_removal(self, population,
                                     target_population_size, num_removed):
        to_be_removed = set()
        num_remaining = len(population) - num_removed
        for i, indv_index_1 in enumerate(self._selected_indices[:-1]):
            for indv_index_2 in self._selected_indices[i+1:]:
                self._update_removal_set(population, indv_index_1,
                                         indv_index_2, to_be_removed)
                if num_remaining - len(to_be_removed) == target_population_size:
                    return to_be_removed
        return to_be_removed

    def _update_removal_set(self, population, indv_index_1,
                            indv_index_2, removal_set):
        indv_1 = self._get_indvidual(population, indv_index_1)
        indv_2 = self._get_indvidual(population, indv_index_2)

        if self._first_dominates(indv_1, indv_2):
            removal_set.add(indv_index_2)
        elif self._first_dominates(indv_2, indv_1):
            removal_set.add(indv_index_1)

    def _get_indvidual(self, population, index):
        population_list_index = self._population_index_array[index]
        return population[population_list_index]

    @staticmethod
    def _first_dominates(indv_a, indv_b):
        return indv_a.genetic_age <= indv_b.genetic_age and \
                indv_a.fitness <= indv_b.fitness

    def _remove_indviduals(self, to_remove_list, num_removed):
        while to_remove_list:
            selection_index = to_remove_list.pop()
            if num_removed in to_remove_list:
                to_remove_list.remove(num_removed)
                to_remove_list.add(selection_index)
            self._swap(self._population_index_array,
                       num_removed, selection_index)
            num_removed += 1
        return num_removed

    @staticmethod
    def _swap(array, index_1, index_2):
        array[index_1], array[index_2] = array[index_2], array[index_1]

    def _update_population(self, population, num_removed):
        new_population = [self._get_indvidual(population, kept_index)
                          for kept_index
                          in range(num_removed, len(population))]
        return new_population
