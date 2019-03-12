import numpy as np

from bingo.Base.Selection import Selection
from bingo.Util.ArgumentValidation import argument_validation

class AgeFitness(Selection):
    WORST_CASE_FACTOR = 50

    @argument_validation(selection_size={">=": 2})
    def __init__(self, selection_size=2):
        self._selection_size = selection_size
        self._selected_indices = []
        self._population_index_array = np.array([])
        self._selection_attempts = 0

    @argument_validation(target_population_size={">": 0})
    def __call__(self, population, target_population_size):
        if target_population_size > len(population):
            raise ValueError("Target population size should\
                              be less than initial population")

        num_removed = [0]
        start_pop_size = len(population)
        self._population_index_array = np.random.permutation(len(population))

        self._selection_attempts = 0
        while (len(population) - num_removed[0]) > target_population_size and \
              self._selection_attempts < start_pop_size * self.WORST_CASE_FACTOR:

            self._get_unique_random_individuals(population, num_removed)
            removed_indv_indexs = self._get_individuals_for_removal(
                population, target_population_size, num_removed)
            self._remove_indviduals(removed_indv_indexs, num_removed)

            self._selection_attempts += 1

        return self._update_population(population, num_removed)

    def _get_unique_random_individuals(self, population, num_removed):
        index_range = range(num_removed[0], len(population))
        selection_size = self._selection_size \
            if self._selection_size <= len(index_range) \
            else len(index_range)
        self._selected_indices = np.random.choice(index_range,
                                                  selection_size,
                                                  replace=False)

    # TODO look into optimizing. Possibly greedy approach
    def _get_individuals_for_removal(self, population,
                                     target_population_size, num_removed):
        to_be_removed = set()
        num_remaining = len(population) - num_removed[0]
        for i in range(len(self._selected_indices)-1):
            for j in range(i+1, len(self._selected_indices)):
                indv_index_1 = self._selected_indices[i]
                indv_index_2 = self._selected_indices[j]
                self._update_removal_set(population, indv_index_1,
                                         indv_index_2, to_be_removed)
                if num_remaining - len(to_be_removed) == target_population_size:
                    return to_be_removed
        return to_be_removed

    def _update_removal_set(self, population, indv_index_1,
                            indv_index_2, removal_set):
        indv_1 = self._get_indvidual(population, indv_index_1)
        indv_2 = self._get_indvidual(population, indv_index_2)

        if self._indv_has_nans(indv_1):
            removal_set.add(indv_index_1)
        elif self._indv_has_nans(indv_2):
            removal_set.add(indv_index_2)
        elif self._one_dominates_all(indv_1, indv_2):
            removal_set.add(indv_index_2)
        elif self._one_dominates_all(indv_2, indv_1):
            removal_set.add(indv_index_1)
        elif (indv_1.genetic_age == indv_2.genetic_age):
            if indv_1.fitness <= indv_2.fitness:
                removal_set.add(indv_index_2)
            else:
                removal_set.add(indv_index_1)

    def _get_indvidual(self, population, index):
        population_list_index = self._population_index_array[index]
        return population[population_list_index]

    def _one_dominates_all(self, indv_a, indv_b):
        return indv_a.genetic_age < indv_b.genetic_age and\
                indv_a.fitness <= indv_b.fitness

    def _indv_has_nans(self, indiviudal):
        return np.any(np.isnan(indiviudal.fitness))

    def _remove_indviduals(self, to_remove_list, num_removed):
        while to_remove_list:
            selection_index = to_remove_list.pop()
            if num_removed[0] in to_remove_list:
                to_remove_list.remove(num_removed[0])
                to_remove_list.add(selection_index)
            self._swap(self._population_index_array,
                       num_removed[0], selection_index)
            num_removed[0] += 1

    def _swap(self, array, index_1, index_2):
        array[index_1], array[index_2] = array[index_2], array[index_1]

    def _update_population(self, population, num_removed):
        new_population = []
        for kept_index in range(num_removed[0], len(population)):
            indv = self._get_indvidual(population, kept_index)
            new_population.append(indv)
        return new_population
