import random

import numpy as np

from bingo.Base.Selection import Selection
from bingo.Util.ArgumentValidation import argument_validation

class AgeFitness(Selection):
    WORST_CASE_FACTOR = 50

    @argument_validation(selection_size={">=": 2})
    def __init__(self, selection_size=2, remove_equals=True):
        self._remove_equals = remove_equals
        self._selection_size = selection_size
        self._selected_indices = []
        self._selection_attempts = 0

    @argument_validation(target_population_size={">": 0})
    def __call__(self, population, target_population_size):
        if target_population_size > len(population):
            raise ValueError("Target population size should\
                              be less than initial population")

        num_removed = [0]
        start_pop_size = len(population)
        population_index_array = np.arange(len(population))
        np.random.shuffle(population_index_array)

        self._selection_attempts = 0
        while (len(population) - num_removed[0]) > target_population_size and \
              self._selection_attempts < start_pop_size * self.WORST_CASE_FACTOR:

            self._get_unique_random_individuals(population, num_removed)
            print("The Selected indicies", self._selected_indices)
            to_remove_set = self._get_dominated_individuals(population, population_index_array)
            print("Those to remove", to_remove_set)
            print("pia_1:", population_index_array)
            self._remove_dominated_indviduals(population_index_array, to_remove_set, num_removed)
            print("pia_2:", population_index_array)
            print() 
            self._selection_attempts += 1
            self._selected_indices.clear()

        print(self._selection_attempts)
        return self._update_population(population, population_index_array, num_removed)

    def _remove_dominated_indviduals(self, population_index_array, to_remove_list, num_removed):
        while to_remove_list:
            selection_index = to_remove_list.pop()
            if num_removed[0] in to_remove_list:
                to_remove_list.remove(num_removed[0])
                to_remove_list.add(selection_index)
            self._swap(population_index_array, num_removed[0], selection_index)
            num_removed[0] += 1

    def _swap(self, array, index_1, index_2):
        array[index_1], array[index_2] = array[index_2], array[index_1]  

    def _update_population(self, population, population_index_array, num_removed):
        new_population = []
        for kept_index in range(num_removed[0], len(population_index_array)):
            new_population.append(population[population_index_array[kept_index]])
        return new_population

    def _get_unique_random_individuals(self, population, num_removed):
        used_indvs = set()
        self._selected_indices.append(np.random.randint(num_removed[0], len(population)))
        used_indvs.add(self._selected_indices[0])
        start = num_removed[0] + 1
        stop = num_removed[0] + self._selection_size if self._selection_size < len(population) - num_removed[0] else len(population)
        print("stop", stop)
        for list_index in range(start, stop):
            print("list index:", list_index)
            val = np.random.randint(num_removed[0], len(population))
            while val in used_indvs:
                val = np.random.randint(num_removed[0], len(population))
                print(used_indvs)
                # input()
            self._selected_indices.append(val)
            print(self._selected_indices[list_index-num_removed[0]])

            used_indvs.add(self._selected_indices[list_index-num_removed[0]])

    def _get_dominated_individuals(self, population, population_index_array):
        to_be_removed = set()
        for i in range(len(self._selected_indices)-1):
            for j in range(i+1, len(self._selected_indices)):
                indv_index_1 = self._selected_indices[i]
                indv_index_2 = self._selected_indices[j]
                if self._check_for_any_nans(population, population_index_array, indv_index_1):
                    to_be_removed.add(indv_index_1)
                elif self._check_for_any_nans(population, population_index_array, indv_index_2):
                    to_be_removed.add(indv_index_2)
                elif self._one_dominates_all(population, population_index_array, indv_index_1, indv_index_2):
                    to_be_removed.add(indv_index_2)
                elif self._one_dominates_all(population, population_index_array, indv_index_2, indv_index_1):
                    to_be_removed.add(indv_index_1)
                elif (population[population_index_array[indv_index_1]].genetic_age == population[population_index_array[indv_index_2]].genetic_age) and self._remove_equals:
                    if population[population_index_array[indv_index_1]].fitness <= population[population_index_array[indv_index_2]].fitness:
                        to_be_removed.add(indv_index_2)
                    else:
                        to_be_removed.add(indv_index_1)
        return to_be_removed

    def _one_dominates_all(self, population, population_index_array, indv_index_1, indv_index_2):
        indv_a = population[population_index_array[indv_index_1]]
        indv_b = population[population_index_array[indv_index_2]]
        # print("indv_a", indv_a.genetic_age, indv_a.list_of_values, indv_a.fitness)
        # print("indv_b", indv_b.genetic_age, indv_b.list_of_values, indv_b.fitness)
        return indv_a.genetic_age < indv_b.genetic_age and\
                indv_a.fitness <= indv_b.fitness

    def _check_for_any_nans(self, population, population_index_array, indv_index):
        return np.any(np.isnan(population[indv_index].fitness))