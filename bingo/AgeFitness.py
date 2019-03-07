import random

import numpy as np

from bingo.Base.Selection import Selection
from bingo.Util.ArgumentValidation import argument_validation

class AgeFitness(Selection):
    WORST_CASE_FACTOR = 50

    @argument_validation(selection_size={">=": 2})
    def __init__(self, selection_size=2, remove_equals=True):
        self._selection_size = selection_size
        self._remove_equals = remove_equals
        self._selected_indices = [None]*self._selection_size

    def __call__(self, population, target_population_size):
        if target_population_size > len(population):
            raise ValueError("Target population size should\
                              be less than initial population")

        random.shuffle(population)
        start_pop_size = len(population)
        selection_attemps = 0
        while len(population) > target_population_size and \
              selection_attemps < start_pop_size * self.WORST_CASE_FACTOR:

            indv_index_1, indv_index_2 = self._get_unique_random_individuals(population)
            self._remove_dominated_individuals(population, 
                                               indv_index_1,
                                               indv_index_2)
            selection_attemps += 1

        return population
    


    def _get_unique_random_individuals(self, population):
        indv_index_1 = np.random.randint(len(population))
        indv_index_2 = np.random.randint(len(population))
        while indv_index_1 == indv_index_2:
            indv_index_2 = np.random.randint(len(population))
        return indv_index_1, indv_index_2

    def _remove_dominated_individuals(self, population, 
                                      indv_index_1, indv_index_2):
        if self._check_for_any_nans(population, indv_index_1):
            del population[indv_index_1]
        elif self._check_for_any_nans(population, indv_index_2):
            del population[indv_index_2]
        elif self._one_dominates_all(population, indv_index_1, indv_index_2):
            del population[indv_index_2]
        elif self._one_dominates_all(population, indv_index_2, indv_index_1):
            del population[indv_index_1]
        elif self._remove_equals:
            if population[indv_index_1].fitness <= population[indv_index_2].fitness:
                del population[indv_index_2]
            else:
                del population[indv_index_1]

    def _one_dominates_all(self, population, indv_index_1, indv_index_2):
        indv_a = population[indv_index_1]
        indv_b = population[indv_index_2]
        return indv_a.genetic_age < indv_b.genetic_age and\
                indv_a.fitness <= indv_b.fitness

    def _check_for_any_nans(self, population, indv_index):
        return np.any(np.isnan(population[indv_index].fitness))