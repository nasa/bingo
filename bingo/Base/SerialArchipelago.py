"""The serial implemenation of the Archipelago

This module defines the Archipelago data structure that runs serially on
one processor.
"""
import copy
import random

from .Archipelago import Archipelago


class SerialArchipelago(Archipelago):
    """An archipelago that executes island generations serially.

    Parameters
    ----------
    island : Island
        The island from which other islands will be copied
    num_islands : int, default = 2
        The number of islands to create in the archipelago's
        list of islands
    """
    def __init__(self, island, num_islands=2):
        self._islands = self._generate_islands(island, num_islands)
        super().__init__(island, num_islands)

    def _step_through_generations(self, num_steps):
        for island in self._islands:
            island._do_evolution(num_steps)

    def _coordinate_migration_between_islands(self):
        island_partners = self._shuffle_island_indices()

        for i in range(self._num_islands//2):
            self._shuffle_island_and_swap_pairs(island_partners, i)

    def get_best_fitness(self):
        """Gets the fitness of most fit island member

        Returns
        -------
         :
            Fitness of best individual in the archipelago
        """
        return self.get_best_individual().fitness

    def get_best_individual(self):
        """Returns the best individual if the islands converged to an
        acceptable fitness.

        Returns
        -------
        Chromosome :
            The best individual whose fitness was within the error
            tolerance.
        """
        list_of_best_indvs = [i.get_best_individual() for i in self._islands]
        list_of_best_indvs.sort(key=lambda x: x.fitness)
        return list_of_best_indvs[0]

    def get_fitness_evaluation_count(self):
        """ Gets the number of fitness evaluations performed

        Returns
        -------
        int :
            number of fitness evaluations
        """
        return sum([island.get_fitness_evaluation_count()
                    for island in self._islands])

    @staticmethod
    def _generate_islands(island, num_islands):
        island_list = [copy.deepcopy(island)
                       for _ in range(num_islands)]
        return island_list

    def _shuffle_island_indices(self):
        indices = list(range(self._num_islands))
        random.shuffle(indices)
        return indices

    def _shuffle_island_and_swap_pairs(self, island_indexes, pair_number):
        partner_1 = self._islands[island_indexes[pair_number*2]]
        partner_2 = self._islands[island_indexes[pair_number*2 + 1]]
        self._swap_island_individuals(partner_1, partner_2)

    def _swap_island_individuals(self, island_1, island_2):
        indexes_to_2, indexes_to_1 = Archipelago.assign_send_receive(
            island_1, island_2)

        indvs_to_2 = [island_1.population[indv] for indv in indexes_to_2]
        indvs_to_1 = [island_2.population[indv] for indv in indexes_to_1]

        new_pop_island_1 = [indv for i, indv in enumerate(island_1.population)
                            if i not in indexes_to_2] + indvs_to_1
        new_pop_island_2 = [indv for i, indv in enumerate(island_2.population)
                            if i not in indexes_to_1] + indvs_to_2

        island_1.load_population(new_pop_island_1)
        island_2.load_population(new_pop_island_2)

    def _get_potential_hof_members(self):
        potential_members = []
        for i in self._islands:
            potential_members += i.population
        return potential_members
