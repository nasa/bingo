"""The serial implemenation of the Archipelago

This module defines the Archipelago data structure that runs serially on
one processor.
"""
import copy
import numpy as np

from .Archipelago import Archipelago

# TODO update all documentation here
# TODO add inherrited attributes in doc
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
    def __init__(self, island, num_islands=2, hall_of_fame=None):
        super().__init__(island, num_islands, hall_of_fame)
        self._islands = self._generate_islands(island, num_islands)
        for i in self._islands:
            if i.hall_of_fame is None:
                i.hall_of_fame = copy.deepcopy(self.hall_of_fame)

    def _step_through_generations(self, num_steps):
        for island in self._islands:
            island._do_evolution(num_steps)

    def _coordinate_migration_between_islands(self):
        island_partners = self._shuffle_island_indices()

        for i in range(self._num_islands//2):
            self._shuffle_island_and_swap_pairs(island_partners, i)

    def get_best_fitness(self):
        """Gets the fitness of most fit member

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

    # TODO Below should regenerate populations for better random seeding
    @staticmethod
    def _generate_islands(island, num_islands):
        island_list = [copy.deepcopy(island)
                       for _ in range(num_islands)]
        return island_list

    def _shuffle_island_indices(self):
        indices = list(range(self._num_islands))
        np.random.shuffle(indices)
        return indices

    def _shuffle_island_and_swap_pairs(self, island_indexes, pair_number):
        partner_1 = self._islands[island_indexes[pair_number*2]]
        partner_2 = self._islands[island_indexes[pair_number*2 + 1]]
        self._population_exchange_program(partner_1, partner_2)

    def _population_exchange_program(self, island_1, island_2):
        indvs_to_2 = island_1.dump_fraction_of_population(0.5)
        indvs_to_1 = island_2.dump_fraction_of_population(0.5)
        island_1.load_population(indvs_to_1, replace=False)
        island_2.load_population(indvs_to_2, replace=False)

    def _get_potential_hof_members(self):
        potential_members = [h for i in self._islands for h in i.hall_of_fame]
        return potential_members
