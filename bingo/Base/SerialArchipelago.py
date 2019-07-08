"""The serial implemenation of the Archipelago

This module defines the Archipelago data structure that runs serially on
one processor.
"""
import copy
import numpy as np
import logging

from .Archipelago import Archipelago
from ..Util.Log import DETAILED_INFO

LOGGER = logging.getLogger(__name__)


class SerialArchipelago(Archipelago):
    """An collection of islands that evolve serially.

    Evolution of the Archipelago involves independent evolution of Islands
    combined with periodic migration of individuals between random pairs of
    islands. The evolution occurs on one Island at a time.

    Parameters
    ----------
    island : Island
        The island that acts as a template for all islands in the archipelago
    num_islands : int, default = 2
        The number of islands to create in the archipelago's
        list of islands

    Attributes
    ----------
    generational_age: int
        The number of generations the archipelago has been evolved
    hall_of_fame: HallOfFame
        An object containing the best individuals seen in the archipelago
    """
    def __init__(self, island, num_islands=2, hall_of_fame=None):
        super().__init__(island, num_islands, hall_of_fame)
        self._islands = self._generate_islands(island, num_islands)
        for i in self._islands:
            if i.hall_of_fame is None:
                i.hall_of_fame = copy.deepcopy(self.hall_of_fame)

    def _step_through_generations(self, num_steps):
        for island in self._islands:
            island.evolve(num_steps, hall_of_fame_update=False)

    def _coordinate_migration_between_islands(self):
        LOGGER.log(DETAILED_INFO, "Performing migration between Islands")
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
        """Returns the best individual

        Returns
        -------
        Chromosome :
            The individual with lowest fitness
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
        for island in island_list:
            island.regenerate_population()
        return island_list

    def _shuffle_island_indices(self):
        indices = list(range(self._num_islands))
        np.random.shuffle(indices)
        return indices

    def _shuffle_island_and_swap_pairs(self, island_indexes, pair_number):
        partner_1_index = island_indexes[pair_number * 2]
        partner_2_index = island_indexes[pair_number * 2 + 1]
        LOGGER.debug("    %d <-> %d", partner_1_index, partner_2_index)
        partner_1 = self._islands[partner_1_index]
        partner_2 = self._islands[partner_2_index]
        self._population_exchange_program(partner_1, partner_2)

    @staticmethod
    def _population_exchange_program(island_1, island_2):
        indvs_to_2 = island_1.dump_fraction_of_population(0.5)
        indvs_to_1 = island_2.dump_fraction_of_population(0.5)
        island_1.load_population(indvs_to_1, replace=False)
        island_2.load_population(indvs_to_2, replace=False)

    def _log_evolution(self, start_time):
        pass

    def _get_potential_hof_members(self):
        for island in self._islands:
            island.update_hall_of_fame()
        potential_members = [h for i in self._islands for h in i.hall_of_fame]
        return potential_members
