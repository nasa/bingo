import time
import copy
import random
import logging

LOGGER = logging.getLogger(__name__)

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
        super().__init__(island, num_islands)
        self._islands = self._generate_islands()

    def step_through_generations(self, num_steps):
        """ Executes 'num_steps' number of generations for
        each island in the archipelago's list of islands

        Parameters
        ----------
        num_steps : int
                    The number of generations to execute per island
        """
        t_0 = time.time()
        for i, island in enumerate(self._islands):
            t_1 = time.time()
            for _ in range(num_steps):
                island.execute_generational_step()
            t_2 = time.time()
            LOGGER.info("%2d >\tage: %d\ttime: %.1fs\tbest fitness: %s",
                        i,
                        self._get_generational_age(island),
                        t_2 - t_1,
                        self._get_pareto_front_fitness(island))
        t_3 = time.time()
        LOGGER.info("total time: %.1fs", (t_3 - t_0))
        self.archipelago_age += 1

    def coordinate_migration_between_islands(self):
        """Shuffles island populations for migration and performs
        migration by swapping pairs of individuals between islands
        """
        island_partners = self._shuffle_island_indices()

        for i in range(self._num_islands//2):
            self._shuffle_island_and_swap_pairs(island_partners, i)

    def test_for_convergence(self, error_tol):
        """Tests that the fitness of individuals is less than
        or equal to the specified error tolerance

        Parameters
        ----------
        error_tol : int
                    Upper bound for acceptable fitness of an individual
        """
        list_of_best_indvs = []
        for island in self._islands:
            best_indv = island.best_individual()
            list_of_best_indvs.append(best_indv)
        list_of_best_indvs.sort(key=lambda x: x.fitness)

        best_indv = list_of_best_indvs[0]
        converged = best_indv.fitness <= error_tol

        return converged

    def _generate_islands(self):
        island_list = []
        for _ in range(self._num_islands):
            island_list.append(copy.deepcopy(self._island))
        return island_list

    def _get_generational_age(self, island):
        return island.generational_age

    def _get_pareto_front_fitness(self, island):
        return island.best_individual().fitness

    def _shuffle_island_indices(self):
        indices = list(range(self._num_islands))
        random.shuffle(indices)
        return indices

    def _shuffle_island_and_swap_pairs(self, island_indexes, pair_number):
        partner_1 = self._islands[island_indexes[pair_number*2]]
        partner_2 = self._islands[island_indexes[pair_number*2 + 1]]
        self._swap_island_individuals(partner_1, partner_2)

    def _swap_island_individuals(self, island_1, island_2):
        indexes_to_2, indexes_to_1 = self._get_send_and_receive_pairs(
            island_1, island_2)
        self._exchange_individuals(island_1, island_2, 
                                   indexes_to_1, indexes_to_2)

    def _get_send_and_receive_pairs(self, island_1, island_2):
        return Archipelago.assign_send_receive(island_1, island_2)

    def _exchange_individuals(self, island_1, island_2,
                                list_of_indexes_to_1, list_of_indexes_to_2):
        indexes_to_1 = set(list_of_indexes_to_1)
        indexes_to_2 = set(list_of_indexes_to_2)
        indvs_to_2 = [island_1.population[indv] for indv in indexes_to_2]
        indvs_to_1 = [island_2.population[indv] for indv in indexes_to_1]

        new_pop_island_1 = [indv for i, indv in enumerate(island_1.population) if i not in indexes_to_2] + indvs_to_1
        new_pop_island_2 = [indv for i, indv in enumerate(island_2.population) if i not in indexes_to_1] + indvs_to_2

        island_1.load_population(new_pop_island_1)
        island_2.load_population(new_pop_island_2)