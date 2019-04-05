"""The serial implemenation of the Archipelago

This module dfines the Archipelago data structure that runs serially on
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
        super().__init__(island, num_islands)
        self._islands = self._generate_islands()
        self._converged = False
        self._best_indv = None

    def step_through_generations(self, num_steps):
        """ Executes 'num_steps' number of generations for
        each island in the archipelago's list of islands

        Parameters
        ----------
        num_steps : int
            The number of generations to execute per island
        """
        for island in self._islands:
            for _ in range(num_steps):
                island.execute_generational_step()
        self.archipelago_age += num_steps

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

        Returns
        -------
        bool :
            Indicates whether a chromosome has converged.
        """
        list_of_best_indvs = []
        for island in self._islands:
            best_indv = island.best_individual()
            list_of_best_indvs.append(best_indv)
        list_of_best_indvs.sort(key=lambda x: x.fitness)

        best_indv = list_of_best_indvs[0]
        converged = best_indv.fitness <= error_tol

        self._best_indv = best_indv
        self._converged = converged
        return converged

    def get_best_individual(self):
        """Returns the best individual if the islands converged to an
        acceptable fitness.

        Returns
        -------
        Chromosome :
            The best individual whose fitness was within the error
            tolerance.
        """
        return self._best_indv

    def _generate_islands(self):
        island_list = []
        for _ in range(self._num_islands):
            island_list.append(copy.deepcopy(self._island))
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
