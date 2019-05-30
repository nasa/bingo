""" The abstract-base class for the Archipelago.

This module defines the data structure that manages a group of `Islands`.
Archipelago implementations will control the generational steps for all
the islands until convergence or until a maximal number of steps is reached.
"""
from abc import ABCMeta, abstractmethod
import random

from .EvolutionaryOptimizer import EvolutionaryOptimizer
from ..Util.ArgumentValidation import argument_validation


class Archipelago(EvolutionaryOptimizer, metaclass=ABCMeta):
    """Collection of islands

    Parameters
    ----------
    island : Island
        Island that contains the generator for the individuals used in the
        EA
    num_islands : int
        The size of the archipelago; the number of islands it contains
    """
    def __init__(self, island, num_islands):
        self.sim_time = 0
        self.start_time = 0
        self._island = island
        self._num_islands = num_islands
        super().__init__()

    def _do_evolution(self, num_generations):
        self._coordinate_migration_between_islands()
        self._step_through_generations(num_generations)
        self.generational_age += num_generations

    @abstractmethod
    def _step_through_generations(self, num_steps):
        raise NotImplementedError

    @abstractmethod
    def _coordinate_migration_between_islands(self):
        raise NotImplementedError

    @staticmethod
    def assign_send_receive(island_1, island_2):
        """
        Assign indices to be exchanged between the islands by random shuffling.

        Paramters
        ---------
        island_1: Island
            Island containing a population of Chromosomes
        island_2:
            Island containing a population of Chromsomes

        Returns
        -------
        tuple of list of ints:
            The indices for individuals in island_1 and island_2, respectively,
            to be sent to the other island. 
        """
        pop_size1 = len(island_1.population)
        pop_size2 = len(island_2.population)
        tot_pop = pop_size1 + pop_size2
        pop_shuffle = list(range(tot_pop))
        random.shuffle(pop_shuffle)
        indvs_to_send = set() 
        indvs_to_receive = set() 
        for i, indv in enumerate(pop_shuffle):
            my_new = (i < tot_pop/2)
            my_old = (indv < pop_size1)
            if my_new and not my_old:
                indvs_to_receive.add(indv-pop_size1)
            if not my_new and my_old:
                indvs_to_send.add(indv)
        return indvs_to_send, indvs_to_receive
