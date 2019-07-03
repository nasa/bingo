""" The abstract-base class for the Archipelago.

This module defines the data structure that manages a group of `Islands`.
Archipelago implementations will control the generational steps for all
the islands until convergence or until a maximal number of steps is reached.
"""
from abc import ABCMeta, abstractmethod

from .EvolutionaryOptimizer import EvolutionaryOptimizer


# TODO update all documentation here
# TODO add inherrited attributes in doc
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
    def __init__(self, island, num_islands, hall_of_fame=None):
        super().__init__(hall_of_fame)
        self.sim_time = 0
        self.start_time = 0
        self._island = island
        self._num_islands = num_islands

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
