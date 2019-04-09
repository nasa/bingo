""" The abstract-base class for the Archipelago.

This module defines the data structure that manages a group of `Islands`.
Archipelago implementations will control the generational steps for all
the islands until convergence or until a maximal number of steps is reached.
"""
from abc import ABCMeta, abstractmethod
import random

from ..Util.ArgumentValidation import argument_validation

class Archipelago(metaclass=ABCMeta):
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
        self.archipelago_age = 0
        self._island = island
        self._num_islands = num_islands


    @argument_validation(max_generations={">=": 1},
                         min_generations={">=": 0},
                         generation_step_report={">=": 0},
                         error_tol={">=": 0})
    def run_islands(self, max_generations, min_generations, 
                    generation_step_report, error_tol=10e-6):
        """Executes generational steps on all the islands for at least
        'min_generations' until 'max_generations' or until `error_tol`
        is acheived for any individual `Chromosome`. Convergence is checked
        every `generation_step_report` generations.

        Parameters
        ----------
        max_generations: int
            The maximum number of generations the algorithm will run.
        min_generations: int
            The minimum number of generations the algorithm will run.
        generation_step_report: int
            The number of generations that will run before checking for
            convergence.
        error_tol: float, default 10e-6
            The minimum fitness that must be achieved in order for the
            algorithm to converge.

        Returns:
        --------
        bool :
            Indicates whether convergence has been acheived.
        """
        self.step_through_generations(generation_step_report)
        converged = self.test_for_convergence(error_tol)

        while self.archipelago_age < min_generations or \
                (self.archipelago_age < max_generations and not converged):
            self.coordinate_migration_between_islands()
            self.step_through_generations(generation_step_report)
            converged = self.test_for_convergence(generation_step_report)

        return converged

    @abstractmethod
    def step_through_generations(self, num_steps):
        raise NotImplementedError

    @abstractmethod
    def coordinate_migration_between_islands(self):
        raise NotImplementedError

    @abstractmethod
    def test_for_convergence(self, error_tol):
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
