"""The parallel implemenation of the Archipelago

This module defines the Archipelago data structure that runs in parallel on
multiple processors.
"""

from copy import deepcopy
import numpy as np
from mpi4py import MPI

from .Archipelago import Archipelago

AGE_UPDATE = 2
EXIT_NOTIFICATION = 3
MIGRATION = 4


# TODO update all documentation here
# TODO add inherrited attributes in doc
class ParallelArchipelago(Archipelago):
    """An archipelago that executes island generations serially.

    Parameters
    ----------
    island : Island
        The island from which other islands will be copied
    non_blocking : boolean, default = True
        Specifies whether to use blocking or non-blocking execution.
        Default is non-blocking.
    sync_frequency : int, default = 10
        How frequently to update the average age for each island
    """
    def __init__(self, island, hall_of_fame=None, non_blocking=True,
                 sync_frequency=10):
        self.comm = MPI.COMM_WORLD
        self.comm_rank = self.comm.Get_rank()
        self.comm_size = self.comm.Get_size()
        super().__init__(island, self.comm_size, hall_of_fame)
        self._non_blocking = non_blocking
        self._sync_frequency = sync_frequency
        if self._island.hall_of_fame is None:
            self._island.hall_of_fame = deepcopy(self.hall_of_fame)

    def get_best_fitness(self):
        """Gets the fitness of most fit member

        Returns
        -------
         :
            Fitness of best individual in the archipelago
        """
        best_on_proc = self._island.get_best_fitness()
        best_fitness = self.comm.allreduce(best_on_proc, op=MPI.MIN)
        return best_fitness

    def get_best_individual(self):
        """Returns the best individual if the islands converged to an
        acceptable fitness.

        Returns
        -------
        Chromosome :
            The best individual whose fitness was within the error
            tolerance.
        """
        best_on_proc = self._island.get_best_individual()
        all_best_indvs = self.comm.allgather(best_on_proc)
        best_indv = min(all_best_indvs, key=lambda x: x.fitness)
        return best_indv

    def _step_through_generations(self, num_steps):
        """ Executes 'num_steps' number of generations for
        each island in the archipelago's list of islands

        Parameters
        ----------
        num_steps : int
            The number of generations to execute per island
        """
        if self._non_blocking:
            self._non_blocking_execution(num_steps)
        else:
            self._island.evolve(num_steps)

    def _non_blocking_execution(self, num_steps):
        if self.comm_rank == 0:
            self._non_blocking_execution_master(num_steps)
        else:
            self._non_blocking_execution_slave()

    def _non_blocking_execution_master(self, num_steps):
        total_age = {}
        average_age = self.generational_age
        target_age = average_age + num_steps

        while average_age < target_age:
            self._island.evolve(self._sync_frequency)
            self._gather_updated_ages(total_age)
            average_age = (sum(total_age.values())) / self.comm.size

        self._send_exit_notifications()
        self.comm.Barrier()
        self._gather_updated_ages(total_age)

    def _gather_updated_ages(self, total_age):
        total_age.update({0: self._island.generational_age})
        status = MPI.Status()
        while self.comm.iprobe(source=MPI.ANY_SOURCE,
                               tag=AGE_UPDATE,
                               status=status):
            data = self.comm.recv(source=status.Get_source(),
                                  tag=AGE_UPDATE)
            total_age.update(data)

    def _send_exit_notifications(self):
        for destination in range(1, self.comm_size):
            req = self.comm.isend(True, dest=destination,
                                  tag=EXIT_NOTIFICATION)
            req.Wait()

    def _non_blocking_execution_slave(self):
        while not self._has_exit_notification():
            self._island.evolve(self._sync_frequency)
            self._send_updated_age()
        self.comm.Barrier()

    def _has_exit_notification(self):
        if self.comm.iprobe(source=0, tag=EXIT_NOTIFICATION):
            _ = self.comm.recv(source=0, tag=EXIT_NOTIFICATION)
            return True
        return False

    def _send_updated_age(self):
        data = {self.comm_rank: self._island.generational_age}
        req = self.comm.isend(data, dest=0, tag=AGE_UPDATE)
        req.Wait()

    def _coordinate_migration_between_islands(self):
        """Shuffles island populations for migration and performs
        migration by swapping pairs of individuals between islands
        """
        partner = self._get_migration_partner()
        if partner is not None:
            self._population_exchange_program(partner)

    def _get_migration_partner(self):
        if self.comm_rank == 0:
            island_partners = self._shuffle_island_indices()
        else:
            island_partners = None
        island_partners = self.comm.bcast(island_partners, root=0)
        island_index = island_partners.index(self.comm_rank)
        if island_index % 2 == 0:
            partner_index = island_index + 1
            if partner_index < self.comm_size:
                partner = island_partners[partner_index]
            else:
                partner = None
        else:
            partner_index = island_index - 1
            partner = island_partners[partner_index]
        return partner

    def _shuffle_island_indices(self):
        indices = list(range(self._num_islands))
        np.random.shuffle(indices)
        return indices

    def _population_exchange_program(self, partner):
        population_to_send = self._island.dump_fraction_of_population(0.5)
        received_population = self.comm.sendrecv(population_to_send,
                                                 dest=partner,
                                                 sendtag=MIGRATION,
                                                 source=partner,
                                                 recvtag=MIGRATION)
        self._island.load_population(received_population, replace=False)

    # TODO manually trigger HOF updates
    def _get_potential_hof_members(self):
        potential_members = [i for i in self._island.hall_of_fame]
        all_potential_members = self.comm.allgather(potential_members)
        all_potential_members = [i for hof in all_potential_members
                                 for i in hof]
        return all_potential_members

    def get_fitness_evaluation_count(self):
        """ Gets the total number of fitness evaluations performed

        Returns
        -------
        int :
            number of fitness evaluations
        """
        my_eval_count = self._island.get_fitness_evaluation_count()
        total_eval_count = self.comm.allreduce(my_eval_count, op=MPI.SUM)
        return total_eval_count
