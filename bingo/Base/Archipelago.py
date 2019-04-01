from abc import ABCMeta, abstractmethod
import random

class Archipelago(metaclass=ABCMeta):

    def __init__(self, island, num_islands):
        self.sim_time = 0
        self.start_time = 0
        self.archipelago_age = 0
        self._island = island
        self._num_islands = num_islands

    def run_islands(self, max_generations, min_generations, 
                    generation_step_report, error_tol=10e-6):
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
        Assign indices for exchange through random shuffling

        :param pop_size: number of individuals in the populations which will be
                         exchanged (must be equal in population size)
        :return: the indices that each island will be swapping
        """
        pop_size1 = len(island_1.population)
        pop_size2 = len(island_2.population)
        tot_pop = pop_size1 + pop_size2
        pop_shuffle = list(range(tot_pop))
        random.shuffle(pop_shuffle)
        indvs_to_send = []
        indvs_to_receive = []
        for i, indv in enumerate(pop_shuffle):
            my_new = (i < tot_pop/2)
            my_old = (indv < pop_size1)
            if my_new and not my_old:
                indvs_to_receive.append(indv-pop_size1)
            if not my_new and my_old:
                indvs_to_send.append(indv)
        return indvs_to_send, indvs_to_receive
