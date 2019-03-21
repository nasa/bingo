from abc import ABCMeta, abstractmethod

class Archipelago(metaclass=ABCMeta):

    def __init__(self, island, num_islands):
        self.sim_time = 0
        self.start_time = 0
        self._island = island
        self._num_islands = num_islands

    @abstractmethod
    def run_islands(self, max_generations, error_tol,
                    min_generations, generation_step_report):
        raise NotImplementedError
    
    @abstractmethod
    def step_through_generations(self):
        raise NotImplementedError
    
    @abstractmethod
    def coordinate_migration_between_islands(self):
        raise NotImplementedError
    
    @abstractmethod
    def test_for_convergence(self):
        """Test for convergence of island system

        Parameters
        ----------
        """
