import time
import copy
import logging

LOGGER = logging.getLogger(__name__)

from .Base.Archipelago import Archipelago

class SerialArchipelago(Archipelago):

    def __init__(self, island, num_islands=2):
        super().__init__(island, num_islands)
        self._islands = self._generate_islands()
    
    def run_islands(self, max_generations, min_generations,
                    error_tol, generation_step_report):
        pass
    
    def step_through_generations(self, num_steps):
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

    def coordinate_migration_between_islands(self):
        pass

    def test_for_convergence(self, max_generations, min_generations, error_tol):
        pass
        # list_of_individuals = self._get_list_of_pareto_indivdiuals()
        # self._update_pareto_individuals()
        #test convergance
        #log output
    
    def _generate_islands(self):
        island_list = []
        for island in range(self._num_islands):
            island_list.append(copy.deepcopy(self._island))
        return island_list

    def _get_generational_age(self, island):
        return island.generational_age

    def _get_pareto_front_fitness(self, island):
        return island.best_individual().fitness