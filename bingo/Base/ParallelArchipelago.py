import time
import copy
import random
import logging

from mpi4py import MPI 
import numpy as np

from .Archipelago import Archipelago

LOGGER = logging.getLogger(__name__)

class ParallelArchipelago(Archipelago):

    def __init__(self, island, num_islands=2):
        super().__init__(island, num_islands)
        self.comm = MPI.COMM_WORLD
        self.comm_rank = self.comm.Get_rank()
        self.comm_size = self.comm.Get_size()
        self._islands = self._generate_islands()
    
    def step_through_generations(self, num_steps, non_block=True,
                                 when_to_update=10):
        t_0 = time.time()
        if non_block:
            # do non-blocking execution
            _non_blocking_execution(num_steps, when_to_update)
        else:
            for _ in range(num_steps):


        t_1 = time.time()
        LOGGER.info("%2d >\tage: %d\ttime: %.1fs\tbest fitness: %s",
                    i,
                    self._get_generational_age(island),
                    t_1 - t_0,
                    self._get_pareto_front_fitness(island))
        self.archipelago_age += 1

    def _non_blocking_execution(num_steps, when_to_update=10):
        total_age = {}
        average_age = self.archipelago_age
        target_age = self.archipelago_age + num_steps
        while average_age < target_age:
            if self.isle.solution_island.age % when_update == 0:
                if self.comm_rank == 0:
                    total_age.update({0: self._island.generational_age})
                    status = MPI.Status()

                    while self.comm.iprobe(source=MPI.ANY_SOURCE,
                                           tag=2,
                                           status=status):
                        data = self.comm.recv(source=status.Get_source(),
                                              tag=2)
                        total_age.update(data)
                    average_age = (sum(total_age.values())) / self.comm.size