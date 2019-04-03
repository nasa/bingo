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

    def step_through_generations(self, num_steps, non_block=True,
                                 when_to_update=10):
        if non_block:
            # do non-blocking execution
            self._non_blocking_execution(num_steps, when_to_update)
        else:
            for _ in range(num_steps):
                self._island.generational_step()

        self.archipelago_age += 1

    def _non_blocking_execution(self, num_steps, when_to_update=10):
        total_age = {}
        average_age = self.archipelago_age
        target_age = self.archipelago_age + num_steps
        while average_age < target_age:
            if self._island.generational_age % when_to_update == 0:
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
                    # send average to all other ranks if time to stop
                    if average_age >= num_steps:
                        send_count = 1
                        while send_count < self.comm_size:
                            self.comm.send(average_age, dest=send_count, tag=0)
                            send_count += 1
                # for every other rank, store rank:age, and send it off to 0
                else:
                    data = {self.comm_rank : self._island.generational_age}
                    req = self.comm.isend(data, dest=0, tag=2)
                    req.Wait()
            # if there is a message from 0 to stop, update averageAge
            if self.comm_rank != 0:
                if self.comm.iprobe(source=0, tag=0):
                    average_age = self.comm.recv(source=0, tag=0)
            self._island.generational_step()
