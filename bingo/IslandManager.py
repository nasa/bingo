"""
Island managers manage a group of coevolution islands.  Specifically, they step
through generations, coordinate migration, and test convergence.  Currently
there are two implementations: ParallelIslandManger which runs islands on
individual processors using mpi, and SerialIslandManager which runs islands one
after one on a single processor
"""

import time
import random
import abc
import copy
import pickle
import logging

from mpi4py import MPI
import numpy as np

from .CoevolutionIsland import CoevolutionIsland as ci
from .Island import Island
from .Plotting import print_latex, print_pareto, print_1d_best_soln

LOGGER = logging.getLogger(__name__)

class IslandManager(object):
    """
    IslandManager is an abstract class used for making controllers of groups of
    coevolution islands
    """

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        """
        Initialization of island manager
        """
        self.sim_time = 0
        self.start_time = 0
        self.age = 0

    def run_islands(self, max_steps, epsilon, min_steps=0,
                    step_increment=1000, make_plots=True,
                    checkpoint_file=None, **do_steps_kwargs):
        """
        Runs co-evolution islands until convergence of best solution

        :param max_steps: maximum number of steps to take
        :param epsilon: error which defines convergence
        :param min_steps: minimum number of steps to take
        :param step_increment: the number of steps between
                               migrations/convergence checks
        :param make_plots: boolean for whether to produce plots
        :param checkpoint_file: base file name for checkpoint files
        :param do_steps_kwargs: extra keyword arguments are passed through
                                to do_steps()
        :return converged: whether a converged solution has been found
        """
        self.start_time = time.time()
        self.do_steps(n_steps=step_increment, **do_steps_kwargs)
        if checkpoint_file is not None:
            self.save_state(checkpoint_file + "_%d.p" % self.age)
        converged = self.test_convergence(epsilon, make_plots)
        while self.age < min_steps or (self.age < max_steps and not converged):
            self.do_migration()
            self.do_steps(n_steps=step_increment, **do_steps_kwargs)
            if checkpoint_file is not None:
                self.save_state(checkpoint_file + "_%d.p" % self.age)
            converged = self.test_convergence(epsilon, make_plots)

        self.do_final_plots(make_plots)

        return converged

    @abc.abstractmethod
    def do_steps(self, n_steps, **kwargs):
        """
        Steps through generations.

        :param n_steps: number of generations through which to step
        :param kwargs: extra keyword args that can be passed (implementation
                       specific)
        """
        pass

    @abc.abstractmethod
    def do_migration(self):
        """
        Coordinates migration between islands
        """
        pass

    @abc.abstractmethod
    def test_convergence(self, epsilon, make_plots):
        """
        Tests for convergence of the island system

        :param epsilon: error which defines convergence
        :param make_plots: boolean for whether to produce plots
        :return: boolean for whether convergence has been reached
        """
        pass

    @abc.abstractmethod
    def do_final_plots(self, make_plots):
        """
        Makes some summary output

        :param make_plots: boolean for whether to produce plots
        """
        pass

    @abc.abstractmethod
    def save_state(self, filename):
        """
        Saves all information in the sim

        :param filename: full name of file to save pickle
        """
        pass

    @abc.abstractmethod
    def load_state(self, filename):
        """
        loads island manager information from file

        :param filename: full name of file to load pickle
        """
        pass

    @staticmethod
    def assign_send_receive(pop_size):
        """
        Assign indices for exchange through random shuffling

        :param pop_size: number of individuals in the populations which will be
                         exchanged (must be equal in population size)
        :return: the indices that each island will be swapping
        """
        pop_shuffle = list(range(pop_size*2))
        random.shuffle(pop_shuffle)
        indvs_to_send = []
        indvs_to_receive = []
        for i, indv in enumerate(pop_shuffle):
            my_new = (i < pop_size)
            my_old = (indv < pop_size)
            if my_new and not my_old:
                indvs_to_receive.append(indv-pop_size)
            if not my_new and my_old:
                indvs_to_send.append(indv)
        assert len(indvs_to_send) == len(indvs_to_receive)
        return indvs_to_send, indvs_to_receive


class ParallelIslandManager(IslandManager):
    """
    ParallelIslandManager is an implementation of the IslandManager class which
    uses mpi4py and message passing to coordinate the distribution of
    coevolution islands in parallel

    developer notes:
    MPI_Tag     Function
    --------------------
        0       average age sent from rank 0 (i.e., avg step limit exceeded)
        2       age uptate sent to rank 0
        4       migration communications
        6       saving state
        7       loading state
    """

    def __init__(self, restart_file=None, *args, **kwargs):
        """
        Initialization of island manager.  The number of islands is set by the
        number of processors in the mpi call.

        :param restart_file: file name from which to load the island manager
        :param args: arguments to be passed to initialization of coevolution
                     islands
        :param kwargs: keyword arguments to be passed to initialization of
                       coevolution islands
        """
        super(ParallelIslandManager, self).__init__(*args, **kwargs)

        self.comm = MPI.COMM_WORLD
        self.comm_rank = self.comm.Get_rank()
        self.comm_size = self.comm.Get_size()

        if restart_file is None:
            # make coevolution islands
            self.isle = ci(*args, **kwargs)

            # make dummy island for joint pareto front calculations
            if self.comm_rank == 0:
                self.pareto_isle = Island(
                    self.isle.solution_island.gene_manipulator,
                    self.isle.true_fitness_plus_complexity,
                    0, 0, 0)
            else:
                self.pareto_isle = None
        else:
            self.load_state(restart_file)

    def do_steps(self, n_steps, non_block=True, when_update=10):
        """
        Steps through generations

        :param n_steps: number of generations through which to step
        :param non_block: boolean to determine blocking or non
        :param when_update: how often each rank updates in non blocking
        """
        t_0 = time.time()
        if non_block:
            # totalAge to hold rank:age, averageAge and target_age to
            # hold when to stop, when_update for when to send/receive data
            total_age = {}
            average_age = self.age
            target_age = self.age + n_steps
            while average_age < target_age:
                if self.isle.solution_island.age % when_update == 0:
                    if self.comm_rank == 0:
                        # update the age in totalAge for self
                        total_age.update({0:self.isle.solution_island.age})
                        # while there is data from any rank, receive until
                        # last, and add the data to totalAge
                        # TODO (gbomarito) could get flooded and never exit loop
                        status = MPI.Status()
                        while self.comm.iprobe(source=MPI.ANY_SOURCE, tag=2,
                                               status=status):
                            data = self.comm.recv(source=status.Get_source(),
                                                  tag=2)
                            total_age.update(data)
                        average_age = (sum(total_age.values())) / self.comm.size
                        # send average to all other ranks if time to stop
                        if average_age >= n_steps:
                            scount = 1
                            while scount < self.comm_size:
                                self.comm.send(average_age, dest=scount, tag=0)
                                scount += 1
                    # for every other rank, store rank:age, and send it off to 0
                    else:
                        data = {self.comm_rank:self.isle.solution_island.age}
                        req = self.comm.isend(data, dest=0, tag=2)
                        req.Wait()
                # if there is a message from 0 to stop, update averageAge
                if self.comm_rank != 0:
                    if self.comm.iprobe(source=0, tag=0):
                        average_age = self.comm.recv(source=0, tag=0)
                self.isle.deterministic_crowding_step()
                # print_pareto(isle.solution_island.pareto_front, "front.png")
        else:
            for _ in range(n_steps):
                self.isle.deterministic_crowding_step()
        t_1 = time.time()
        LOGGER.info("%2d >\tage: %d\ttime: %.1fs\tbest fitness: %s",
                    self.comm_rank,
                    self.isle.solution_island.age,
                    t_1 - t_0,
                    self.isle.solution_island.pareto_front[0].fitness)

        if non_block:
            # perform message cleanup before moving on
            self.comm.Barrier()
            if self.comm_rank == 0:
                status = MPI.Status()
                while self.comm.iprobe(source=MPI.ANY_SOURCE, tag=2,
                                       status=status):
                    data = self.comm.recv(source=status.Get_source(), tag=2)

        if np.isnan(self.isle.solution_island.pareto_front[0].fitness[0]):
            for i in self.isle.solution_island.pop:
                LOGGER.error(str(i.fitness))
            for indv in self.isle.solution_island.pareto_front:
                LOGGER.error("pareto > %s  %s",
                             str(indv.fitness), indv.latexstring())
        self.age += n_steps

    def do_migration(self):
        """
        Coordinates migration between islands
        """
        # assign partners
        if self.comm_rank == 0:
            partners = list(range(self.comm_size))
            random.shuffle(partners)
        else:
            partners = None
        partners = self.comm.bcast(partners, root=0)
        ind = partners.index(self.comm_rank)

        # primary partner
        primary = (ind % 2 == 0)
        if primary:
            if self.comm_rank != partners[-1]:
                my_partner = partners[ind + 1]

                # find which indvs to send/receive
                s_send, s_receive = \
                    IslandManager.assign_send_receive(
                        self.isle.solution_island.pop_size)
                p_send, p_receive = \
                    IslandManager.assign_send_receive(
                        self.isle.predictor_island.pop_size)
                t_send, t_receive = IslandManager.assign_send_receive(
                    len(self.isle.trainers))
                LOGGER.debug("Migration: %2d <-> %2d  mixing = %s",
                             self.comm_rank,
                             my_partner,
                             str((float(len(s_send)) /
                                  self.isle.solution_island.pop_size,
                                  float(len(p_send)) /
                                  self.isle.predictor_island.pop_size,
                                  float(len(t_send)) /
                                  len(self.isle.trainers))))
                self.comm.send((s_receive, p_receive, t_receive),
                               dest=my_partner, tag=4)

                # exchange populations
                send_package = self.isle.dump_populations(s_send, p_send,
                                                          t_send)
                self.comm.send(send_package, dest=my_partner, tag=4)
                recv_package = self.comm.recv(source=my_partner, tag=4)
                self.isle.load_populations(recv_package, s_send, p_send,
                                           t_send)

        # secondary partner
        else:
            my_partner = partners[ind - 1]

            # find which indvs to send/receive
            s_send, p_send, t_send = self.comm.recv(source=my_partner, tag=4)

            # exchange populations
            send_package = self.isle.dump_populations(s_send, p_send, t_send)
            recv_package = self.comm.recv(source=my_partner, tag=4)
            self.comm.send(send_package, dest=my_partner, tag=4)
            self.isle.load_populations(recv_package, s_send, p_send, t_send)

    def test_convergence(self, epsilon, make_plots):
        """
        Tests for convergence of the island system

        :param epsilon: error which defines convergence
        :param make_plots: boolean for whether to produce plots
        :return: boolean for whether convergence has been reached
        """
        # gather all pareto fronts
        par_list = self.isle.solution_island.dump_pareto()
        par_list = self.comm.gather(par_list, root=0)

        # test combined pareto front for convergence
        if self.comm_rank == 0:
            par_list = par_list[0] + self.pareto_isle.dump_pareto()
            self.pareto_isle.load_population(par_list)
            self.pareto_isle.update_pareto_front()
            converged = (self.pareto_isle.pareto_front[0].fitness[0] < epsilon)

            # output
            LOGGER.info("current age: %d", self.age)
            LOGGER.info("current best true fitness: %s",
                        str(self.pareto_isle.pareto_front[0].fitness[0]))
            LOGGER.info("current best solution: %s",
                        self.pareto_isle.pareto_front[0].latexstring())
            if make_plots:
                print_latex(self.pareto_isle.pareto_front, "eq.png")
                print_pareto(self.pareto_isle.pareto_front, "front.png")
                if self.isle.fitness_metric_args['x'].shape[1] == 1:
                    if 'y' in self.isle.fitness_metric_args:
                        print_1d_best_soln(
                            self.isle.fitness_metric_args['x'],
                            self.isle.fitness_metric_args['y'],
                            self.pareto_isle.pareto_front[0].evaluate,
                            self.isle.fitness_metric,
                            "comparison.png")
            with open("log.txt", "a") as o_file:
                o_file.write("%d\t" % self.age)
                o_file.write("%le\t" % (time.time() - self.start_time))
                for par_indv in self.pareto_isle.pareto_front:
                    o_file.write("%e\t" % par_indv.fitness[0])
                o_file.write("\n")
                if converged:
                    o_file.write("\n")
        else:
            converged = None
        converged = self.comm.bcast(converged, root=0)
        return converged

    def do_final_plots(self, make_plots):
        """
        Makes some summary output

        :param make_plots: boolean for whether to produce plots
        """
        # gather all populations to a single island
        s_pop, p_pop, t_pop = self.isle.dump_populations()
        s_pop = self.comm.gather(s_pop, root=0)
        p_pop = self.comm.gather(p_pop, root=0)
        t_pop = self.comm.gather(t_pop, root=0)
        if self.comm_rank == 0:
            s_pop[0] = s_pop[0] + self.pareto_isle.dump_pareto()
            temp_isle = copy.copy(self.isle)  # TODO should this be deep copy?
            temp_isle.load_populations((s_pop[0], p_pop[0], t_pop[0]))

            # find true pareto front
            temp_isle.use_true_fitness()
            temp_isle.solution_island.update_pareto_front()

            # output the front to screen
            for indv in temp_isle.solution_island.pareto_front:
                LOGGER.info("pareto> %s  %s",
                            str(indv.fitness),
                            indv.latexstring())
            LOGGER.info("BEST_SOLUTION> %s",
                        temp_isle.solution_island.pareto_front[0].
                        latexstring())

            # make plots
            if make_plots:
                print_latex(temp_isle.solution_island.pareto_front, "eq.png")
                print_pareto(temp_isle.solution_island.pareto_front,
                             "front.png")
                if self.isle.fitness_metric_args['x'].shape[1] == 1:
                    if 'y' in self.isle.fitness_metric_args:
                        print_1d_best_soln(
                            self.isle.fitness_metric_args['x'],
                            self.isle.fitness_metric_args['y'],
                            self.pareto_isle.pareto_front[0].evaluate,
                            self.isle.fitness_metric,
                            "comparison.png")

    def save_state(self, filename):
        """
        Currently this relies on everything being serializable and small
        enough to fit in memory on 1 proc

        :param filename: full name of file to save pickle
        """

        if self.comm_rank == 0:
            isles = [self.isle, ]
            for i in range(1, self.comm_size):
                isles.append(self.comm.recv(source=i, tag=6))

            with open(filename, "wb") as out_file:
                pickle.dump((isles, self.pareto_isle, self.age), out_file)
        else:
            self.comm.send(self.isle, dest=0, tag=6)

    def load_state(self, filename):
        """
        Currently this relies on everything being serializable and small
        enough to fit in memory on 1 proc

        :param filename: full name of file to load pickle
        """

        if self.comm_rank == 0:
            with open(filename, "rb") as in_file:
                isles, pareto, age = pickle.load(in_file)

            self.isle = isles[0]
            self.age = age
            self.pareto_isle = pareto
            for i in range(1, self.comm_size):
                if i < len(isles):
                    isle_to_send = isles[i]
                else:
                    isle_to_send = random.choice(isles)
                self.comm.send((isle_to_send, age), dest=i, tag=7)
        else:
            self.pareto_isle = None
            self.isle, self.age = self.comm.recv(source=0, tag=7)


class SerialIslandManager(IslandManager):
    """
    ParallelIslandManager is an implementation of the IslandManager class which
    uses contains many coevoltution islands in a list and performs all
    operations in serial
    """

    def __init__(self, n_islands=2, restart_file=None,
                 *args, **kwargs):
        """
        Initialization of serial island manager.

        :param n_islands: number of coevolution islands to be managed
        :param restart_file: file name from which to load the island manager
        :param args: arguments to be passed to initialization of coevolution
                     islands
        :param kwargs: keyword arguments to be passed to initialization of
                       coevolution islands
        """
        super(SerialIslandManager, self).__init__(*args, **kwargs)

        if restart_file is None:
            self.n_isles = n_islands

            # make coevolution islands
            self.isles = []
            for _ in range(self.n_isles):
                self.isles.append(ci(*args, **kwargs))

            # make dummy island for joint pareto front calculations
            self.pareto_isle = Island(
                self.isles[0].solution_island.gene_manipulator,
                self.isles[0].true_fitness_plus_complexity,
                0, 0, 0)
        else:
            self.load_state(restart_file)

    def do_steps(self, n_steps):
        """
        Steps through generations

        :param n_steps: number of generations through which to step
        """
        t_0 = time.time()
        for i, isle in enumerate(self.isles):
            t_1 = time.time()
            for _ in range(n_steps):
                isle.deterministic_crowding_step()
            t_2 = time.time()
            LOGGER.info("%2d >\tage: %d\ttime: %.1fs\tbest fitness: %s",
                        i, isle.solution_island.age, t_2 - t_1,
                        isle.solution_island.pareto_front[0].fitness)

        t_3 = time.time()
        LOGGER.info("total time: %.1fs", (t_3 - t_0))

        self.age += n_steps

    def do_migration(self):
        """
        Coordinates migration between islands
        """
        # assign partners
        partners = list(range(self.n_isles))
        random.shuffle(partners)

        # get population sizes
        s_pop_size = self.isles[0].solution_island.pop_size
        p_pop_size = self.isles[0].predictor_island.pop_size
        t_pop_size = len(self.isles[0].trainers)

        # loop over partner pairs
        for i in range(self.n_isles//2):
            partner_1 = self.isles[partners[i*2]]
            partner_2 = self.isles[partners[i*2+1]]

            # figure out which individuals will be sent/received from partner 1
            s_to_2, s_to_1 = IslandManager.assign_send_receive(s_pop_size)
            p_to_2, p_to_1 = IslandManager.assign_send_receive(p_pop_size)
            t_to_2, t_to_1 = IslandManager.assign_send_receive(t_pop_size)
            LOGGER.debug("Migration: %2d <-> %2d  mixing = %s",
                         partners[i*2],
                         partners[i*2+1],
                         str((float(len(s_to_2)) / s_pop_size,
                              float(len(p_to_2)) / p_pop_size,
                              float(len(t_to_2)) / t_pop_size)))

            # swap the individuals
            pops_to_2 = partner_1.dump_populations(s_to_2, p_to_2, t_to_2)
            pops_to_1 = partner_2.dump_populations(s_to_1, p_to_1, t_to_1)
            partner_1.load_populations(pops_to_1, s_to_2, p_to_2, t_to_2)
            partner_2.load_populations(pops_to_2, s_to_1, p_to_1, t_to_1)

    def test_convergence(self, epsilon, make_plots):
        """
        Tests for convergence of the island system

        :param epsilon: error which defines convergence
        :param make_plots: boolean for whether to produce plots
        :return: boolean for whether convergence has been reached
        """
        # get list of all pareto individuals
        par_list = []
        for isle in self.isles:
            par_list = par_list + isle.solution_island.dump_pareto()
        par_list = par_list + self.pareto_isle.dump_pareto()

        # load into pareto island
        self.pareto_isle.load_population(par_list)
        self.pareto_isle.update_pareto_front()

        # test convergence
        converged = (self.pareto_isle.pareto_front[0].fitness[0] < epsilon)

        # output
        LOGGER.info("current best true fitness: %s",
                    str(self.pareto_isle.pareto_front[0].fitness[0]))
        LOGGER.info("best solution: %s",
                    self.pareto_isle.pareto_front[0].latexstring())

        if make_plots:
            print_latex(self.pareto_isle.pareto_front, "eq.png")
            print_pareto(self.pareto_isle.pareto_front, "front.png")
            if self.isles[0].fitness_metric_args['x'].shape[1] == 1:
                if 'y' in self.isles[0].fitness_metric_args:
                    print_1d_best_soln(
                        self.isles[0].fitness_metric_args['x'],
                        self.isles[0].fitness_metric_args['y'],
                        self.pareto_isle.pareto_front[0].evaluate,
                        self.isles[0].fitness_metric,
                        "comparison.png")
        with open("log.txt", "a") as o_file:
            o_file.write("%d\t" % self.age)
            for par_indv in self.pareto_isle.pareto_front:
                o_file.write("%e\t" % par_indv.fitness[0])
            o_file.write("\n")

        return converged

    def do_final_plots(self, make_plots):
        """
        Makes some summary output

        :param make_plots: boolean for whether to produce plots
        """
        # gather all populations
        s_pop = []
        p_pop = []
        t_pop = []
        for isle in self.isles:
            s_pop_i, p_pop_i, t_pop_i = isle.dump_populations()
            s_pop = s_pop + s_pop_i
            p_pop = p_pop + p_pop_i
            t_pop = t_pop + t_pop_i
        s_pop = s_pop + self.pareto_isle.dump_population()

        # load them all into a temporary island
        temp_isle = copy.copy(self.isles[0])  # TODO should this be deep copy?
        temp_isle.load_populations((s_pop, p_pop, t_pop))
        temp_isle.use_true_fitness()
        temp_isle.solution_island.update_pareto_front()

        # output
        for indv in temp_isle.solution_island.pareto_front:
            LOGGER.info("pareto> " + str(indv.fitness) +\
                        "  " + indv.latexstring())

        if make_plots:
            print_latex(temp_isle.solution_island.pareto_front, "eq.png")
            print_pareto(temp_isle.solution_island.pareto_front, "front.png")
            if self.isles[0].fitness_metric_args['x'].shape[1] == 1:
                if 'y' in self.isles[0].fitness_metric_args:
                    print_1d_best_soln(
                        self.isles[0].fitness_metric_args['x'],
                        self.isles[0].fitness_metric_args['y'],
                        self.pareto_isle.pareto_front[0].evaluate,
                        self.isles[0].fitness_metric,
                        "comparison.png")
        with open("log.txt", "a") as o_file:
            o_file.write("\n\n")

    def save_state(self, filename):
        """
        currently this relies on everything being serializable

        :param filename: full name of file to save pickle
        """

        with open(filename, "wb") as out_file:
            pickle.dump((self.isles, self.pareto_isle, self.age), out_file)

    def load_state(self, filename):
        """
        currently this relies on everything being serializable

        :param filename: full name of file to load pickle
        """

        with open(filename, "rb") as in_file:
            self.isles, self.pareto_isle, self.age = pickle.load(in_file)
