"""
Island managers manage a group of coevolution islands.  Specifically, they step
through generations, coordinate migration, and test convergence
"""

import time
import random
import abc

from mpi4py import MPI
import numpy as np

from .CoevolutionIsland import CoevolutionIsland as ci
from .Island import Island
from .Plotting import print_latex, print_pareto, print_1d_best_soln


class IslandManager(object):
    """
    IslandManager is an abstract class used for making controllers of groups of
    coevolution islands
    """

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        """initialization of island manager"""
        self.age = 0

    def run_islands(self, max_steps, epsilon, min_steps=0,
                    step_increment=1000):
        """
        Runs co-evolution islands until convergence of best solution
        :param max_steps: maximum number of steps to take
        :param epsilon: error which defines convergence
        :param step_increment: the number of steps between
                               migrations/convergence checks
        :return converged: whether a converged solution has been found
        """
        self.do_steps(n_steps=step_increment)
        converged = self.test_convergence(epsilon)
        while self.age < min_steps or (self.age < max_steps and not converged):
            self.do_migration()
            self.do_steps(n_steps=step_increment)
            converged = self.test_convergence(epsilon)

        self.do_final_plots()

        return converged

    @abc.abstractmethod
    def do_steps(self, n_steps):
        """
        steps through generations
        :param n_steps: number of generations through which to step
        """
        pass

    @abc.abstractmethod
    def do_migration(self):
        """
        coordinates migration between islands
        """
        pass

    @abc.abstractmethod
    def test_convergence(self, epsilon):
        """
        tests for convergence of the island system
        :param epsilon: error which defines convergence
        :return: boolean for whether convergence has been reached
        """
        pass

    @abc.abstractmethod
    def do_final_plots(self):
        """
        makes some summary output
        """
        pass

    @staticmethod
    def assign_send_receive(pop_size):
        """assign indices for exchange through random shuffling"""
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
    """

    def __init__(self, *args, **kwargs):
        super(ParallelIslandManager, self).__init__(*args, **kwargs)

        self.comm = MPI.COMM_WORLD
        self.comm_rank = self.comm.Get_rank()
        self.comm_size = self.comm.Get_size()

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

    def do_steps(self, n_steps):
        """
        steps through generations
        :param n_steps: number of generations through which to step
        """
        t_0 = time.time()
        for i in range(n_steps):
            self.isle.deterministic_crowding_step()
            # print_pareto(isle.solution_island.pareto_front, "front.tif")
        t_1 = time.time()
        print self.comm_rank, ">\tage:", self.isle.solution_island.age,\
            "\ttime: %.1fs" % (t_1 - t_0), \
            "\tbest fitness:", \
            self.isle.solution_island.pareto_front[0].fitness

        if np.isnan(self.isle.solution_island.pareto_front[0].fitness[0]):
            for i in self.isle.solution_island.pop:
                print i.fitness
            for indv in self.isle.solution_island.pareto_front:
                print "pareto>", indv.fitness, indv.latexstring()
        self.age += n_steps

    def do_migration(self):
        """
        coordinates migration between islands
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
                print "Migration:", self.comm_rank, "<->", my_partner, \
                    " mixing =",\
                    (float(len(s_send)) / self.isle.solution_island.pop_size,
                     float(len(p_send)) / self.isle.predictor_island.pop_size,
                     float(len(t_send)) / len(self.isle.trainers))
                self.comm.send((s_receive, p_receive, t_receive),
                               dest=my_partner)

                # exchange populations
                send_package = self.isle.dump_populations(s_send, p_send,
                                                          t_send)
                self.comm.send(send_package, dest=my_partner)
                recv_package = self.comm.recv(source=my_partner)
                self.isle.load_populations(recv_package, s_send, p_send,
                                           t_send)

        # secondary partner
        else:
            my_partner = partners[ind - 1]

            # find which indvs to send/receive
            s_send, p_send, t_send = self.comm.recv(source=my_partner)

            # exchange populations
            send_package = self.isle.dump_populations(s_send, p_send, t_send)
            recv_package = self.comm.recv(source=my_partner)
            self.comm.send(send_package, dest=my_partner)
            self.isle.load_populations(recv_package, s_send, p_send, t_send)

    def test_convergence(self, epsilon):
        """
        tests for convergence of the island system
        :param epsilon: error which defines convergence
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
            print "current best true fitness: ", \
                self.pareto_isle.pareto_front[0].fitness[0]
            print "best solution:", \
                self.pareto_isle.pareto_front[0].latexstring()
            print_latex(self.pareto_isle.pareto_front, "eq.tif")
            print_pareto(self.pareto_isle.pareto_front, "front.tif")
            if self.isle.data_x.shape[1] == 1:
                print_1d_best_soln(self.isle.data_x,
                                   self.isle.data_y,
                                   self.pareto_isle.pareto_front[0].evaluate,
                                   "comparison.tif")
        else:
            converged = None
        converged = self.comm.bcast(converged, root=0)
        return converged

    def do_final_plots(self):
        """
        makes some summary output
        """
        # gather all populations to a single island
        s_pop, p_pop, t_pop = self.isle.dump_populations()
        s_pop = self.comm.gather(s_pop, root=0)
        p_pop = self.comm.gather(p_pop, root=0)
        t_pop = self.comm.gather(t_pop, root=0)
        if self.comm_rank == 0:
            self.isle.load_populations((s_pop[0], p_pop[0], t_pop[0]))

            # find true pareto front
            self.isle.use_true_fitness()
            self.isle.solution_island.update_pareto_front()

            # output the front to screen
            for indv in self.isle.solution_island.pareto_front:
                print "pareto>", indv.fitness, indv.latexstring()

            # make plots
            print_latex(self.isle.solution_island.pareto_front, "eq.tif")
            print_pareto(self.isle.solution_island.pareto_front, "front.tif")
            if self.isle.data_x.shape[1] == 1:
                print_1d_best_soln(
                    self.isle.data_x, self.isle.data_y,
                    self.isle.solution_island.pareto_front[0].evaluate,
                    "comparison.tif")


class SerialIslandManager(IslandManager):
    """
    ParallelIslandManager is an implementation of the IslandManager class which
    uses contains many coevoltution islands in a list and performs all
    operations in serial
    """

    def __init__(self, n_islands=2, *args, **kwargs):
        super(SerialIslandManager, self).__init__(*args, **kwargs)

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

    def do_steps(self, n_steps):
        """
        steps through generations
        :param n_steps: number of generations through which to step
        """
        t_0 = time.time()
        for i, isle in enumerate(self.isles):
            t_1 = time.time()
            for _ in range(n_steps):
                isle.deterministic_crowding_step()
            t_2 = time.time()

            print i, ">\tage:", isle.solution_island.age, \
                "\ttime: %.1fs" % (t_2 - t_1), \
                "\tbest fitness:", \
                isle.solution_island.pareto_front[0].fitness

        t_3 = time.time()
        print "total time: %.1fs" % (t_3 - t_0)

        self.age += n_steps

    def do_migration(self):
        """
        coordinates migration between islands
        """
        # assign partners
        partners = list(range(self.n_isles))
        random.shuffle(partners)

        # get population sizes
        s_pop_size = self.isles[0].solution_island.pop_size
        p_pop_size = self.isles[0].predictor_island.pop_size
        t_pop_size = len(self.isles[0].trainers)

        # loop over partner pairs
        for i in range(self.n_isles/2):
            partner_1 = self.isles[partners[i*2]]
            partner_2 = self.isles[partners[i*2+1]]

            # figure out which individuals will be sent/received from partner 1
            s_to_2, s_to_1 = IslandManager.assign_send_receive(s_pop_size)
            p_to_2, p_to_1 = IslandManager.assign_send_receive(p_pop_size)
            t_to_2, t_to_1 = IslandManager.assign_send_receive(t_pop_size)
            print "Migration:", partners[i*2], "<->", partners[i*2+1], \
                " mixing =",\
                (float(len(s_to_2)) / s_pop_size,
                 float(len(p_to_2)) / p_pop_size,
                 float(len(t_to_2)) / t_pop_size)

            # swap the individuals
            pops_to_2 = partner_1.dump_populations(s_to_2, p_to_2, t_to_2)
            pops_to_1 = partner_2.dump_populations(s_to_1, p_to_1, t_to_1)
            partner_1.load_populations(pops_to_1, s_to_2, p_to_2, t_to_2)
            partner_2.load_populations(pops_to_2, s_to_1, p_to_1, t_to_1)

    def test_convergence(self, epsilon):
        """
        tests for convergence of the island system
        :param epsilon: error which defines convergence
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
        print "current best true fitness: ", \
            self.pareto_isle.pareto_front[0].fitness[0]
        print "best solution:", self.pareto_isle.pareto_front[0].latexstring()

        print_latex(self.pareto_isle.pareto_front, "eq.tif")
        print_pareto(self.pareto_isle.pareto_front, "front.tif")
        if self.isles[0].data_x.shape[1] == 1:
            print_1d_best_soln(self.isles[0].data_x,
                               self.isles[0].data_y,
                               self.pareto_isle.pareto_front[0].evaluate,
                               "comparison.tif")

        return converged

    def do_final_plots(self):
        """
        makes some summary output
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

        # load them all into the first island and update
        self.isles[0].load_populations((s_pop, p_pop, t_pop))
        self.isles[0].use_true_fitness()
        self.isles[0].solution_island.update_pareto_front()

        # output
        for indv in self.isles[0].solution_island.pareto_front:
            print "pareto>", indv.fitness, indv.latexstring()
        print_latex(self.isles[0].solution_island.pareto_front, "eq.tif")
        print_pareto(self.isles[0].solution_island.pareto_front, "front.tif")
        if self.isles[0].data_x.shape[1] == 1:
            print_1d_best_soln(
                self.isles[0].data_x, self.isles[0].data_y,
                self.isles[0].solution_island.pareto_front[0].evaluate,
                "comparison.tif")
