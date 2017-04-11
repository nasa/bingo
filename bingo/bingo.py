import time
import random
from mpi4py import MPI
import numpy as np

from AGraph import AGraphManipulator as agm
from AGraph import AGNodes
from FitnessPredictor import FPManipulator as fpm
from CoevolutionIsland import CoevolutionIsland as ci
from Island import Island
from Plotting import print_latex, print_pareto, print_1d_best_soln

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


def main(X, Y, max_steps, epsilon):
    """main function where evolution takes place"""

    # make solution manipulator
    sol_manip = agm(1, 128, nloads=2)
    sol_manip.add_node_type(AGNodes.Add)
    sol_manip.add_node_type(AGNodes.Subtract)
    sol_manip.add_node_type(AGNodes.Multiply)
    sol_manip.add_node_type(AGNodes.Divide)
    sol_manip.add_node_type(AGNodes.Exp)
    sol_manip.add_node_type(AGNodes.Log)
    sol_manip.add_node_type(AGNodes.Sin)
    sol_manip.add_node_type(AGNodes.Cos)
    sol_manip.add_node_type(AGNodes.Abs)

    # make predictor manipulator
    pred_manip = fpm(128, DATA_SIZE)

    # make coevolution island
    isle = ci(X, Y,
              64, 0.7, 0.01, sol_manip,
              16, 0.5, 0.1, pred_manip, 0.1, 50,
              16, 50,
              comm_rank == -1)

    # make dummy island for joint pareto front calculations
    if comm_rank == 0:
        pareto_isle = Island(sol_manip,isle.true_fitness_plus_complexity,
                             0, 0, 0)
    else:
        pareto_isle = None

    do_steps(isle, n_steps=1000)
    converged = test_convergence(isle, pareto_isle, epsilon)

    while isle.solution_island.age < max_steps and not converged:
        do_migration(isle)
        do_steps(isle, n_steps=1000)
        converged = test_convergence(isle, pareto_isle, epsilon)

    do_final_plots(isle)


def test_convergence(isle, pareto_isle, epsilon):
    par_list = isle.solution_island.dump_pareto()
    par_list = comm.gather(par_list, root=0)
    if comm_rank == 0:
        par_list = par_list[0] + pareto_isle.dump_pareto()
        pareto_isle.load_population(par_list)
        pareto_isle.update_pareto_front()
        converged = (pareto_isle.pareto_front[0].fitness[0] < epsilon)

        # output
        print "current best true fitness: ", \
            pareto_isle.pareto_front[0].fitness[0]
        print_latex(pareto_isle.pareto_front, "eq.tif")
        print_pareto(pareto_isle.pareto_front, "front.tif")
        print_1d_best_soln(X, Y,
                           pareto_isle.pareto_front[0].evaluate,
                           "comparison.tif")
    else:
        converged = None
    converged = comm.bcast(converged, root=0)
    return converged


def do_final_plots(isle):
    s_pop, p_pop, t_pop = isle.dump_populations()
    s_pop = comm.gather(s_pop, root=0)
    p_pop = comm.gather(p_pop, root=0)
    t_pop = comm.gather(t_pop, root=0)
    if comm_rank == 0:
        isle.load_populations((s_pop[0], p_pop[0], t_pop[0]))
        isle.use_true_fitness()
        isle.solution_island.update_pareto_front()

        for indv in isle.solution_island.pareto_front:
            print "pareto>", indv.fitness, indv.latexstring()
        print_latex(isle.solution_island.pareto_front, "eq.tif")
        print_pareto(isle.solution_island.pareto_front, "front.tif")
        print_1d_best_soln(X, Y,
                           isle.solution_island.pareto_front[0].evaluate,
                           "comparison.tif")


def do_migration(isle):
    # assign partners
    if comm_rank == 0:
        partners = list(range(comm_size))
        random.shuffle(partners)
    else:
        partners = None
    partners = comm.bcast(partners, root=0)
    ind = partners.index(comm_rank)
    primary = (ind % 2 == 0)
    if primary:
        if comm_rank != partners[-1]:
            my_partner = partners[ind + 1]
            s_send, s_receive = \
                assign_send_receive(isle.solution_island.pop_size)
            p_send, p_receive = \
                assign_send_receive(isle.predictor_island.pop_size)
            t_send, t_receive = assign_send_receive(len(isle.trainers))
            print "Migration:", comm_rank, "<->", my_partner, " mixing =", \
                (float(len(s_send)) / isle.solution_island.pop_size,
                 float(len(p_send)) / isle.predictor_island.pop_size,
                 float(len(t_send)) / len(isle.trainers))
            comm.send((s_receive, p_receive, t_receive), dest=my_partner)
            send_package = isle.dump_populations(s_send, p_send, t_send)
            comm.send(send_package, dest=my_partner)
            recv_package = comm.recv(source=my_partner)
            isle.load_populations(recv_package, s_send, p_send, t_send)
    else:
        my_partner = partners[ind - 1]
        s_send, p_send, t_send = comm.recv(source=my_partner)
        send_package = isle.dump_populations(s_send, p_send, t_send)
        recv_package = comm.recv(source=my_partner)
        comm.send(send_package, dest=my_partner)
        isle.load_populations(recv_package, s_send, p_send, t_send)


def assign_send_receive(pop_size):
    """assign indices for exchange through random shuffling"""
    s_shuffle = list(range(pop_size*2))
    random.shuffle(s_shuffle)
    s_to_send = []
    s_to_receive = []
    for i, s in enumerate(s_shuffle):
        my_new = (i < pop_size)
        my_old = (s < pop_size)
        if my_new and not my_old:
            s_to_receive.append(s-pop_size)
        if not my_new and my_old:
            s_to_send.append(s)
    assert len(s_to_send) == len(s_to_receive)
    return s_to_send, s_to_receive


def do_steps(isle, n_steps):
    t0 = time.time()
    for i in range(n_steps):
        isle.deterministic_crowding_step()
        # print_pareto(isle.solution_island.pareto_front, "front.tif")
    t1 = time.time()
    print comm_rank, ">\tage:", isle.solution_island.age,\
        "\ttime: %.1fs" % (t1 - t0), \
        "\tbest fitness:", isle.solution_island.pareto_front[0].fitness

    if np.isnan(isle.solution_island.pareto_front[0].fitness[0]):
        for i in isle.solution_island.pop:
            print i.fitness
        for indv in isle.solution_island.pareto_front:
            print "pareto>", indv.fitness, indv.latexstring()


if __name__ == "__main__":

    MAX_STEPS = 10000
    CONVERGENCE_EPSILON = 0.01
    DATA_SIZE = 100
    DATA_RANGE = [-3, 3]

    # load data on rank 0
    if comm_rank == 0:
        # make data
        X = np.linspace(DATA_RANGE[0], DATA_RANGE[1], DATA_SIZE, False)
        # Y = X*X + 0.5
        # Y = 1.5*X*X - X*X*X
        Y = np.exp(np.abs(X))*np.sin(X)
        # Y = X*X*np.exp(np.sin(X)) + X + np.sin(3.14159/4 - X*X*X)
        X = X.reshape([-1, 1])
    else:
        X = None
        Y = None
    # then broadcast to all ranks
    X = comm.bcast(X, root=0)
    Y = comm.bcast(Y, root=0)

    main(X, Y, MAX_STEPS, CONVERGENCE_EPSILON)



