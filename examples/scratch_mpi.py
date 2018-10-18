#mpitest.py
import random
from mpi4py import MPI
import numpy as np
import time
import pickle

from AGraph import AGraphManipulator as agm
from AGraph import AGNodes
from FitnessPredictor import FPManipulator as fpm
from CoevolutionIsland import CoevolutionIsland as ci
from Plotting import print_latex, print_pareto, print_1d_best_soln


comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()


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




full_data_size = 100
data_range = [-5, 5]

if comm_rank == 0:
    print("making data on proc", comm_rank)
    # make data
    X = np.linspace(data_range[0], data_range[1], full_data_size, False)
    #Y = X*X + 0.5
    Y = 1.5*X*X - X*X*X
    #Y = np.exp(np.abs(X))*np.sin(X)
    #Y = X*X*np.exp(np.sin(X)) + X + np.sin(3.14159/4 - X*X*X)
    X = X.reshape([-1, 1])
    #print comm_rank, "/", comm_size, " X:", X
else:
    X = None
    Y = None

X = comm.bcast(X, root=0)
Y = comm.bcast(Y, root=0)

# print comm_rank, "/", comm_size


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
pred_manip = fpm(128, full_data_size)

# make coevolution island
isle = ci(X, Y,
          64, 0.7, 0.01, sol_manip,
          16, 0.5, 0.1, pred_manip, 0.1, 50,
          16, 50,
          comm_rank == -1)


# --do steps--
t0 = time.time()
for i in range(1000):
    isle.deterministic_crowding_step()
    # print_pareto(isle.solution_island.pareto_front, "front.tif")
t1 = time.time()
print(comm_rank, "/", comm_size, "> time for 1000 steps:", t1-t0)


# --do migration--
# assign partners
if comm_rank == 0:
    partners = list(range(comm_size))
    random.shuffle(partners)
    print("Migration: partners=", partners)
else:
    partners = None
partners = comm.bcast(partners, root=0)
ind = partners.index(comm_rank)
primary = (ind % 2 == 0)
if primary:
    my_partner = partners[ind+1]
    s_send, s_receive = assign_send_receive(isle.solution_island.pop_size)
    p_send, p_receive = assign_send_receive(isle.predictor_island.pop_size)
    t_send, t_receive = assign_send_receive(len(isle.trainers))
    print("Migration:", comm_rank, "<->", my_partner, " mixing =", \
          (float(len(s_send)) / isle.solution_island.pop_size,
           float(len(p_send)) / isle.predictor_island.pop_size,
           float(len(t_send)) / len(isle.trainers)))
    comm.send((s_receive, p_receive, t_receive), dest=my_partner)
    send_package = isle.dump_populations(s_send, p_send, t_send)
    comm.send(send_package, dest=my_partner)
    recv_package = comm.recv(source=my_partner)
    isle.load_populations(recv_package, s_send, p_send, t_send)
else:
    my_partner = partners[ind-1]
    s_send, p_send, t_send = comm.recv(source=my_partner)
    send_package = isle.dump_populations(s_send, p_send, t_send)
    recv_package = comm.recv(source=my_partner)
    comm.send(send_package, dest=my_partner)
    isle.load_populations(recv_package, s_send, p_send, t_send)


# --do more steps--
t0 = time.time()
for i in range(1000):
    isle.deterministic_crowding_step()
    # print_pareto(isle.solution_island.pareto_front, "front.tif")
t1 = time.time()
print(comm_rank, "/", comm_size, "> time for another 1000 steps:", t1-t0)



# end work
s_pop, p_pop, t_pop = isle.dump_populations()
s_pop = comm.gather(s_pop, root=0)
p_pop = comm.gather(p_pop, root=0)
t_pop = comm.gather(t_pop, root=0)
if comm_rank == 0:
    isle.load_populations((s_pop[0], p_pop[0], t_pop[0]))
    isle.use_true_fitness()
    isle.solution_island.update_pareto_front()

    for indv in isle.solution_island.pareto_front:
        print("pareto>", indv.fitness, indv.get_latex_string())
    print_latex(isle.solution_island.pareto_front, "eq.tif")
    print_pareto(isle.solution_island.pareto_front, "front.tif")
    print_1d_best_soln(X, Y,
                       isle.solution_island.pareto_front[0].evaluate_equation_at,
                       "comparison.tif")


