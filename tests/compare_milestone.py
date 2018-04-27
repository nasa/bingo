"""
example of regression done using the parallel island manager (islands done
in parallel on multiple mpi processes) - showing difference between
blocking and non blocking
"""

import math
# import random
import time
from mpi4py import MPI
import numpy as np
import logging 
logging.basicConfig(level=logging.INFO, format='%(message)s')

from bingo.AGraph import AGraphManipulator as agm
from bingo.AGraph import AGNodes
from bingo import AGraphCpp
from bingo.FitnessPredictor import FPManipulator as fpm
from bingo.IslandManager import ParallelIslandManager
from bingo.FitnessMetric import StandardRegression, ImplicitRegression
from bingo.TrainingData import ExplicitTrainingData, ImplicitTrainingData
from bingocpp.build import bingocpp

agesN = list()
agesF = list()
timesN = list()
timesF = list()

def main(max_steps, epsilon, data_size):
    """main function which runs regression"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # load data on rank 0
    if rank == 0:
        # make data
        n_lin = int(math.pow(data_size, 1.0/3)) + 1
        x_1 = np.linspace(0, 5, n_lin)
        x_2 = np.linspace(0, 5, n_lin)
        x_3 = np.linspace(0, 5, n_lin)
        x = np.array(np.meshgrid(x_1, x_2, x_3)).T.reshape(-1, 3)
        x = x[np.random.choice(x.shape[0], data_size, replace=False), :]
        # make solution
        # y = (x[:, 0]+3.5*x[:, 1])
        y = (x[:,0]*x[:,0]+3.5*x[:,1])
        x_true = x
        y_true = y
    else:
        x_true = None
        y_true = None
    # then broadcast to all ranks
    x_true = MPI.COMM_WORLD.bcast(x_true, root=0)
    y_true = MPI.COMM_WORLD.bcast(y_true, root=0)

    # make solution manipulator
    # sol_manip = agm(x_true.shape[1], 64, nloads=2)
    # sol_manip.add_node_type(AGNodes.Add)
    # sol_manip.add_node_type(AGNodes.Subtract)
    # sol_manip.add_node_type(AGNodes.Multiply)
    # sol_manip.add_node_type(AGNodes.Divide)
    # sol_manip.add_node_type(AGNodes.Exp)
    # sol_manip.add_node_type(AGNodes.Log)
    # sol_manip.add_node_type(AGNodes.Sin)
    # sol_manip.add_node_type(AGNodes.Cos)
    # sol_manip.add_node_type(AGNodes.Abs)
    # sol_manip.add_node_type(AGNodes.Sqrt)


    # make solution manipulator
    y_true = y_true.reshape(-1, 1)
    sol_manip2 = AGraphCpp.AGraphCppManipulator(x_true.shape[1], 64, nloads=2)
    # sol_manip2 = bingocpp.AcyclicGraphManipulator(x_true.shape[1], 64, nloads=2)
    # sol_manip2 = bingocpp.AcyclicGraphManipulator(x_true.shape[1], 64, nloads=2, opt_rate=0)

    # sol_manip.add_node_type(2)  # +
    # sol_manip.add_node_type(3)  # -
    # sol_manip.add_node_type(4)  # *
    # sol_manip.add_node_type(5)  # /
    # sol_manip.add_node_type(6)  # sin
    # sol_manip.add_node_type(7)  # cos
    # sol_manip.add_node_type(8)  # exp
    # sol_manip.add_node_type(9)  # log
    # # sol_manip.add_node_type(10)  # pow
    # sol_manip.add_node_type(11)  # abs
    # sol_manip.add_node_type(12)  # sqrt


    sol_manip2.add_node_type(2)  # +
    sol_manip2.add_node_type(3)  # -
    sol_manip2.add_node_type(4)  # *
    sol_manip2.add_node_type(5)  # /
    sol_manip2.add_node_type(6)  # sin
    sol_manip2.add_node_type(7)  # cos
    sol_manip2.add_node_type(8)  # exp
    sol_manip2.add_node_type(9)  # log
    # sol_manip2.add_node_type(10)  # pow
    sol_manip2.add_node_type(11)  # abs
    sol_manip2.add_node_type(12)  # sqrt

    # make predictor manipulator
    pred_manip = fpm(128, data_size)

    # make training data
    # training_data = ImplicitTrainingData(x_true)
    training_data = ExplicitTrainingData(x_true, y_true)
    training_data2 = bingocpp.ExplicitTrainingData(x_true, y_true)

    # make fitness metric
    # explicit_regressor = ImplicitRegression()
    explicit_regressor = StandardRegression(const_deriv=True)
    explicit_regressor2 = bingocpp.StandardRegression()


    # make and run island manager
    islmngr = ParallelIslandManager(#restart_file='test.p',
        solution_training_data=training_data,
        solution_manipulator=sol_manip2,
        predictor_manipulator=pred_manip,
        solution_pop_size=64,
        fitness_metric=explicit_regressor)

    # islmngr2 = ParallelIslandManager(#restart_file='test.p',
    #     solution_training_data=training_data,
    #     solution_manipulator=sol_manip,
    #     predictor_manipulator=pred_manip,
    #     solution_pop_size=64,
    #     fitness_metric=explicit_regressor)


    # islmngr = ParallelIslandManager(#restart_file='test.p',
    #     data_x=x_true, data_y=y_true,
    #     solution_manipulator=sol_manip,
    #     predictor_manipulator=pred_manip,
    #     solution_pop_size=64,
    #     fitness_metric=StandardRegression)

    # islmngr2 = ParallelIslandManager(#restart_file='test.p',
    #     data_x=x_true, data_y=y_true,
    #     solution_manipulator=sol_manip,
    #     predictor_manipulator=pred_manip,
    #     solution_pop_size=64,
    #     fitness_metric=StandardRegression)
    non_one = time.time()
    islmngr.run_islands(max_steps, epsilon, min_steps=500,
                        step_increment=500, when_update=50)
    non_two = time.time()
    non_time = non_two - non_one
    
    timesN.append(non_time)
    agesN.append(islmngr.age)

    # fit_one = time.time()
    # # run island manager with blocking mpi
    # islmngr2.run_islands(max_steps, epsilon, min_steps=500,
    #                      step_increment=500, when_update=50)
    # fit_two = time.time()
    # fit_time = fit_two - fit_one
    
    # timesF.append(fit_time)
    # agesF.append(islmngr2.age)


if __name__ == "__main__":

    MAX_STEPS = 30000
    CONVERGENCE_EPSILON = 0.001
    DATA_SIZE = 500

    # if using c++
    bingocpp.rand_init()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    for x in range(0, 10):
        # if rank == 0:
        #     print("CYCLE:", x + 1)
        main(MAX_STEPS, CONVERGENCE_EPSILON, DATA_SIZE)
    if rank == 0:
        print("C++ agcpp times:", timesN)
        print("C++ agcpp times avg:", np.mean(timesN))
        print("C++ agcpp ages:", agesN)
        print("C++ agcpp ages avg:", np.mean(agesN))
        # print("Python times:", timesF)
        # print("Python times avg:", np.mean(timesF))
        # print("Python ages:", agesF)
        # print("Python ages avg:", np.mean(agesF))
