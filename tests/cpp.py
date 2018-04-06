"""
example of regression done using the parallel island manager (islands done
in parallel on multiple mpi processes) - showing difference between
blocking and non blocking using agraphCPP
"""

import math
# from matplotlib import pyplot
# from mpl_toolkits.mplot3d import Axes3D
import random
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
from bingo.TrainingData import ExplicitTrainingData
from bingocpp.build import bingocpp

agesN = list();
timesN = list();

def main(max_steps, epsilon, data_size):
    bingocpp.rand_init()

    """main function which runs regression"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # fig = pyplot.figure()
    # ax = Axes3D(fig)

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
        # y = x[:, 0] + x[:, 2]
        y = y.reshape(-1, 1)
        x_true = x
        y_true = y
        # ax.scatter(x[:,0], x[:,1], y)

        # ax.set_xlabel('x_0')
        # ax.set_ylabel('x_1')
        # ax.set_zlabel('y')
        # pyplot.show()
        # np.savetxt('arrayx.txt', x)
        # np.savetxt('arrayy.txt', y)
        # print("y\n", y_true)
    else:
        x_true = None
        y_true = None
    # then broadcast to all ranks
    x_true = MPI.COMM_WORLD.bcast(x_true, root=0)
    y_true = MPI.COMM_WORLD.bcast(y_true, root=0)

    # make solution manipulator
    # sol_manip2 = AGraphCpp.AGraphCppManipulator(x_true.shape[1], 16, nloads=2)
    sol_manip2 = bingocpp.AcyclicGraphManipulator(x_true.shape[1], 16, nloads=2)
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
    # training_data = ExplicitTrainingData(x_true, y_true)
    training_data = bingocpp.ExplicitTrainingData(x_true, y_true)

    # make fitness metric
    # explicit_regressor = StandardRegression()
    explicit_regressor = bingocpp.StandardRegression()

    # make and run island manager
    islmngr = ParallelIslandManager(#restart_file='test.p',
        solution_training_data=training_data,
        solution_manipulator=sol_manip2,
        predictor_manipulator=pred_manip,
        solution_pop_size=64,
        fitness_metric=explicit_regressor)

    non_one = time.time()
    islmngr.run_islands(max_steps, epsilon, min_steps=1000,
                        step_increment=1000, when_update=100)
    non_two = time.time()
    non_time = non_two - non_one

    timesN.append(non_time)
    agesN.append(islmngr.age)


if __name__ == "__main__":

    MAX_STEPS = 30000
    CONVERGENCE_EPSILON = 0.001
    DATA_SIZE = 500

    bingocpp.rand_init()

    for x in range(0, 10):
        main(MAX_STEPS, CONVERGENCE_EPSILON, DATA_SIZE)
        print("CYCLE:", x + 1)
    print("Non-blocking times:", timesN)
    print("Non-blocking ages:", agesN)
