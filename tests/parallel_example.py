"""
example of regression done using the parallel island manager (islands done
in parallel on multiple mpi processes)
"""

import math
from mpi4py import MPI
import numpy as np

from bingo.AGraph import AGraphManipulator as agm
from bingo.AGraph import AGNodes
from bingo.FitnessPredictor import FPManipulator as fpm
from bingo.IslandManager import ParallelIslandManager


def main(max_steps, epsilon, data_size, data_range):
    """main function which runs regression"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # load data on rank 0
    if rank == 0:
        # make data
        # X = np.linspace(data_range[0], data_range[1], data_size, False)
        # Y = X * X + 0.5
        # Y = 1.5*X*X - X*X*X
        # Y = np.exp(np.abs(X))*np.sin(X)
        # Y = X*X*np.exp(np.sin(X)) + X + np.sin(3.14159/4 - X*X*X)
        # X = X.reshape([-1, 1])

        X = np.empty([data_size, 2])
        for i, theta in enumerate(np.linspace(0, 3.14 * 2, data_size)):
            X[i, 0] = math.cos(theta)
            X[i, 1] = math.sin(theta)
        Y = None

    else:
        X = None
        Y = None
    # then broadcast to all ranks
    X = MPI.COMM_WORLD.bcast(X, root=0)
    Y = MPI.COMM_WORLD.bcast(Y, root=0)

    # make solution manipulator
    sol_manip = agm(X.shape[1], 64, nloads=2)
    sol_manip.add_node_type(AGNodes.Add)
    sol_manip.add_node_type(AGNodes.Subtract)
    sol_manip.add_node_type(AGNodes.Multiply)
    # sol_manip.add_node_type(AGNodes.Divide)
    # sol_manip.add_node_type(AGNodes.Exp)
    # sol_manip.add_node_type(AGNodes.Log)
    # sol_manip.add_node_type(AGNodes.Sin)
    # sol_manip.add_node_type(AGNodes.Cos)
    # sol_manip.add_node_type(AGNodes.Abs)

    # make predictor manipulator
    pred_manip = fpm(16, data_size)

    # make and run island manager
    islmngr = ParallelIslandManager(X, Y, sol_manip, pred_manip,
                                    solution_pop_size=64)
    islmngr.run_islands(max_steps, epsilon, step_increment=100)


if __name__ == "__main__":

    MAX_STEPS = 400
    CONVERGENCE_EPSILON = 0.01
    DATA_SIZE = 100
    DATA_RANGE = [-3, 3]

    main(MAX_STEPS, CONVERGENCE_EPSILON, DATA_SIZE, DATA_RANGE)
