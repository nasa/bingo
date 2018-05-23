"""
example of regression done using the parallel island manager (islands done
in parallel on multiple mpi processes)
"""

import random
import logging
from mpi4py import MPI
import numpy as np

from bingo.AGraph import AGraphManipulator as agm
from bingo.AGraph import AGNodes
from bingo import AGraphCpp
from bingo.FitnessPredictor import FPManipulator as fpm
from bingo.IslandManager import ParallelIslandManager
from bingo.FitnessMetric import ImplicitRegression, StandardRegression
from bingo.TrainingData import ExplicitTrainingData, ImplicitTrainingData
from bingocpp.build import bingocpp


logging.basicConfig(level=logging.INFO, format='%(message)s')


def make_circle_data(data_size):
    """makes test data for circular constant regression"""
    x = np.empty([data_size, 2])
    for i, theta in enumerate(np.linspace(0, 3.14 * 2, data_size)):
        x[i, 0] = np.cos(theta)
        x[i, 1] = np.sin(theta)
    return x


def make_norm_data(data_size):
    """makes test data for finding 3d norm with standard regression"""
    n_lin = int(np.power(data_size, 1.0/3)) + 1
    x_1 = np.linspace(0, 5, n_lin)
    x_2 = np.linspace(0, 5, n_lin)
    x_3 = np.linspace(0, 5, n_lin)
    x = np.array(np.meshgrid(x_1, x_2, x_3)).T.reshape(-1, 3)
    x = x[np.random.choice(x.shape[0], data_size, replace=False), :]
    # make solution
    y = (np.linalg.norm(x, axis=1))
    y = y.reshape([-1, 1])

    return x, y


def make_1d_data(data_size, test_num):
    """makes test data for 1d standard symbolic regression"""
    x = np.empty(0)
    if test_num == 1:
        x = np.linspace(-2, 2, data_size, False)
        y = x * x + 0.5
    elif test_num == 2:
        x = np.linspace(-5, 5, data_size, False)
        y = 1.5*x*x - x*x*x
    elif test_num == 3:
        x = np.linspace(-3, 3, data_size, False)
        y = np.exp(np.abs(x))*np.sin(x)
    elif test_num == 4:
        x = np.linspace(-10, 10, data_size, False)
        y = x*x*np.exp(np.sin(x)) + x + np.sin(3.14159/4 - x*x*x)
    x = x.reshape([-1, 1])
    y = y.reshape([-1, 1])
    return x, y


def make_sphere_data(data_size):
    """makes test data for spherical constant regression"""
    x = np.empty([data_size, 3])
    for i in range(data_size):
        phi = (3.140*.9*i)/data_size+3.14*.05
        theta = (3.14*20*i)/data_size+3.14/2
        x[i, 0] = np.cos(theta)*np.sin(phi)
        x[i, 1] = np.sin(theta)*np.sin(phi)
        x[i, 2] = np.cos(phi)
        i += 1
    return x


def make_sphere_data_changing_rad(data_size):
    """makes test data for spherical constant regression with varying radius"""
    x = np.empty([data_size, 4])
    radius = 10
    for i in range(data_size):
        if random.random() > 0.5:
            radius += 0.2
        else:
            radius += -0.2
        phi = (3.140 * .9 * i) / data_size + 3.14 * .05
        theta = (3.14 * 20 * i) / data_size + 3.14 / 2
        x[i, 0] = np.cos(theta) * np.sin(phi) * radius
        x[i, 1] = np.sin(theta) * np.sin(phi) * radius
        x[i, 2] = np.cos(phi) * radius
        x[i, 3] = radius
        i += 1
    return x


def main(max_steps, epsilon, data_size):
    """main function which runs regression"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # load data on rank 0 and send it to all ranks
    x_true = None
    y_true = None
    if rank == 0:
        # --------------------------------------------
        # MAKE DATA (uncomment one of the below lines)
        # --------------------------------------------

        # standard regression
        # (use with ExplicitTrainingData and StandardRegression)
        # x_true, y_true = make_1d_data(data_size, 1)
        x_true, y_true = make_1d_data(data_size, 2)
        # x_true, y_true = make_1d_data(data_size, 3)
        # x_true, y_true = make_1d_data(data_size, 4)
        # x_true, y_true = make_norm_data(data_size)

        # implicit regression
        # (use with ImplicitTrainingData and ImplicitRegression)
        # x_true = make_circle_data(data_size)

    # then broadcast to all ranks
    x_true = MPI.COMM_WORLD.bcast(x_true, root=0)
    y_true = MPI.COMM_WORLD.bcast(y_true, root=0)

    # ------------------------------------------------
    # MAKE TRAINING DATA  (uncomment one of the below)
    # ------------------------------------------------
    training_data = ExplicitTrainingData(x_true, y_true)
    # training_data = ImplicitTrainingData(x_true)

    # ------------------------------------------------------------
    # MAKE SOLUTION MANIPULATOR (uncomment one of the below blocks)
    # ------------------------------------------------------------

    # using AGraph.py for the solution manipulator
    # sol_manip = agm(x_true.shape[1], 32, nloads=2)
    # sol_manip.add_node_type(AGNodes.Add)
    # sol_manip.add_node_type(AGNodes.Subtract)
    # sol_manip.add_node_type(AGNodes.Multiply)
    # sol_manip.add_node_type(AGNodes.Divide)
    # # sol_manip.add_node_type(AGNodes.Exp)
    # # sol_manip.add_node_type(AGNodes.Log)
    # # sol_manip.add_node_type(AGNodes.Sin)
    # # sol_manip.add_node_type(AGNodes.Cos)
    # # sol_manip.add_node_type(AGNodes.Abs)
    # sol_manip.add_node_type(AGNodes.Sqrt)

    # using AGraphCpp.py for the solution manipulator
    sol_manip = AGraphCpp.AGraphCppManipulator(x_true.shape[1], 32, nloads=2)
    sol_manip.add_node_type(2)  # +
    sol_manip.add_node_type(3)  # -
    sol_manip.add_node_type(4)  # *
    sol_manip.add_node_type(5)  # /
    sol_manip.add_node_type(12)  # sqrt

    # ----------------------------------
    # MAKE FITNESS PREDICTOR MANIPULATOR
    # ----------------------------------
    pred_manip = fpm(128, data_size)

    # ------------------------------------------------
    # MAKE FITNESS METRIC (uncomment one of the below)
    # ------------------------------------------------
    regressor = StandardRegression(const_deriv=True)
    # regressor = ImplicitRegression()

    # --------------------------------------
    # MAKE SERIAL ISLAND MANAGER THEN RUN IT
    # --------------------------------------
    islmngr = ParallelIslandManager(solution_training_data=training_data,
                                    solution_manipulator=sol_manip,
                                    predictor_manipulator=pred_manip,
                                    solution_pop_size=64,
                                    fitness_metric=regressor,
                                    solution_age_fitness=True)
    islmngr.run_islands(max_steps, epsilon, min_steps=1000,
                        step_increment=1000)


if __name__ == "__main__":

    MAX_STEPS = 2000
    CONVERGENCE_EPSILON = 0.001
    DATA_SIZE = 500

    bingocpp.rand_init()

    main(MAX_STEPS, CONVERGENCE_EPSILON, DATA_SIZE)
