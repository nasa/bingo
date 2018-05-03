"""
The basics to get started with bingo™
"""

import math
import time
from mpi4py import MPI
import numpy as np
import logging 
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(module)s:   %(message)s')
import matplotlib
matplotlib.use('Agg')

from bingo import AGraphCpp
from bingo.FitnessPredictor import FPManipulator as fpm
from bingo.IslandManager import ParallelIslandManager
from bingo.FitnessMetric import StandardRegression, ImplicitRegression
from bingo.TrainingData import ExplicitTrainingData, ImplicitTrainingData

def main(max_steps, epsilon, data_size, mpi):
    # STEP 1
    # Create your x and y data, on parallel, broadcast it to all other ranks

    ##################################################
    ##################### SINGLE #####################
    ##################################################
    if not mpi:
        n_lin = int(math.pow(data_size, 1.0/3)) + 1
        x_1 = np.linspace(0, 5, n_lin)
        x_2 = np.linspace(0, 5, n_lin)
        x_3 = np.linspace(0, 5, n_lin)
        x = np.array(np.meshgrid(x_1, x_2, x_3)).T.reshape(-1, 3)
        x = x[np.random.choice(x.shape[0], data_size, replace=False), :]
        #x_true = np.log(np.array([2e-8, 4e-8, 1.6e-7, 3.2e-7, 6.4e-7, 1.28e-6, 2.56e-6, 5e-6]))
        #y_true = np.log(np.array([1.002e15, 6.151e14, 4.082e14, 4.898e14, 4.913e14, 3.984e14, 2.464e14, 1.078e14]))
        # make solution
        y = (x[:,0]*x[:,0]+3.5*x[:,1])
        x_true = x
        y_true = y
        y_true = y_true.reshape(-1, 1)
    ##################################################
    ##################################################
    ##################################################


    ##################################################
    #################### PARALLEL ####################
    ##################################################
    if mpi:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            #n_lin = int(math.pow(data_size, 1.0/3)) + 1
            #x_1 = np.linspace(0, 5, n_lin)
            #x_2 = np.linspace(0, 5, n_lin)
            #x_3 = np.linspace(0, 5, n_lin)
            #x = np.array(np.meshgrid(x_1, x_2, x_3)).T.reshape(-1, 3)
            #x = x[np.random.choice(x.shape[0], data_size, replace=False), :]

            x_true = np.array([i for i in range(25,425,25)])  # 25e-9:25e-9:400e-9])
            y_true = np.array([5.38, 2.91, 2.07, 1.71, 1.46, 1.35, 1.29, 1.24, 1.2, 1.19, 1.22, 1.23, 1.23, 1.23, 1.26, 1.26])
            
            #x_true = np.array([2e-8, 4e-8, 8e-8, 1.6e-7, 3.2e-7, 6.4e-7, 1.28e-6, 2.56e-6, 5e-6])*1e8
            #y_true = np.array([1.002e15, 6.151e14, 4.435e14, 4.082e14, 4.898e14, 4.913e14, 3.984e14, 2.464e14, 1.078e14])*1e-14

            #y = (x[:,0]*x[:,0]+3.5*x[:,1])
            #x_true = x
            #y_true = y
            y_true = y_true.reshape(-1, 1)
        else:
            x_true = None
            y_true = None
        # then broadcast to all ranks
        x_true = MPI.COMM_WORLD.bcast(x_true, root=0)
        y_true = MPI.COMM_WORLD.bcast(y_true, root=0)
    ##################################################
    ##################################################
    ##################################################

    # STEP 2
    # Create solution manipulators. The solution manipulator is what creates
    # the representations of the functions as the acyclic graph


    ####### SOLUTION MANIPULATOR #######
    # nvars - number of independent variables
    # ag_size - length of the command stack
    # nloads - number of load operation which are required at the start of stack - Default 1
    # float_lim - (0, max) of floats which are generated - Default 10.0
    # terminal_prob: probability that a new node will be a terminal - Default 0.1
    sol_manip = AGraphCpp.AGraphCppManipulator(1, 64, nloads=2)

    ####### OPERATIONS #######
    sol_manip.add_node_type(2)  # +
    sol_manip.add_node_type(3)  # -
    sol_manip.add_node_type(4)  # *
    sol_manip.add_node_type(5)  # /
    sol_manip.add_node_type(6)  # sin
    sol_manip.add_node_type(7)  # cos
    sol_manip.add_node_type(8)  # exp
    sol_manip.add_node_type(9)  # log
    sol_manip.add_node_type(10)  # pow
    sol_manip.add_node_type(11)  # abs
    sol_manip.add_node_type(12)  # sqrt

    ####### PREDICTION MANIPULATOR #######
    pred_manip = fpm(16, data_size)

    # STEP 3
    # Create the training data from your x and y data, and create the fitness metric
    # For this example, we are using explicit (standard)

    ####### TRAINING DATA #######
    training_data = ExplicitTrainingData(x_true, y_true)
    
    ####### FITNESS METRIC #######
    explicit_regressor = StandardRegression()

    # STEP 4
    # Create the island manager, this will run the steps on the population, and
    # determine when to stop running
    
    ####### ISLAND MANAGER #######
    islmngr = ParallelIslandManager(#restart_file='test.p',
        solution_training_data=training_data,
        solution_manipulator=sol_manip,
        predictor_manipulator=pred_manip,
        solution_pop_size=64,
        fitness_metric=explicit_regressor)

    ####### RUN ISLAND MANAGER #######

    ##################################################
    ##################### SINGLE #####################
    ##################################################    
    # max_steps - Max amount to go if no convergence happens
    # epsilon - error which defines convergence
    # min_steps - minimum number of steps required - Default 0
    # step_increment - number of steps between convergence checks / migration - Default 1000
    # make_plots - bool whether or not to produce plots - Default True
    # checkpoint_file - base file name for checkpoint files
    if not mpi:
        islmngr.run_islands(max_steps, epsilon, min_steps=500,
                            step_increment=500)
    ##################################################
    ##################################################
    ##################################################

    ##################################################
    #################### PARALLEL ####################
    ##################################################
    # when_update - how often rank 0 gets updated on ages - Default 10
    # non_block - bool to determine to run nonblocking - Default True
    if mpi:
        islmngr.run_islands(max_steps, epsilon, min_steps=500,
                            step_increment=500, when_update=50)
    ##################################################
    ##################################################
    ##################################################

if __name__ == "__main__":

    MAX_STEPS = 30000
    CONVERGENCE_EPSILON = 0.001
    DATA_SIZE = 500
    
    #####  SINGLE   #####
    #mpi = False

    ##### PARALLEL #####
    mpi = True

    main(MAX_STEPS, CONVERGENCE_EPSILON, DATA_SIZE, mpi)