"""
example of regression done using the serial island manager (islands done
serially on a single process)
"""

import numpy as np

from bingo.AGraph import AGraphManipulator as agm
from bingo.AGraph import AGNodes
from bingo.FitnessPredictor import FPManipulator as fpm
from bingo.IslandManager import SerialIslandManager
from bingo.Utils import snake_walk

def main(max_steps, epsilon, data_size, data_range, n_islands):
    """main regression function"""

    # make data
    # X = np.linspace(data_range[0], data_range[1], data_size, False)
    # Y = X * X + 0.5
    # Y = 1.5*X*X - X*X*X
    # Y = np.exp(np.abs(X))*np.sin(X)
    # Y = X*X*np.exp(np.sin(X)) + X + np.sin(3.14159/4 - X*X*X)
    # X = X.reshape([-1, 1])

    # get independent vars
    X = snake_walk()
    Y = (X[:, 0] + X[:, 1]).reshape([-1, 1])
    X = np.hstack((X, Y))
    Y = None

    # make solution manipulator
    sol_manip = agm(2, 32, nloads=2)
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
    pred_manip = fpm(32, data_size)

    # make and run island manager
    islmngr = SerialIslandManager(n_islands, X, Y, sol_manip, pred_manip)
    islmngr.run_islands(max_steps, epsilon)


if __name__ == "__main__":

    MAX_STEPS = 2000
    CONVERGENCE_EPSILON = 0.01
    DATA_SIZE = 100
    DATA_RANGE = [-3, 3]
    N_ISLANDS = 2

    main(MAX_STEPS, CONVERGENCE_EPSILON, DATA_SIZE, DATA_RANGE, N_ISLANDS)
