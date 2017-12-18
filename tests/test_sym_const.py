"""
test_sym_const tests the standard symbolic nodes in const regression
"""

import numpy as np

from bingo.AGraph import AGraphManipulator as agm
from bingo.AGraph import AGNodes
from bingo.FitnessPredictor import FPManipulator as fpm
from bingo.IslandManager import SerialIslandManager
from bingo.Utils import snake_walk
from bingo.FitnessMetric import ImplicitRegression

N_ISLANDS = 2
MAX_STEPS = 1000
N_STEPS = 100



def test_sym_const_add():
    """test add primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = (x_true[:, 0] + x_true[:, 1])

    # test solution
    epsilon = 1e-8
    compare_sym_const(x_true, y, epsilon)


def test_sym_const_sub():
    """test subtract primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = (x_true[:, 0] - x_true[:, 1])

    # test solution
    epsilon = 1e-8
    compare_sym_const(x_true, y, epsilon)


def test_sym_const_mul():
    """test multiply primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = (x_true[:, 0] * x_true[:, 1])

    # test solution
    epsilon = 7e-4
    compare_sym_const(x_true, y, epsilon)


def test_sym_const_div():
    """test divide primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = (x_true[:, 0] / x_true[:, 1])

    # test solution
    epsilon = 6e-4
    compare_sym_const(x_true, y, epsilon)


def test_sym_const_cos():
    """test cosine primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = np.cos(x_true[:, 0])

    # test solution
    epsilon = 3e-3
    compare_sym_const(x_true, y, epsilon)


def test_sym_const_sin():
    """test sine primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = np.sin(x_true[:, 0])

    # test solution
    epsilon = 2.3e-3
    compare_sym_const(x_true, y, epsilon)


def test_sym_const_exp():
    """test exponential primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = np.exp(x_true[:, 0])

    # test solution
    epsilon = 6e-4
    compare_sym_const(x_true, y, epsilon)


def test_sym_const_log():
    """test logarithm primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = np.log(x_true[:, 0])

    # test solution
    epsilon = 2.2e-3
    compare_sym_const(x_true, y, epsilon)


def test_sym_const_abs():
    """test absolute value primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = np.abs(x_true[:, 0])

    # test solution
    epsilon = 1e-8
    compare_sym_const(x_true, y, epsilon)


def compare_sym_const(X, Y, epsilon):
    """does const symbolic regression and tests convergence"""
    # convert to single array
    X = np.hstack((X, Y.reshape([-1, 1])))
    Y = None

    # make solution manipulator
    sol_manip = agm(X.shape[1], 16, nloads=2)
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
    pred_manip = fpm(32, X.shape[0])

    # make and run island manager
    islmngr = SerialIslandManager(N_ISLANDS,
                                  data_x=X,
                                  data_y=Y,
                                  solution_manipulator=sol_manip,
                                  predictor_manipulator=pred_manip,
                                  fitness_metric=ImplicitRegression)
    assert islmngr.run_islands(MAX_STEPS, epsilon, step_increment=N_STEPS)
