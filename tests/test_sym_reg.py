"""
test_sym_reg tests the standard symbolic regression nodes
"""

import numpy as np


from bingo.AGraph import AGraphManipulator as agm
from bingo.AGraph import AGNodes
from bingo.FitnessPredictor import FPManipulator as fpm
from bingo.IslandManager import SerialIslandManager
from bingo.Utils import snake_walk
from bingo.FitnessMetric import StandardRegression


N_ISLANDS = 2
MAX_STEPS = 1000
EPSILON = 1e-8
N_STEPS = 100


def test_sym_reg_add():
    """test add primative in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = (x_true[:, 0] + x_true[:, 1])

    # test solution
    compare_sym_reg(x_true, y)


def test_sym_reg_sub():
    """test add primative in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = (x_true[:, 0] - x_true[:, 1])

    # test solution
    compare_sym_reg(x_true, y)


def test_sym_reg_mul():
    """test add primative in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = (x_true[:, 0] * x_true[:, 1])

    # test solution
    compare_sym_reg(x_true, y)


def test_sym_reg_div():
    """test add primative in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = (x_true[:, 0] / x_true[:, 1])

    # test solution
    compare_sym_reg(x_true, y)


def test_sym_reg_cos():
    """test add primative in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = np.cos(x_true[:, 0])

    # test solution
    compare_sym_reg(x_true, y)


def test_sym_reg_sin():
    """test add primative in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = np.sin(x_true[:, 0])

    # test solution
    compare_sym_reg(x_true, y)


def test_sym_reg_exp():
    """test add primative in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = np.exp(x_true[:, 0])

    # test solution
    compare_sym_reg(x_true, y)


def test_sym_reg_log():
    """test add primative in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = np.log(x_true[:, 0])

    # test solution
    compare_sym_reg(x_true, y)


def test_sym_reg_abs():
    """test add primative in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = np.abs(x_true[:, 0])

    # test solution
    compare_sym_reg(x_true, y)


def compare_sym_reg(X, Y):
    """does the comparison"""
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
    pred_manip = fpm(32, Y.shape[0])

    # make and run island manager
    islmngr = SerialIslandManager(N_ISLANDS,
                                  data_x=X,
                                  data_y=Y,
                                  solution_manipulator=sol_manip,
                                  predictor_manipulator=pred_manip,
                                  fitness_metric=StandardRegression)
    assert islmngr.run_islands(MAX_STEPS, EPSILON, step_increment=N_STEPS)
