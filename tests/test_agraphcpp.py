"""
test_sym_const tests the standard symbolic nodes in const regression
"""

import numpy as np

from bingo.AGraphCpp import AGraphCppManipulator as agm
from bingo.FitnessPredictor import FPManipulator as fpm
from bingo.IslandManager import SerialIslandManager
from bingo.Utils import snake_walk
from bingo.FitnessMetric import ImplicitRegression, StandardRegression

N_ISLANDS = 2
MAX_STEPS = 1000
N_STEPS = 100
STANDARD_EPSILON = 1e-8


def test_agcpp_implicit_add():
    """test add primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = (x_true[:, 0] + x_true[:, 1])

    # test solution
    operator = 2
    params = (0, 1)
    compare_agcpp_implicit(x_true, y, operator, params)


def test_agcpp_implicit_sub():
    """test subtract primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = (x_true[:, 0] - x_true[:, 1])

    # test solution
    operator = 3
    params = (0, 1)
    compare_agcpp_implicit(x_true, y, operator, params)


def test_agcpp_implicit_mul():
    """test multiply primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = (x_true[:, 0] * x_true[:, 1])

    # test solution
    operator = 4
    params = (0, 1)
    compare_agcpp_implicit(x_true, y, operator, params)


def test_agcpp_implicit_div():
    """test divide primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = (x_true[:, 0] / x_true[:, 1])

    # test solution
    operator = 5
    params = (0, 1)
    compare_agcpp_implicit(x_true, y, operator, params)


def compare_agcpp_implicit(X, Y, operator, params):
    """does const symbolic regression and tests convergence"""
    # convert to single array
    X = np.hstack((X, Y.reshape([-1, 1])))
    Y = None

    # make solution manipulator
    sol_manip = agm(X.shape[1], 16, nloads=2)
    sol_manip.add_node_type(2)
    sol_manip.add_node_type(3)
    sol_manip.add_node_type(4)
    sol_manip.add_node_type(5)

    # make true equation
    equ = sol_manip.generate()
    equ.command_list[0] = (0, (0,))
    equ.command_list[1] = (0, (1,))
    equ.command_list[2] = (0, (2,))
    equ.command_list[3] = (operator, params)
    equ.command_list[-1] = (3, (3, 2))

    # make predictor manipulator
    pred_manip = fpm(32, X.shape[0])

    # make and run island manager
    islmngr = SerialIslandManager(N_ISLANDS,
                                  data_x=X,
                                  data_y=Y,
                                  solution_manipulator=sol_manip,
                                  predictor_manipulator=pred_manip,
                                  fitness_metric=ImplicitRegression)
    epsilon = 1.05 * islmngr.isles[0].solution_fitness_true(equ) + 1.0e-10
    assert islmngr.run_islands(MAX_STEPS, epsilon, step_increment=N_STEPS)


def test_agcpp_explicit_add():
    """test add primative in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = (x_true[:, 0] + x_true[:, 1])

    # test solution
    operator = 2
    params = (0, 1)
    compare_agcpp_explicit(x_true, y, operator, params)


def test_agcpp_explicit_sub():
    """test add primative in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = (x_true[:, 0] - x_true[:, 1])

    # test solution
    operator = 3
    params = (0, 1)
    compare_agcpp_explicit(x_true, y, operator, params)


def test_agcpp_explicit_mul():
    """test add primative in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = (x_true[:, 0] * x_true[:, 1])

    # test solution
    operator = 4
    params = (0, 1)
    compare_agcpp_explicit(x_true, y, operator, params)


def test_agcpp_explicit_div():
    """test add primative in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = (x_true[:, 0] / x_true[:, 1])

    # test solution
    operator = 5
    params = (0, 1)
    compare_agcpp_explicit(x_true, y, operator, params)


def compare_agcpp_explicit(X, Y, operator, params):
    """does the comparison"""
    Y = Y.reshape([-1, 1])
    # make solution manipulator
    sol_manip = agm(X.shape[1], 16, nloads=2)
    sol_manip.add_node_type(2)
    sol_manip.add_node_type(3)
    sol_manip.add_node_type(4)
    sol_manip.add_node_type(5)

    # make true equation
    equ = sol_manip.generate()
    equ.command_list[0] = (0, (0,))
    equ.command_list[1] = (0, (1,))
    equ.command_list[-1] = (operator, params)

    # make predictor manipulator
    pred_manip = fpm(32, X.shape[0])

    # make and run island manager
    islmngr = SerialIslandManager(N_ISLANDS,
                                  data_x=X,
                                  data_y=Y,
                                  solution_manipulator=sol_manip,
                                  predictor_manipulator=pred_manip,
                                  fitness_metric=StandardRegression)
    epsilon = 1.05 * islmngr.isles[0].solution_fitness_true(equ) + 1.0e-10
    assert islmngr.run_islands(MAX_STEPS, epsilon, step_increment=N_STEPS)
