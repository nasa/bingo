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
from bingo.TrainingData import ImplicitTrainingData

N_ISLANDS = 2
MAX_STEPS = 1000
N_STEPS = 100


def test_ag_implicit_add():
    """test add primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = (x_true[:, 0] + x_true[:, 1])

    # test solution
    operator = AGNodes.Add
    params = (0, 1)
    compare_ag_implicit(x_true, y, operator, params)


def test_ag_implicit_sub():
    """test subtract primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = (x_true[:, 0] - x_true[:, 1])

    # test solution
    operator = AGNodes.Subtract
    params = (0, 1)
    compare_ag_implicit(x_true, y, operator, params)


def test_ag_implicit_mul():
    """test multiply primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = (x_true[:, 0] * x_true[:, 1])

    # test solution
    operator = AGNodes.Multiply
    params = (0, 1)
    compare_ag_implicit(x_true, y, operator, params)


def test_ag_implicit_div():
    """test divide primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = (x_true[:, 0] / x_true[:, 1])

    # test solution
    operator = AGNodes.Divide
    params = (0, 1)
    compare_ag_implicit(x_true, y, operator, params)


def test_ag_implicit_cos():
    """test cosine primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = np.cos(x_true[:, 0])

    # test solution
    operator = AGNodes.Cos
    params = (0,)
    compare_ag_implicit(x_true, y, operator, params)


def test_ag_implicit_sin():
    """test sine primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = np.sin(x_true[:, 0])

    # test solution
    operator = AGNodes.Sin
    params = (0,)
    compare_ag_implicit(x_true, y, operator, params)


def test_ag_implicit_exp():
    """test exponential primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = np.exp(x_true[:, 0])

    # test solution
    operator = AGNodes.Exp
    params = (0,)
    compare_ag_implicit(x_true, y, operator, params)


def test_ag_implicit_log():
    """test logarithm primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = np.log(x_true[:, 0])

    # test solution
    operator = AGNodes.Log
    params = (0,)
    compare_ag_implicit(x_true, y, operator, params)


def test_ag_implicit_abs():
    """test absolute value primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = np.abs(x_true[:, 0])

    # test solution
    operator = AGNodes.Abs
    params = (0,)
    compare_ag_implicit(x_true, y, operator, params)


def test_ag_implicit_pow():
    """test absolute value primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = np.power(x_true[:, 0], x_true[:,1])

    # test solution
    operator = AGNodes.Pow
    params = (0, 1)
    compare_ag_implicit(x_true, y, operator, params)


def compare_ag_implicit(X, Y, operator, params):
    """does const symbolic regression and tests convergence"""
    # convert to single array
    X = np.hstack((X, Y.reshape([-1, 1])))

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
    sol_manip.add_node_type(AGNodes.Pow)

    # make true equation
    equ = sol_manip.generate()
    equ.command_list[0] = (AGNodes.LoadData, (0,))
    equ.command_list[1] = (AGNodes.LoadData, (1,))
    equ.command_list[2] = (AGNodes.LoadData, (2,))
    equ.command_list[3] = (operator, params)
    equ.command_list[-1] = (AGNodes.Subtract, (3, 2))

    # make predictor manipulator
    pred_manip = fpm(32, X.shape[0])

    # make training data
    training_data = ImplicitTrainingData(X)

    # make fitness metric
    implicit_regressor = ImplicitRegression()

    # make and run island manager
    islmngr = SerialIslandManager(N_ISLANDS,
                                  solution_training_data=training_data,
                                  solution_manipulator=sol_manip,
                                  predictor_manipulator=pred_manip,
                                  fitness_metric=implicit_regressor)
    epsilon = 1.05 * islmngr.isles[0].solution_fitness_true(equ) + 1.0e-10
    assert islmngr.run_islands(MAX_STEPS, epsilon, step_increment=N_STEPS, 
                               make_plots=False)
