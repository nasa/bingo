"""
testing... testing...
"""
import numpy as np
import math

from bingo import AGraphCpp
from bingo.FitnessPredictor import FPManipulator as fpm
from bingo.IslandManager import SerialIslandManager
from bingo.Utils import calculate_partials, savitzky_golay_gram, snake_walk
from bingo.FitnessMetric import ImplicitRegression, StandardRegression
from bingo.TrainingData import ExplicitTrainingData, ImplicitTrainingData
from bingocpp.build import bingocpp

import logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

N_ISLANDS = 4
MAX_STEPS = 1000
N_STEPS = 100
STANDARD_EPSILON = 1e-8

def test_cpp_agcpp_explicit_add():
    """test add primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = (x_true[:, 0] + x_true[:, 1])

    # test solution
    operator = 2
    params = (0, 1)
    compare_cpp_agcpp_explicit(x_true, y, operator, params)

def test_cpp_agcpp_explicit_sub():
    """test add primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = (x_true[:, 0] + x_true[:, 1])

    # test solution
    operator = 3
    params = (0, 1)
    compare_cpp_agcpp_explicit(x_true, y, operator, params)

def test_cpp_agcpp_explicit_mul():
    """test add primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = (x_true[:, 0] + x_true[:, 1])

    # test solution
    operator = 4
    params = (0, 1)
    compare_cpp_agcpp_explicit(x_true, y, operator, params)

def test_cpp_agcpp_explicit_div():
    """test add primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = (x_true[:, 0] + x_true[:, 1])

    # test solution
    operator = 5
    params = (0, 1)
    compare_cpp_agcpp_explicit(x_true, y, operator, params)

def test_cpp_agcpp_explicit_sin():
    """test add primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = np.sin(x_true[:, 0])

    # test solution
    operator = 6
    params = (0, 0)
    compare_cpp_agcpp_explicit(x_true, y, operator, params)

def test_cpp_agcpp_explicit_cos():
    """test add primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = np.cos(x_true[:, 0])

    # test solution
    operator = 7
    params = (0, 0)
    compare_cpp_agcpp_explicit(x_true, y, operator, params)

def test_cpp_agcpp_explicit_exp():
    """test add primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = np.exp(x_true[:, 0])

    # test solution
    operator = 8
    params = (0, 0)
    compare_cpp_agcpp_explicit(x_true, y, operator, params)

def test_cpp_agcpp_explicit_log():
    """test add primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = np.log(x_true[:, 0])

    # test solution
    operator = 9
    params = (0, 0)
    compare_cpp_agcpp_explicit(x_true, y, operator, params)
    
def test_cpp_agcpp_explicit_pow():
    """test add primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = np.power(x_true[:, 0], x_true[:, 1])

    # test solution
    operator = 10
    params = (0, 1)
    compare_cpp_agcpp_explicit(x_true, y, operator, params)

def test_cpp_agcpp_explicit_abs():
    """test add primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = np.abs(x_true[:, 0])

    # test solution
    operator = 11
    params = (0, 0)
    compare_cpp_agcpp_explicit(x_true, y, operator, params)
    
def test_cpp_agcpp_explicit_sqrt():
    """test add primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = np.sqrt(x_true[:, 0])

    # test solution
    operator = 12
    params = (0, 0)
    compare_cpp_agcpp_explicit(x_true, y, operator, params)

def compare_cpp_agcpp_explicit(X, Y, operator, params):
    """does the comparison"""
    Y = Y.reshape([-1, 1])
    # make solution manipulator
    sol_manip = bingocpp.AcyclicGraphManipulator(X.shape[1], 16, nloads=2)
    sol_manip.add_node_type(2)
    sol_manip.add_node_type(3)
    sol_manip.add_node_type(4)
    sol_manip.add_node_type(5)
    sol_manip.add_node_type(6)
    sol_manip.add_node_type(7)
    sol_manip.add_node_type(8)
    sol_manip.add_node_type(9)
    sol_manip.add_node_type(10)
    sol_manip.add_node_type(11)
    sol_manip.add_node_type(12)

    # make true equation
    equ = sol_manip.generate()
    stack = np.copy(equ.stack)
    stack[0] = (0, 0, 0)
    stack[1] = (0, 1, 1)
    stack[-1] = (operator, params[0], params[1])
    equ.stack = np.copy(stack)

    # make predictor manipulator
    pred_manip = fpm(32, X.shape[0])

    # make training data
    training_data = bingocpp.ExplicitTrainingData(X, Y)

    # make fitness_metric
    explicit_regressor = bingocpp.StandardRegression()

    # make and run island manager
    islmngr = SerialIslandManager(N_ISLANDS,
                                  solution_training_data=training_data,
                                  solution_manipulator=sol_manip,
                                  predictor_manipulator=pred_manip,
                                  fitness_metric=explicit_regressor)
    epsilon = 1.05 * islmngr.isles[0].solution_fitness_true(equ) + 1.0e-10
    print("epsilon: ", epsilon)
    converged =  islmngr.run_islands(MAX_STEPS, epsilon, step_increment=N_STEPS, 
                                     make_plots=False)
    
    if not converged:
        # try to run again if it fails
        islmngr = SerialIslandManager(N_ISLANDS,
                                  solution_training_data=training_data,
                                  solution_manipulator=sol_manip,
                                  predictor_manipulator=pred_manip,
                                  fitness_metric=explicit_regressor)
        epsilon = 1.05 * islmngr.isles[0].solution_fitness_true(equ) + 1.0e-10
        print("epsilon: ", epsilon)
        converged =  islmngr.run_islands(MAX_STEPS, epsilon, 
                                         step_increment=N_STEPS, 
                                         make_plots=False)
                                         
    assert converged

def test_cpp_agcpp_implicit_add():
    """test add primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = (x_true[:, 0] + x_true[:, 1])

    # test solution
    operator = 2
    params = (0, 1)
    compare_cpp_agcpp_implicit(x_true, y, operator, params)

def test_cpp_agcpp_implicit_sub():
    """test add primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = (x_true[:, 0] + x_true[:, 1])

    # test solution
    operator = 3
    params = (0, 1)
    compare_cpp_agcpp_implicit(x_true, y, operator, params)

def test_cpp_agcpp_implicit_mul():
    """test add primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = (x_true[:, 0] + x_true[:, 1])

    # test solution
    operator = 4
    params = (0, 1)
    compare_cpp_agcpp_implicit(x_true, y, operator, params)

def test_cpp_agcpp_implicit_div():
    """test add primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = (x_true[:, 0] + x_true[:, 1])

    # test solution
    operator = 5
    params = (0, 1)
    compare_cpp_agcpp_implicit(x_true, y, operator, params)

def test_cpp_agcpp_implicit_sin():
    """test add primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = np.sin(x_true[:, 0])

    # test solution
    operator = 6
    params = (0, 0)
    compare_cpp_agcpp_implicit(x_true, y, operator, params)

def test_cpp_agcpp_implicit_cos():
    """test add primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = np.cos(x_true[:, 0])

    # test solution
    operator = 7
    params = (0, 0)
    compare_cpp_agcpp_implicit(x_true, y, operator, params)

def test_cpp_agcpp_implicit_exp():
    """test add primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = np.exp(x_true[:, 0])

    # test solution
    operator = 8
    params = (0, 0)
    compare_cpp_agcpp_implicit(x_true, y, operator, params)

def test_cpp_agcpp_implicit_log():
    """test add primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = np.log(x_true[:, 0])

    # test solution
    operator = 9
    params = (0, 0)
    compare_cpp_agcpp_implicit(x_true, y, operator, params)
    
def test_cpp_agcpp_implicit_pow():
    """test add primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = np.power(x_true[:, 0], x_true[:, 1])

    # test solution
    operator = 10
    params = (0, 1)
    compare_cpp_agcpp_implicit(x_true, y, operator, params)

def test_cpp_agcpp_implicit_abs():
    """test add primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = np.abs(x_true[:, 0])

    # test solution
    operator = 11
    params = (0, 0)
    compare_cpp_agcpp_implicit(x_true, y, operator, params)
    
def test_cpp_agcpp_implicit_sqrt():
    """test add primitive in sym reg"""
    # get independent vars
    x_true = snake_walk()

    # make solutions
    y = np.sqrt(x_true[:, 0])

    # test solution
    operator = 12
    params = (0, 0)
    compare_cpp_agcpp_implicit(x_true, y, operator, params)

def compare_cpp_agcpp_implicit(X, Y, operator, params):
    """does the comparison"""
    X = np.hstack((X, Y.reshape([-1, 1])))
    Y = None
    # make solution manipulator
    sol_manip = bingocpp.AcyclicGraphManipulator(X.shape[1], 16, nloads=2)
    sol_manip.add_node_type(2)
    sol_manip.add_node_type(3)
    sol_manip.add_node_type(4)
    sol_manip.add_node_type(5)
    sol_manip.add_node_type(6)
    sol_manip.add_node_type(7)
    sol_manip.add_node_type(8)
    sol_manip.add_node_type(9)
    sol_manip.add_node_type(10)
    sol_manip.add_node_type(11)
    sol_manip.add_node_type(12)

    # make true equation
    equ = sol_manip.generate()
    stack = np.copy(equ.stack)
    stack[0] = (0, 0, 0)
    stack[1] = (0, 1, 1)
    stack[2] = (0, 2, 2)
    stack[3] = (operator, params[0], params[1])
    stack[-1] = (3, 3, 2)
    equ.stack = np.copy(stack)
    sol_manip.simplify_stack(equ)

    print(stack)
    print("equstack\n", equ.stack)

    # make predictor manipulator
    pred_manip = fpm(32, X.shape[0])

    # make training data
    training_data = bingocpp.ImplicitTrainingData(X)

    # make fitness_metric
    explicit_regressor = bingocpp.ImplicitRegression()

    # make and run island manager
    islmngr = SerialIslandManager(N_ISLANDS,
                                  solution_training_data=training_data,
                                  solution_manipulator=sol_manip,
                                  predictor_manipulator=pred_manip,
                                  fitness_metric=explicit_regressor)
    epsilon = 1.05 * islmngr.isles[0].solution_fitness_true(equ) + 1.0e-10
    print("EPSILON IS - ", epsilon, equ.latexstring())
    converged =  islmngr.run_islands(MAX_STEPS, epsilon, step_increment=N_STEPS, 
                                     make_plots=False)
    
    if not converged:
        # try to run again if it fails
        islmngr = SerialIslandManager(N_ISLANDS,
                                      solution_training_data=training_data,
                                      solution_manipulator=sol_manip,
                                      predictor_manipulator=pred_manip,
                                      fitness_metric=explicit_regressor)
        epsilon = 1.05 * islmngr.isles[0].solution_fitness_true(equ) + 1.0e-10
        print("EPSILON IS - ", epsilon, equ.latexstring())
        converged =  islmngr.run_islands(MAX_STEPS, epsilon, 
                                         step_increment=N_STEPS, 
                                         make_plots=False)
