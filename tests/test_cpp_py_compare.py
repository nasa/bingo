"""
testing... testing...
"""
import numpy as np
import math
import pytest

from bingo import AGraphCpp
from bingo.FitnessPredictor import FPManipulator as fpm
from bingo.IslandManager import SerialIslandManager
from bingo.Utils import calculate_partials, savitzky_golay_gram, snake_walk
from bingo.FitnessMetric import ImplicitRegression, StandardRegression
from bingo.TrainingData import ExplicitTrainingData, ImplicitTrainingData
from bingocpp.build import bingocpp

import logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

###################### UTILS ########################

def test_savitzky_golay():
    print("-----test_savitzky_golay-----")
    # get independent vars
    y = (7, 4, 3, 11, 2, 13, 6, 15, 10, 22, 0, 14, 18, 19)

    py = savitzky_golay_gram(y, 7, 3, 1)
    cpp = bingocpp.savitzky_golay(y, 7, 3, 1)

    assert py.all() == cpp.all()

def test_calculate_partials():
    print("-----test_calculate_partials-----")
    X = np.array([[4, 0], [3, 1], [2, 2], [1, 3], [9, 3], [-3, 8], [-4, 9], [10, 3]])

    x_all, time_deriv_all, _ = calculate_partials(X)
    c_x, c_time = bingocpp.calculate_partials(X)

    assert x_all.all() == c_x.all()
    assert time_deriv_all.all() == c_time.all()

###################### GRAPH_MANIP ########################

def test_distance():
    print("-----test_distance-----")
    py_manip = AGraphCpp.AGraphCppManipulator(3, 64, nloads=2)
    py_manip.add_node_type(2)
    py_manip.add_node_type(3)
    py_manip.add_node_type(4)
    c_manip = bingocpp.AcyclicGraphManipulator(3, 64, nloads=2)
    c_manip.add_node_type(2)
    c_manip.add_node_type(3)
    c_manip.add_node_type(4)

    py_1 = py_manip.generate()
    py_2 = py_manip.generate()
    c_1 = c_manip.generate()
    c_2 = c_manip.generate()
    c_1.stack = np.copy(py_1.command_array)
    c_2.stack = np.copy(py_2.command_array)
    c_manip.simplify_stack(c_1)
    c_manip.simplify_stack(c_2)


    py_dist = py_manip.distance(py_1, py_2)
    c_dist = c_manip.distance(c_1, c_2)

    assert py_dist == c_dist

def test_needs_optimization():
    print("-----test_needs_optimization-----")
    
    py_manip = AGraphCpp.AGraphCppManipulator(3, 64, nloads=2)
    py_manip.add_node_type(2)
    py_manip.add_node_type(3)
    py_manip.add_node_type(4)
    c_manip = bingocpp.AcyclicGraphManipulator(3, 64, nloads=2)
    c_manip.add_node_type(2)
    c_manip.add_node_type(3)
    c_manip.add_node_type(4)

    py_1 = py_manip.generate()
    c_1 = c_manip.generate()
    py_1.command_array[0] = (0, 0, 0)
    py_1.command_array[1] = (1, -1, -1)
    py_1.command_array[-1] = (2, 1, 0)
    c_1.stack = np.copy(py_1.command_array)
    c_manip.simplify_stack(c_1)

    py_opt = py_1.needs_optimization()
    c_opt = c_1.needs_optimization()

    assert py_opt == c_opt

def test_set_constants():
    print("-----test_set_constants-----")
    
    py_manip = AGraphCpp.AGraphCppManipulator(3, 64, nloads=2)
    py_manip.add_node_type(2)
    py_manip.add_node_type(3)
    py_manip.add_node_type(4)
    c_manip = bingocpp.AcyclicGraphManipulator(3, 64, nloads=2)
    c_manip.add_node_type(2)
    c_manip.add_node_type(3)
    c_manip.add_node_type(4)

    py_1 = py_manip.generate()
    c_1 = c_manip.generate()
    c_1.stack = np.copy(py_1.command_array)
    c_manip.simplify_stack(c_1)

    constants = np.array([1, 5, 6, 7, 8, 32, 54, 68])

    py_1.set_constants(constants)
    c_1.set_constants(constants)
    
    assert py_1.constants.all() == c_1.constants.all()

def test_count_constants():
    print("-----test_count_constants-----")
    
    py_manip = AGraphCpp.AGraphCppManipulator(3, 64, nloads=2)
    py_manip.add_node_type(2)
    py_manip.add_node_type(3)
    py_manip.add_node_type(4)
    c_manip = bingocpp.AcyclicGraphManipulator(3, 64, nloads=2)
    c_manip.add_node_type(2)
    c_manip.add_node_type(3)
    c_manip.add_node_type(4)

    py_1 = py_manip.generate()
    c_1 = c_manip.generate()
    py_1.command_array[0] = (0, 0, 0)
    py_1.command_array[1] = (1, 0, 0)
    py_1.command_array[1] = (1, 1, 1)
    py_1.command_array[-1] = (2, 2, 1)
    c_1.stack = np.copy(py_1.command_array)
    c_manip.simplify_stack(c_1)

    constants = np.array([1, 5, 6, 7, 8, 32, 54, 68])

    py_1.set_constants(constants)
    c_1.set_constants(constants)

    py_con = py_1.count_constants()
    c_con = c_1.count_constants()

    assert py_con == c_con

def test_agcpp_evaluate():
    print("-----test_agcpp_evaluate-----")
    n_lin = int(math.pow(500, 1.0/3)) + 1
    x_1 = np.linspace(0, 5, n_lin)
    x_2 = np.linspace(0, 5, n_lin)
    x_3 = np.linspace(0, 5, n_lin)
    x = np.array(np.meshgrid(x_1, x_2, x_3)).T.reshape(-1, 3)
    x = x[np.random.choice(x.shape[0], 500, replace=False), :]
    
    py_manip = AGraphCpp.AGraphCppManipulator(3, 64, nloads=2)
    py_manip.add_node_type(2)
    py_manip.add_node_type(3)
    py_manip.add_node_type(4)
    c_manip = bingocpp.AcyclicGraphManipulator(3, 64, nloads=2)
    c_manip.add_node_type(2)
    c_manip.add_node_type(3)
    c_manip.add_node_type(4)

    py_1 = py_manip.generate()
    c_1 = c_manip.generate()
    py_1.command_array[0] = (0, 0, 0)
    py_1.command_array[1] = (0, 1, 1)
    py_1.command_array[2] = (1, 0, 0)
    py_1.command_array[3] = (1, 1, 1)
    py_1.command_array[4] = (2, 3, 1)
    py_1.command_array[-1] = (2, 4, 2)

    c_1.stack = np.copy(py_1.command_array)
    c_manip.simplify_stack(c_1)

    constants = np.array([1, 5, 6, 7, 8, 32, 54, 68])

    py_1.set_constants(constants)
    c_1.set_constants(constants)

    py_fit = py_1.evaluate(x)
    c_fit = c_1.evaluate(x)

    assert py_fit == pytest.approx(c_fit)

def test_agcpp_evaluate_deriv():
    print("-----test_agcpp_evaluate_deriv-----")
    n_lin = int(math.pow(500, 1.0/3)) + 1
    x_1 = np.linspace(0, 5, n_lin)
    x_2 = np.linspace(0, 5, n_lin)
    x_3 = np.linspace(0, 5, n_lin)
    x = np.array(np.meshgrid(x_1, x_2, x_3)).T.reshape(-1, 3)
    x = x[np.random.choice(x.shape[0], 500, replace=False), :]
    
    py_manip = AGraphCpp.AGraphCppManipulator(3, 64, nloads=2)
    py_manip.add_node_type(2)
    py_manip.add_node_type(3)
    py_manip.add_node_type(4)
    c_manip = bingocpp.AcyclicGraphManipulator(3, 64, nloads=2)
    c_manip.add_node_type(2)
    c_manip.add_node_type(3)
    c_manip.add_node_type(4)

    py_1 = py_manip.generate()
    c_1 = c_manip.generate()
    py_1.command_array[0] = (0, 0, 0)
    py_1.command_array[1] = (0, 1, 1)
    py_1.command_array[2] = (1, 0, 0)
    py_1.command_array[3] = (1, 1, 1)
    py_1.command_array[4] = (2, 3, 1)
    py_1.command_array[-1] = (2, 4, 2)

    c_1.stack = np.copy(py_1.command_array)
    c_manip.simplify_stack(c_1)

    constants = np.array([1, 5, 6, 7, 8, 32, 54, 68])

    py_1.set_constants(constants)
    c_1.set_constants(constants)

    py_fit = py_1.evaluate_deriv(x)
    c_fit = c_1.evaluate_deriv(x)
    py_fit_const = py_1.evaluate_with_const_deriv(x)
    c_fit_const = c_1.evaluate_with_const_deriv(x)

    assert py_fit[0].all() == pytest.approx(c_fit[0].all())
    assert py_fit[1].all() == pytest.approx(c_fit[1].all())
    assert py_fit_const[0].all() == pytest.approx(c_fit_const[0].all())
    assert py_fit_const[1].all() == pytest.approx(c_fit_const[1].all())

def test_latexstring():
    print("-----test_latexstring-----")
    py_manip = AGraphCpp.AGraphCppManipulator(3, 64, nloads=2)
    py_manip.add_node_type(2)
    py_manip.add_node_type(3)
    py_manip.add_node_type(4)
    c_manip = bingocpp.AcyclicGraphManipulator(3, 64, nloads=2)
    c_manip.add_node_type(2)
    c_manip.add_node_type(3)
    c_manip.add_node_type(4)

    py_1 = py_manip.generate()
    c_1 = c_manip.generate()
    c_1.stack = np.copy(py_1.command_array)
    c_manip.simplify_stack(c_1)

    py_latex = py_1.latexstring()
    c_latex = c_1.latexstring()

    assert py_latex == c_latex

def test_complexity():
    print("-----test_complexity-----")    
    py_manip = AGraphCpp.AGraphCppManipulator(3, 64, nloads=2)
    py_manip.add_node_type(2)
    py_manip.add_node_type(3)
    py_manip.add_node_type(4)
    c_manip = bingocpp.AcyclicGraphManipulator(3, 64, nloads=2)
    c_manip.add_node_type(2)
    c_manip.add_node_type(3)
    c_manip.add_node_type(4)

    py_1 = py_manip.generate()
    c_1 = c_manip.generate()
    c_1.stack = np.copy(py_1.command_array)
    c_manip.simplify_stack(c_1)

    py_complexity = py_1.complexity()
    c_complexity = c_1.complexity()

    assert py_complexity == c_complexity

def test_dump():
    print("-----test_dump-----")
    constants = np.array([1, 5, 6, 7, 8, 32, 54, 68])
    
    py_manip = AGraphCpp.AGraphCppManipulator(3, 64, nloads=2)
    py_manip.add_node_type(2)
    py_manip.add_node_type(3)
    py_manip.add_node_type(4)
    c_manip = bingocpp.AcyclicGraphManipulator(3, 64, nloads=2)
    c_manip.add_node_type(2)
    c_manip.add_node_type(3)
    c_manip.add_node_type(4)

    py_1 = py_manip.generate()
    py_1.constants = np.copy(constants)
    orig_stack = np.copy(py_1.command_array)
    orig_const = np.copy(py_1.constants)
    orig_age = py_1.genetic_age

    c_1 = c_manip.generate()
    c_1.stack = np.copy(orig_stack)
    c_1.constants = np.copy(orig_const)
    c_manip.simplify_stack(c_1)

    py_stack, py_con, py_age = py_manip.dump(py_1)
    c_pair, c_age = c_manip.dump(c_1)
    c_stack = c_pair[0]
    c_con = c_pair[1]

    assert py_stack.all() == c_stack.all()
    assert py_con.all() == c_con.all()
    assert py_age == c_age

def test_load():
    print("-----test_load-----")
    constants = np.array([1, 5, 6, 7, 8, 32, 54, 68])
    
    py_manip = AGraphCpp.AGraphCppManipulator(3, 64, nloads=2)
    py_manip.add_node_type(2)
    py_manip.add_node_type(3)
    py_manip.add_node_type(4)
    c_manip = bingocpp.AcyclicGraphManipulator(3, 64, nloads=2)
    c_manip.add_node_type(2)
    c_manip.add_node_type(3)
    c_manip.add_node_type(4)

    py_1 = py_manip.generate()
    orig_stack = np.copy(py_1.command_array)

    py_1 = py_manip.load([orig_stack, constants, 0])
    c_1 = c_manip.load([[orig_stack, constants], 0])

    assert py_1.command_array.all() == c_1.stack.all()
    assert py_1.constants.all() == c_1.constants.all()
    assert py_1.genetic_age == c_1.genetic_age

###################### TRAINING_DATA ########################

def test_explicit_get_item():
    print("-----test_explicit_get_item-----")
    n_lin = int(math.pow(500, 1.0/3)) + 1
    x_1 = np.linspace(0, 5, n_lin)
    x_2 = np.linspace(0, 5, n_lin)
    x_3 = np.linspace(0, 5, n_lin)
    x = np.array(np.meshgrid(x_1, x_2, x_3)).T.reshape(-1, 3)
    x = x[np.random.choice(x.shape[0], 500, replace=False), :]
    # make solution
    y_t = (x[:,0]*x[:,0]+3.5*x[:,1])
    y = y_t.reshape(-1, 1)

    py_training_data = ExplicitTrainingData(x, y)
    c_training_data = bingocpp.ExplicitTrainingData(x, y)

    items = [5, 10, 15, 18, 64, 92, 129, 186, 201, 215, 293, 355, 389]

    py_result = py_training_data.__getitem__(items)
    c_result = c_training_data.__getitem__(items)

    assert py_result.x.all() == c_result.x.all()
    assert py_result.y.all() == c_result.y.all()


def test_implicit_get_item():
    print("-----test_implicit_get_item-----")
    n_lin = int(math.pow(500, 1.0/3)) + 1
    x_1 = np.linspace(0, 5, n_lin)
    x_2 = np.linspace(0, 5, n_lin)
    x_3 = np.linspace(0, 5, n_lin)
    x = np.array(np.meshgrid(x_1, x_2, x_3)).T.reshape(-1, 3)
    x = x[np.random.choice(x.shape[0], 500, replace=False), :]
    # make solution
    y_t = (x[:,0]*x[:,0]+3.5*x[:,1])
    y = y_t.reshape(-1, 1)
    x = np.hstack((x, y))

    py_training_data = ImplicitTrainingData(x)
    c_training_data = bingocpp.ImplicitTrainingData(x)

    items = [5, 10, 15, 18, 64, 92, 129, 186, 201, 215, 293, 355, 389]

    py_result = py_training_data.__getitem__(items)
    c_result = c_training_data.__getitem__(items)

    assert py_result.x.all() == c_result.x.all()
    assert py_result.dx_dt.all() == c_result.dx_dt.all()

###################### FITNESS_METRIC ########################

def test_evaluate_explicit():
    print("-----test_evaluate_explicit-----")
    n_lin = int(math.pow(500, 1.0/3)) + 1
    x_1 = np.linspace(0, 5, n_lin)
    x_2 = np.linspace(0, 5, n_lin)
    x_3 = np.linspace(0, 5, n_lin)
    x = np.array(np.meshgrid(x_1, x_2, x_3)).T.reshape(-1, 3)
    x = x[np.random.choice(x.shape[0], 500, replace=False), :]
    # make solution
    y_t = (x[:,0]*x[:,0]+3.5*x[:,1])
    y = y_t.reshape(-1, 1)

    py_training_data = ExplicitTrainingData(x, y)
    py_explicit_regressor = StandardRegression()
    c_explicit_regressor = bingocpp.StandardRegression()
    c_training_data = bingocpp.ExplicitTrainingData(x, y)

    py_manip = AGraphCpp.AGraphCppManipulator(3, 64, nloads=2)
    py_manip.add_node_type(2)
    py_manip.add_node_type(3)
    py_manip.add_node_type(4)
    c_manip = bingocpp.AcyclicGraphManipulator(3, 64, nloads=2)
    c_manip.add_node_type(2)
    c_manip.add_node_type(3)
    c_manip.add_node_type(4)

    py_1 = py_manip.generate()
    c_1 = c_manip.generate()
    c_1.stack = np.copy(py_1.command_array)
    c_manip.simplify_stack(c_1)

    py_fit = py_explicit_regressor.evaluate_fitness(py_1, py_training_data)
    c_fit = c_explicit_regressor.evaluate_fitness(c_1, c_training_data)

    assert py_fit == pytest.approx(c_fit)

def test_evaluate_implicit():
    print("-----test_evaluate_implicit-----")
    x_t = snake_walk()
    y = (x_t[:, 0] + x_t[:, 1])
    x = np.hstack((x_t, y.reshape([-1, 1])))
                     
    py_training_data = ImplicitTrainingData(x)
    py_implicit_regressor = ImplicitRegression()
    c_training_data = bingocpp.ImplicitTrainingData(x)
    c_implicit_regressor = bingocpp.ImplicitRegression()

    py_manip = AGraphCpp.AGraphCppManipulator(3, 64, nloads=2)
    py_manip.add_node_type(2)
    py_manip.add_node_type(3)
    py_manip.add_node_type(4)
    c_manip = bingocpp.AcyclicGraphManipulator(3, 64, nloads=2)
    c_manip.add_node_type(2)
    c_manip.add_node_type(3)
    c_manip.add_node_type(4)

    temp = np.array([[ 1,  0,  0],
                     [ 0,  0,  0],
                     [ 0,  1,  1],
                     [ 3,  2,  2],
                     [ 4,  2,  3],
                     [ 4,  4,  0],
                     [ 3,  2,  0],
                     [ 3,  0,  0],
                     [ 3,  4,  2],
                     [ 3,  1,  4],
                     [ 2,  6,  0],
                     [ 3,  8,  7],
                     [ 2,  3, 11],
                     [ 3,  7,  7],
                     [ 3,  2,  7],
                     [ 2,  9,  5],
                     [ 4,  4, 14],
                     [ 3,  9,  1],
                     [ 3,  4,  5],
                     [ 0,  1,  1],
                     [ 4,  8,  4],
                     [ 1, -1, -1],
                     [ 3, 20,  5],
                     [ 2,  9, 17],
                     [ 1, -1, -1],
                     [ 4, 23,  4],
                     [ 4, 25, 19],
                     [ 2, 26, 14],
                     [ 4, 22, 10],
                     [ 0,  0,  0],
                     [ 2,  0, 17],
                     [ 2, 29, 16],
                     [ 4, 23, 14],
                     [ 4,  3, 22],
                     [ 0,  2,  2],
                     [ 2, 31, 27],
                     [ 2, 35, 28],
                     [ 2, 25, 29],
                     [ 2, 36, 28],
                     [ 3,  8, 29],
                     [ 3,  4, 24],
                     [ 2, 14,  9],
                     [ 4, 25,  9],
                     [ 0,  2,  2],
                     [ 3, 26, 10],
                     [ 3, 12,  6],
                     [ 0,  1,  1],
                     [ 4, 42, 26],
                     [ 3, 41,  6],
                     [ 3, 13,  1],
                     [ 3, 42, 36],
                     [ 3, 15, 34],
                     [ 2, 14, 23],
                     [ 2, 13, 12],
                     [ 0,  2,  2],
                     [ 2, 28, 45],
                     [ 3,  1, 12],
                     [ 3, 15, 30],
                     [ 3, 34, 38],
                     [ 3, 43, 50],
                     [ 2, 31,  9],
                     [ 2, 54, 46],
                     [ 1, -1, -1],
                     [ 4, 60, 29]])

    py_1 = py_manip.generate()
    c_1 = c_manip.generate()
    py_1.command_array = np.copy(temp)
    c_1.stack = np.copy(temp)
    c_manip.simplify_stack(c_1)

    py_fit = py_implicit_regressor.evaluate_fitness(py_1, py_training_data)
    c_fit = c_implicit_regressor.evaluate_fitness(c_1, c_training_data)

    assert py_training_data.x.all() == pytest.approx(c_training_data.x.all())
    assert py_training_data.dx_dt.all() == pytest.approx(c_training_data.dx_dt.all())

    assert py_fit == pytest.approx(c_fit)

def test_evaluate_fitness_vector_implicit():
    print("-----test_evaluate_fitness_vector_implicit-----")
    x_t = snake_walk()
    y = (x_t[:, 0] + x_t[:, 1])
    x = np.hstack((x_t, y.reshape([-1, 1])))

    py_training_data = ImplicitTrainingData(x)
    py_implicit_regressor = ImplicitRegression()
    c_training_data = bingocpp.ImplicitTrainingData(x)
    c_implicit_regressor = bingocpp.ImplicitRegression()

    py_manip = AGraphCpp.AGraphCppManipulator(3, 64, nloads=2)
    py_manip.add_node_type(2)
    py_manip.add_node_type(3)
    py_manip.add_node_type(4)
    c_manip = bingocpp.AcyclicGraphManipulator(3, 64, nloads=2)
    c_manip.add_node_type(2)
    c_manip.add_node_type(3)
    c_manip.add_node_type(4)

    temp = np.array([[ 1,  0,  0],
                     [ 0,  0,  0],
                     [ 0,  1,  1],
                     [ 3,  2,  2],
                     [ 4,  2,  3],
                     [ 4,  4,  0],
                     [ 3,  2,  0],
                     [ 3,  0,  0],
                     [ 3,  4,  2],
                     [ 3,  1,  4],
                     [ 2,  6,  0],
                     [ 3,  8,  7],
                     [ 2,  3, 11],
                     [ 3,  7,  7],
                     [ 3,  2,  7],
                     [ 2,  9,  5],
                     [ 4,  4, 14],
                     [ 3,  9,  1],
                     [ 3,  4,  5],
                     [ 0,  1,  1],
                     [ 4,  8,  4],
                     [ 1, -1, -1],
                     [ 3, 20,  5],
                     [ 2,  9, 17],
                     [ 1, -1, -1],
                     [ 4, 23,  4],
                     [ 4, 25, 19],
                     [ 2, 26, 14],
                     [ 4, 22, 10],
                     [ 0,  0,  0],
                     [ 2,  0, 17],
                     [ 2, 29, 16],
                     [ 4, 23, 14],
                     [ 4,  3, 22],
                     [ 0,  2,  2],
                     [ 2, 31, 27],
                     [ 2, 35, 28],
                     [ 2, 25, 29],
                     [ 2, 36, 28],
                     [ 3,  8, 29],
                     [ 3,  4, 24],
                     [ 2, 14,  9],
                     [ 4, 25,  9],
                     [ 0,  2,  2],
                     [ 3, 26, 10],
                     [ 3, 12,  6],
                     [ 0,  1,  1],
                     [ 4, 42, 26],
                     [ 3, 41,  6],
                     [ 3, 13,  1],
                     [ 3, 42, 36],
                     [ 3, 15, 34],
                     [ 2, 14, 23],
                     [ 2, 13, 12],
                     [ 0,  2,  2],
                     [ 2, 28, 45],
                     [ 3,  1, 12],
                     [ 3, 15, 30],
                     [ 3, 34, 38],
                     [ 3, 43, 50],
                     [ 2, 31,  9],
                     [ 2, 54, 46],
                     [ 1, -1, -1],
                     [ 4, 60, 29]])

    py_1 = py_manip.generate()
    c_1 = c_manip.generate()
    py_1.command_array = np.copy(temp)
    c_1.stack = np.copy(temp)
    c_manip.simplify_stack(c_1)

    assert py_training_data.x.all() == pytest.approx(c_training_data.x.all())
    assert py_training_data.dx_dt.all() == pytest.approx(c_training_data.dx_dt.all())

    py_fit = py_implicit_regressor.evaluate_fitness(py_1, py_training_data)
    py_fit = py_implicit_regressor.evaluate_fitness_vector(py_1, py_training_data)
    
    c_fit = c_implicit_regressor.evaluate_fitness(c_1, c_training_data)
    c_fit = c_implicit_regressor.evaluate_fitness_vector(c_1, c_training_data)

    assert py_fit.all() == pytest.approx(c_fit.all())