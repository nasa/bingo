"""
testing... testing...
"""
import numpy as np
import math

from bingo.AGraph import AGraphManipulator as agm
from bingo.AGraph import AGNodes
from bingo import AGraphCpp
from bingo.FitnessPredictor import FPManipulator as fpm
from bingo.IslandManager import SerialIslandManager
from bingo.Utils import calculate_partials, savitzky_golay_gram
from bingo.FitnessMetric import ImplicitRegression, StandardRegression
from bingo.TrainingData import ExplicitTrainingData, ImplicitTrainingData
from bingocpp.build import bingocpp

import logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

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

def test_evaluate():
    print("-----test_evaluate-----")
    n_lin = int(math.pow(500, 1.0/3)) + 1
    x_1 = np.linspace(0, 5, n_lin)
    x_2 = np.linspace(0, 5, n_lin)
    x_3 = np.linspace(0, 5, n_lin)
    x = np.array(np.meshgrid(x_1, x_2, x_3)).T.reshape(-1, 3)
    x = x[np.random.choice(x.shape[0], 500, replace=False), :]
    # make solution
    y_t = (x[:,0]*x[:,0]+3.5*x[:,1])
    y = y_t.reshape(-1, 1)

    training_data = ExplicitTrainingData(x, y)
    explicit_regressor = StandardRegression()

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

    py_fit = explicit_regressor.evaluate_fitness(py_1, training_data)
    c_fit = explicit_regressor.evaluate_fitness(c_1, training_data)

    assert py_fit == c_fit