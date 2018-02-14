"""
test_sym_reg tests the standard symbolic regression nodes
"""

import numpy as np
import time


from bingo.AGraph import AGraphManipulator as agm
from bingo.AGraphCpp import AGraphCppManipulator as agcm
from bingo.AGraph import AGNodes
from bingo.FitnessPredictor import FPManipulator as fpm
from bingo.IslandManager import SerialIslandManager
from bingo.Utils import snake_walk, calculate_partials
from bingo.FitnessMetric import StandardRegression, ImplicitRegression


N_ISLANDS = 2
MAX_STEPS = 3000
EPSILON = 1.0e-2
N_STEPS = 100


def test_const_opt_agraph_explicit():
    """test optimization code"""
    print("-----test_const_opt_agraph_explicit-----")
    # get independent vars
    x_true = snake_walk()

    # make solutions
    consts = np.random.rand(6)*90+10
    print("CONSTANTS: ", consts)
    y = consts[0] + consts[1] * x_true[:, 0] + consts[2] * x_true[:, 1] + \
        consts[3] * x_true[:, 0] * x_true[:, 0] + \
        consts[4] * x_true[:, 1] * x_true[:, 1] + \
        consts[5] * x_true[:, 0] * x_true[:, 1]

    # create manipulator to set things up
    sol_manip = agm(x_true.shape[1], 21, nloads=2)
    sol_manip.add_node_type(AGNodes.Add)
    sol_manip.add_node_type(AGNodes.Multiply)

    # create gene with proper functional form
    sol = sol_manip.generate()
    sol.command_list[0] = (AGNodes.Load_Const, (None,))
    sol.command_list[1] = (AGNodes.Load_Const, (None,))
    sol.command_list[2] = (AGNodes.Load_Const, (None,))
    sol.command_list[3] = (AGNodes.Load_Const, (None,))
    sol.command_list[4] = (AGNodes.Load_Const, (None,))
    sol.command_list[5] = (AGNodes.Load_Const, (None,))

    sol.command_list[6] = (AGNodes.Load_Data, (0,))
    sol.command_list[7] = (AGNodes.Load_Data, (1,))
    sol.command_list[8] = (AGNodes.Multiply, (6, 6))
    sol.command_list[9] = (AGNodes.Multiply, (7, 7))
    sol.command_list[10] = (AGNodes.Multiply, (6, 7))

    sol.command_list[11] = (AGNodes.Multiply, (1, 6))
    sol.command_list[12] = (AGNodes.Multiply, (2, 7))
    sol.command_list[13] = (AGNodes.Multiply, (3, 8))
    sol.command_list[14] = (AGNodes.Multiply, (4, 9))
    sol.command_list[15] = (AGNodes.Multiply, (5, 10))

    sol.command_list[16] = (AGNodes.Add, (0, 11))
    sol.command_list[17] = (AGNodes.Add, (16, 12))
    sol.command_list[18] = (AGNodes.Add, (17, 13))
    sol.command_list[19] = (AGNodes.Add, (18, 14))
    sol.command_list[20] = (AGNodes.Add, (19, 15))
    # print(sol.latexstring())

    # fit the constants
    sol.evaluate(x_true, StandardRegression, x=x_true, y=y)

    # make sure constants are close
    c_fit = sol.constants
    print("FITTED:    ", c_fit)
    for tru, fit in zip(consts, c_fit):
        assert np.abs(tru - fit) < 1e-8


def test_const_opt_agraph_implicit():
    """test optimization code"""
    print("-----test_const_opt_agraph_implicit-----")
    # get independent vars
    x_true = snake_walk()

    # make solutions
    consts = np.random.rand(6)*90+10
    print("CONSTANTS: ", consts[1:])
    y = consts[0] + consts[1] * x_true[:, 0] + consts[2] * x_true[:, 1] + \
        consts[3] * x_true[:, 0] * x_true[:, 0] + \
        consts[4] * x_true[:, 1] * x_true[:, 1] + \
        consts[5] * x_true[:, 0] * x_true[:, 1]
    x_true = np.hstack((x_true, y.reshape((-1, 1))))
    x_true, dx_dt, _ = calculate_partials(x_true)

    # create manipulator to set things up
    sol_manip = agm(x_true.shape[1], 23, nloads=2)
    sol_manip.add_node_type(AGNodes.Add)
    sol_manip.add_node_type(AGNodes.Subtract)
    sol_manip.add_node_type(AGNodes.Multiply)

    # create gene with proper functional form
    sol = sol_manip.generate()
    sol.command_list[0] = (AGNodes.Load_Const, (None,))
    sol.command_list[1] = (AGNodes.Load_Const, (None,))
    sol.command_list[2] = (AGNodes.Load_Const, (None,))
    sol.command_list[3] = (AGNodes.Load_Const, (None,))
    sol.command_list[4] = (AGNodes.Load_Const, (None,))
    sol.command_list[5] = (AGNodes.Load_Const, (None,))

    sol.command_list[6] = (AGNodes.Load_Data, (0,))
    sol.command_list[7] = (AGNodes.Load_Data, (1,))
    sol.command_list[8] = (AGNodes.Multiply, (6, 6))
    sol.command_list[9] = (AGNodes.Multiply, (7, 7))
    sol.command_list[10] = (AGNodes.Multiply, (6, 7))

    sol.command_list[11] = (AGNodes.Multiply, (1, 6))
    sol.command_list[12] = (AGNodes.Multiply, (2, 7))
    sol.command_list[13] = (AGNodes.Multiply, (3, 8))
    sol.command_list[14] = (AGNodes.Multiply, (4, 9))
    sol.command_list[15] = (AGNodes.Multiply, (5, 10))

    sol.command_list[16] = (AGNodes.Add, (0, 11))
    sol.command_list[17] = (AGNodes.Add, (16, 12))
    sol.command_list[18] = (AGNodes.Add, (17, 13))
    sol.command_list[19] = (AGNodes.Add, (18, 14))
    sol.command_list[20] = (AGNodes.Add, (19, 15))
    sol.command_list[21] = (AGNodes.Load_Data, (2,))
    sol.command_list[22] = (AGNodes.Subtract, (20, 21))
    # print(sol.latexstring())

    # fit the constants
    t0 = time.time()
    _, df_dx = sol.evaluate_deriv(x_true, ImplicitRegression,
                                  x=x_true, dx_dt=dx_dt, required_params=3)
    t1 = time.time()
    print("fit time: ", t1-t0, "seconds")

    # make sure constants are close
    c_fit = sol.constants
    print("average accuracy: ",
          np.mean(np.abs(np.array(c_fit[1:]) - np.array(consts[1:]))/
                  np.array(consts[1:])))
    print("FITTED:    ", c_fit[1:])
    for tru, fit in zip(consts[1:], c_fit[1:]):
        print(np.abs(tru - fit) / tru)
        assert np.abs(tru - fit)/tru < 0.1


def test_const_opt_agraph_explicit_evo():
    """test add primative in sym reg"""
    print("-----test_const_opt_agraph_explicit_evo-----")
    # get independent vars
    x_true = snake_walk()

    # make solutions
    const_1 = np.random.rand()*90+10
    const_2 = np.random.rand()*90+10
    print("CONSTANTS: ", const_1, const_2)
    y = const_1*x_true[:, 0] + const_2*x_true[:, 1]

    # test solution
    compare_agraph_explicit(x_true, y)


def compare_agraph_explicit(X, Y):
    """does the comparison"""
    # make solution manipulator
    sol_manip = agm(X.shape[1], 16, nloads=2)
    sol_manip.add_node_type(AGNodes.Add)
    sol_manip.add_node_type(AGNodes.Multiply)

    # make predictor manipulator
    pred_manip = fpm(32, Y.shape[0])

    # make and run island manager
    islmngr = SerialIslandManager(N_ISLANDS,
                                  data_x=X,
                                  data_y=Y,
                                  solution_manipulator=sol_manip,
                                  predictor_manipulator=pred_manip,
                                  fitness_metric=StandardRegression)
    assert islmngr.run_islands(MAX_STEPS, EPSILON, step_increment=N_STEPS, 
                               make_plots=False)


def test_const_opt_agraphcpp_explicit():
    """test optimization code"""
    print("-----test_const_opt_agraphcpp_explicit-----")
    # get independent vars
    x_true = snake_walk()

    # make solutions
    consts = np.random.rand(6)*90+10
    print("CONSTANTS: ", consts)
    y = consts[0] + consts[1] * x_true[:, 0] + consts[2] * x_true[:, 1] + \
        consts[3] * x_true[:, 0] * x_true[:, 0] + \
        consts[4] * x_true[:, 1] * x_true[:, 1] + \
        consts[5] * x_true[:, 0] * x_true[:, 1]
    y = y.reshape([-1, 1])

    # create manipulator to set things up
    sol_manip = agcm(x_true.shape[1], 21, nloads=2)
    sol_manip.add_node_type(2)
    sol_manip.add_node_type(4)

    # create gene with proper functional form
    sol = sol_manip.generate()
    sol.command_list[0] = (1, (-1,))
    sol.command_list[1] = (1, (-1,))
    sol.command_list[2] = (1, (-1,))
    sol.command_list[3] = (1, (-1,))
    sol.command_list[4] = (1, (-1,))
    sol.command_list[5] = (1, (-1,))

    sol.command_list[6] = (0, (0,))
    sol.command_list[7] = (0, (1,))
    sol.command_list[8] = (4, (6, 6))
    sol.command_list[9] = (4, (7, 7))
    sol.command_list[10] = (4, (6, 7))

    sol.command_list[11] = (4, (1, 6))
    sol.command_list[12] = (4, (2, 7))
    sol.command_list[13] = (4, (3, 8))
    sol.command_list[14] = (4, (4, 9))
    sol.command_list[15] = (4, (5, 10))

    sol.command_list[16] = (2, (0, 11))
    sol.command_list[17] = (2, (16, 12))
    sol.command_list[18] = (2, (17, 13))
    sol.command_list[19] = (2, (18, 14))
    sol.command_list[20] = (2, (19, 15))
    # print(sol.latexstring())

    # fit the constants
    sol.evaluate(x_true, StandardRegression, x=x_true, y=y)

    # make sure constants are close
    c_fit = sol.constants
    print("FITTED:    ", c_fit)
    for tru, fit in zip(consts, c_fit):
        assert np.abs(tru - fit) < 1e-8


def test_const_opt_agraphcpp_implicit():
    """test optimization code"""
    print("-----test_const_opt_agraphcpp_implicit-----")
    # get independent vars
    x_true = snake_walk()

    # make solutions
    consts = np.random.rand(6)*90+10
    print("CONSTANTS: ", consts[1:])
    y = consts[0] + consts[1] * x_true[:, 0] + consts[2] * x_true[:, 1] + \
        consts[3] * x_true[:, 0] * x_true[:, 0] + \
        consts[4] * x_true[:, 1] * x_true[:, 1] + \
        consts[5] * x_true[:, 0] * x_true[:, 1]
    x_true = np.hstack((x_true, y.reshape((-1, 1))))
    x_true, dx_dt, _ = calculate_partials(x_true)

    # create manipulator to set things up
    sol_manip = agcm(x_true.shape[1], 23, nloads=2)
    sol_manip.add_node_type(2)
    sol_manip.add_node_type(3)
    sol_manip.add_node_type(4)

    # create gene with proper functional form
    sol = sol_manip.generate()
    sol.command_list[0] = (1, (-1,))
    sol.command_list[1] = (1, (-1,))
    sol.command_list[2] = (1, (-1,))
    sol.command_list[3] = (1, (-1,))
    sol.command_list[4] = (1, (-1,))
    sol.command_list[5] = (1, (-1,))

    sol.command_list[6] = (0, (0,))
    sol.command_list[7] = (0, (1,))
    sol.command_list[8] = (4, (6, 6))
    sol.command_list[9] = (4, (7, 7))
    sol.command_list[10] = (4, (6, 7))

    sol.command_list[11] = (4, (1, 6))
    sol.command_list[12] = (4, (2, 7))
    sol.command_list[13] = (4, (3, 8))
    sol.command_list[14] = (4, (4, 9))
    sol.command_list[15] = (4, (5, 10))

    sol.command_list[16] = (2, (0, 11))
    sol.command_list[17] = (2, (16, 12))
    sol.command_list[18] = (2, (17, 13))
    sol.command_list[19] = (2, (18, 14))
    sol.command_list[20] = (2, (19, 15))
    sol.command_list[21] = (0, (2,))
    sol.command_list[22] = (3, (20, 21))
    # print(sol.latexstring())

    # fit the constants
    t0 = time.time()
    _, df_dx = sol.evaluate_deriv(x_true, ImplicitRegression,
                                  x=x_true, dx_dt=dx_dt, required_params=3)
    t1 = time.time()
    print("fit time: ", t1-t0, "seconds")

    # make sure constants are close
    c_fit = sol.constants
    print("average accuracy: ",
          np.mean(np.abs(np.array(c_fit[1:]) - np.array(consts[1:]))/
                  np.array(consts[1:])))
    print("FITTED:    ", c_fit[1:])
    for tru, fit in zip(consts[1:], c_fit[1:]):
        print(np.abs(tru - fit) / tru)
        assert np.abs(tru - fit)/tru < 0.1


def test_const_opt_agraphcpp_explicit_evo():
    """test add primative in sym reg"""
    print("-----test_const_opt_agraphcpp_explicit_evo-----")
    # get independent vars
    x_true = snake_walk()

    # make solutions
    const_1 = np.random.rand()*90+10
    const_2 = np.random.rand()*90+10
    print("CONSTANTS: ", const_1, const_2)
    y = (const_1*x_true[:, 0] + const_2*x_true[:, 1]).reshape([-1, 1])

    # test solution
    compare_agraphcpp_explicit(x_true, y)


def compare_agraphcpp_explicit(X, Y):
    """does the comparison"""
    # make solution manipulator
    sol_manip = agcm(X.shape[1], 16, nloads=2)
    sol_manip.add_node_type(2)
    sol_manip.add_node_type(4)

    # make predictor manipulator
    pred_manip = fpm(32, Y.shape[0])

    # make and run island manager
    islmngr = SerialIslandManager(N_ISLANDS,
                                  data_x=X,
                                  data_y=Y,
                                  solution_manipulator=sol_manip,
                                  predictor_manipulator=pred_manip,
                                  fitness_metric=StandardRegression)
    assert islmngr.run_islands(MAX_STEPS, EPSILON, step_increment=N_STEPS, 
                               make_plots=False)
