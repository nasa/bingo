"""
example of regression done using the serial island manager (islands done
serially on a single process)
"""
import numpy as np

from bingo.AGraph import AGraphManipulator as agm
from bingo.AGraph import AGNodes
from bingo import AGraphCpp
from bingo.FitnessPredictor import FPManipulator as fpm
from bingo.IslandManager import SerialIslandManager
from bingo.Utils import snake_walk
from bingo.FitnessMetric import StandardRegression, ImplicitRegression


def main(max_steps, epsilon, data_size, data_range, n_islands):
    """main regression function"""

    # make data
    # X = np.linspace(data_range[0], data_range[1], data_size, False)
    # Y = X * X + 0.5
    # Y = 1.5*X*X - X*X*X
    # Y = np.exp(np.abs(X))*np.sin(X)
    # Y = X*X*np.exp(np.sin(X)) + X + np.sin(3.14159/4 - X*X*X)
    # X = X.reshape([-1, 1])

    # make data
    Xx = snake_walk()
    Yx = (Xx[:, 0] + Xx[:, 1]*0.5).reshape([-1, 1])
    Xi = np.hstack((Xx, Yx))
    Yi = None

    X=Xx
    Y=Yx
    fitness_metric=StandardRegression

    # make solution manipulator

    # sol_manip.add_node_type(AGNodes.Divide)
    # sol_manip.add_node_type(AGNodes.Exp)
    # sol_manip.add_node_type(AGNodes.Log)
    # sol_manip.add_node_type(AGNodes.Sin)
    # sol_manip.add_node_type(AGNodes.Cos)
    # sol_manip.add_node_type(AGNodes.Abs)

    # # debugging
    # sol = sol_manip.generate()
    # sol.command_list[0] = (AGNodes.Load_Data, (0,))
    # sol.command_list[1] = (AGNodes.Load_Data, (1,))
    # sol.command_list[2] = (AGNodes.Load_Const, (None,))
    # sol.command_list[3] = (AGNodes.Exp, (0,))
    # sol.command_list[4] = (AGNodes.Add, (2, 3))
    # sol.command_list[5] = (AGNodes.Exp, (4,))
    # sol.command_list[6] = (AGNodes.Subtract, (0, 1))
    # sol.command_list[-1] = (AGNodes.Divide, (5, 6))

    # make solution manipulator
    sol_manip2 = AGraphCpp.AGraphCppManipulator(X.shape[1], 16, nloads=2)
    sol_manip2.add_node_type(2)  # +
    sol_manip2.add_node_type(3)  # -
    sol_manip2.add_node_type(4)  # *
    #sol_manip.add_node_type(5)  # /

    # make predictor manipulator
    pred_manip = fpm(32, data_size)

    # make and run island manager
    islmngr = SerialIslandManager(n_islands, X, Y, sol_manip2, pred_manip,
                                  fitness_metric=fitness_metric,
                                  # required_params=3,
                                  )
    # print("--------------------------")
    # print("True fitness: %le" % islmngr.isles[0].solution_fitness_true(sol))
    # print(sol.latexstring())

    # islmngr.load_state('test.p')
    islmngr.run_islands(max_steps, epsilon, step_increment=1000)
    islmngr.save_state('test.p')
    islmngr.load_state('test.p')
    islmngr.run_islands(max_steps, epsilon, step_increment=1000)



if __name__ == "__main__":

    MAX_STEPS = 2000
    CONVERGENCE_EPSILON = 1.0e-8
    DATA_SIZE = 100
    DATA_RANGE = [-3, 3]
    N_ISLANDS = 4

    main(MAX_STEPS, CONVERGENCE_EPSILON, DATA_SIZE, DATA_RANGE, N_ISLANDS)
