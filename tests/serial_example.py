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
from bingo.FitnessMetric import StandardRegression, ImplicitRegression


def main(max_steps, epsilon, data_size, data_range, n_islands):
    """main regression function"""

    # make data
    # X = np.linspace(data_range[0], data_range[1], data_size, False)

    # Y = 1.5*X*X - X*X*X
    # Y = np.exp(X) * np.sin(X)

    # X = X.reshape([-1, 1])


    # make data
    # Xx = snake_walk()

    ########################### Circle problem
    # create a radius, and fill an array(angle) with points depending on the data size
    radius = 4
    angle = np.linspace(0, 2 * np.pi, data_size)
    # a variable to get the position in angle
    position = 0
    # create the array to hold the points representing the circle and fill the first position
    circleArray = np.array([[radius * np.cos(angle[position])], [radius * np.sin(angle[position])]])
    # loop through the data size to fill the rest of the circle Array
    for ang in angle:
        circleArray = np.append(circleArray, [[radius * np.cos(ang)], [radius * np.sin(ang)]], axis=1)

    circleArray = np.transpose(circleArray)
    # assign the array created with the points for circle to X
    X = circleArray
    ############################

    # 1.5x0 squared - x1 cubed
    # Yx = (1.5 * np.square(Xx[:, 0]) - np.power(Xx[:, 1], 3))
    # Yx = (1.5 * (Xx[:, 0] * Xx[:, 0]) - (Xx[:, 1] * Xx[:, 1] * Xx[:, 1]))

    # Yx = (Xx[:, 0] / Xx[:, 1])

    # Xi = np.hstack((Xx, Yx))
    # Yi = None

    # X=Xx
    # Y=Yx
    # fitness_metric=StandardRegression
    Y = None
    fitness_metric=ImplicitRegression

    # make solution manipulator

    sol_manip = agm(X.shape[1], 16, nloads=2)
    
    sol_manip.add_node_type(AGNodes.Add)
    sol_manip.add_node_type(AGNodes.Subtract)
    sol_manip.add_node_type(AGNodes.Multiply)
    sol_manip.add_node_type(AGNodes.Divide)
    # sol_manip.add_node_type(AGNodes.Exp)
    # sol_manip.add_node_type(AGNodes.Log)
    # sol_manip.add_node_type(AGNodes.Sin)
    # sol_manip.add_node_type(AGNodes.Cos)
    sol_manip.add_node_type(AGNodes.Abs)

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
    #sol_manip2 = AGraphCpp.AGraphCppManipulator(X.shape[1], 16, nloads=2)
    #sol_manip2.add_node_type(2)  # +
    #sol_manip2.add_node_type(3)  # -
    #sol_manip2.add_node_type(4)  # *
    #sol_manip2.add_node_type(5)  # /

    # make predictor manipulator
    pred_manip = fpm(32, X.shape[0])

    # make and run island manager
    islmngr = SerialIslandManager(n_islands,
                                  data_x=X,
                                  data_y=Y,
                                  solution_manipulator=sol_manip,
                                  predictor_manipulator=pred_manip,
                                  fitness_metric=fitness_metric,
                                  # required_params=3,
                                  )
    # print("--------------------------")
    # print("True fitness: %le" % islmngr.isles[0].solution_fitness_true(sol))
    # print(sol.latexstring())

    # islmngr.load_state('test.p')
    islmngr.run_islands(max_steps, epsilon, step_increment=100)
    # islmngr.save_state('test.p')
    # islmngr.load_state('test.p')
    islmngr.run_islands(max_steps, epsilon, step_increment=100)


if __name__ == "__main__":

    MAX_STEPS = 2000
    CONVERGENCE_EPSILON = 1.0e-8
    DATA_SIZE = 100
    DATA_RANGE = [-3, 3]
    N_ISLANDS = 2

    main(MAX_STEPS, CONVERGENCE_EPSILON, DATA_SIZE, DATA_RANGE, N_ISLANDS)
