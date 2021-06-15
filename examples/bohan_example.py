import numpy as np
from bingo.symbolic_regression import ComponentGenerator, AGraphGenerator
from bingo.symbolic_regression.agraph.evaluation_backend.evaluation_backend \
    import evaluate
from bingo.symbolic_regression.agraph.string_generation \
    import get_formatted_string

from time import time


def _create_random_equation():
    comp_gen = ComponentGenerator(2)
    for op in ["+", "-", "*"]:
        comp_gen.add_operator(op)
    equ_gen = AGraphGenerator(32, comp_gen, use_simplification=True)

    equ = equ_gen()
    while equ.get_number_local_optimization_params() < 2 \
            or equ.get_complexity() < 10:
        equ = equ_gen()

    return equ._simplified_command_array, \
           equ.get_number_local_optimization_params()


if __name__ == "__main__":
    # equation to use in the example
    COMMAND_ARRAY = np.array([[1, 0, 0],
                              [-1, 2, 2],
                              [1, 1, 1],
                              [4, 1, 2],
                              [0, 0, 0],
                              [4, 3, 4],
                              [-1, -1, -1],
                              [4, 6, 2],
                              [4, 7, 4],
                              [2, 5, 8],
                              [2, 0, 9],
                              [0, 1, 1],
                              [3, 10, 11]])
    NUM_CONSTS = 2

    # you can test out other equations by uncommenting the next line
    # COMMAND_ARRAY, NUM_CONSTS = _create_random_equation()

    # you can print out the equation by uncommenting one of the following lines
    # print(get_formatted_string("console", COMMAND_ARRAY,
    #                            [f"C_{i}" for i in range(NUM_CONSTS)]))
    # print(get_formatted_string("stack", COMMAND_ARRAY,
    #                            [f"C_{i}" for i in range(NUM_CONSTS)]))

    # this is roughly representative of the dimensions of the data we are using
    CONSTANTS = np.linspace(0, 1, 800 * NUM_CONSTS).reshape(NUM_CONSTS, 800)
    X_DATA = np.linspace(-10, 10, 400).reshape(200, 2)

    # this is the evaluation of the equation
    # the evaluation function is where we want to start off looking for speedup
    # we may end up moving more work to the GPU but lets start with this
    start = time()
    Y_PREDICTION = evaluate(COMMAND_ARRAY, X_DATA, CONSTANTS)
    end = time()
    print("Time elapsed (seconds): ", end - start)







