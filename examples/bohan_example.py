import numpy as np
import cupy as cp
from cupyx.time import repeat
from bingo.symbolic_regression import ComponentGenerator, AGraphGenerator
from bingo.symbolic_regression.agraph.evaluation_backend.evaluation_backend \
    import evaluate
from bingo.symbolic_regression.agraph.string_generation \
    import get_formatted_string

import bingo.symbolic_regression.agraph.evaluation_backend.operator_eval as operator_eval

from time import time

import sys


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
    COMMAND_ARRAY = np.array([[1, 0, 0],    # 0
                              [-1, 2, 2],   # 1
                              [1, 1, 1],    # 2
                              [4, 1, 2],    # 3
                              [0, 0, 0],    # 4
                              [4, 3, 4],    # 5
                              [-1, -1, -1], # 6
                              [4, 6, 2],    # 7
                              [4, 7, 4],    # 8
                              [2, 5, 8],    # 9
                              [2, 0, 9],    # 10
                              [0, 1, 1],    # 11
                              [3, 10, 11]]) # 12
    NUM_CONSTS = 2

    # you can test out other equations by uncommenting the next line
    # COMMAND_ARRAY, NUM_CONSTS = _create_random_equation()

    # you can print out the equation by uncommenting one of the following lines
    # print(get_formatted_string("console", COMMAND_ARRAY,
    #                            [f"C_{i}" for i in range(NUM_CONSTS)]))
    # print(get_formatted_string("stack", COMMAND_ARRAY,
    #                            [f"C_{i}" for i in range(NUM_CONSTS)]))

    # this is roughly representative of the dimensions of the data we are using
    constant_data_size = int(sys.argv[1])
    data_size = constant_data_size
    CONSTANTS = np.linspace(0, 1, constant_data_size * NUM_CONSTS).reshape(NUM_CONSTS, constant_data_size)
    X_DATA = np.linspace(-10, 10, data_size * 2).reshape(data_size, 2)

    # this is the evaluation of the equation
    # the evaluation function is where we want to start off looking for speedup
    # we may end up moving more work to the GPU but lets start with this

    operator_eval.set_use_gpu(False)
    start = time()
    Y_PREDICTION = evaluate(COMMAND_ARRAY, X_DATA, CONSTANTS)
    mid = time()
    operator_eval.set_use_gpu(True)

    with cp.cuda.Device(0):
        CONSTANTS_GPU = cp.array(CONSTANTS)
        X_DATA_GPU = cp.array(X_DATA)

    """
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()
    start_gpu.record()
    start_cpu = time()
    Y_PREDICTION_GPU = evaluate(COMMAND_ARRAY, X_DATA_GPU, CONSTANTS_GPU, use_gpu=True)
    end_cpu = time()
    end_gpu.record()
    end_gpu.synchronize()

    #print(Y_PREDICTION_GPU)
    #print(Y_PREDICTION)
    np.testing.assert_allclose(Y_PREDICTION_GPU.get(), Y_PREDICTION)
    
    print("Time elapsed on CPU for parallelized example (seconds): ", end_cpu - start_cpu)
    print("Time elapsed on GPU for parallelized example (seconds): ", cp.cuda.get_elapsed_time(start_gpu, end_gpu) / 1000)
    """

    print("Time elapsed for original example (seconds): ", mid - start)
    results = repeat(evaluate, (COMMAND_ARRAY, X_DATA_GPU, CONSTANTS_GPU), kwargs = {'use_gpu': True}, n_repeat = 1000)
    print(results)
    









