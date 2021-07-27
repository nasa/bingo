import numpy as np
import cupy as cp
from cupyx.time import repeat
from bingo.symbolic_regression import ComponentGenerator, AGraphGenerator, AGraph
from bingo.symbolic_regression.agraph.evaluation_backend.evaluation_backend \
    import evaluate
from bingo.symbolic_regression.agraph.string_generation \
    import get_formatted_string

import bingo.util.global_imports as gi

from time import time
from scipy.stats import describe

import sys

def _evaluate_from_np(graph, data, constants):
    cp_data = cp.asarray(data)
    cp_constants = cp.asarray(constants)

    graph.set_local_optimization_params(cp_constants)
    return graph.evaluate_equation_at(cp_data).get()

def _create_random_equation():
    comp_gen = ComponentGenerator(2)
    for op in ["+", "-", "*", "/", "|",
               "sin", "cos",
                "log", "sqrt", "pow", "exp"]:
        comp_gen.add_operator(op)
    equ_gen = AGraphGenerator(32, comp_gen, use_simplification=True)

    equ = equ_gen()
    while equ.get_number_local_optimization_params() < 2 \
            or equ.get_complexity() < 10:
        equ = equ_gen()

    return equ

def _get_smcbingo_model():
    command_array = np.array([[-1, 2, 2],
                              [1, 0, 0],
                              [4, 0, 1],
                              [1, 1, 1],
                              [0, 0, 0],
                              [4, 3, 4],
                              [2, 2, 5],
                              [-1, -2, -2],
                              [2, 7, 2],
                              [4, 8, 4],
                              [2, 1, 9],
                              [1, 2, 2],
                              [-1, -1, -1],
                              [4, 12, 1],
                              [4, 13, 4],
                              [2, 12, 1],
                              [4, 15, 4],
                              [2, 7, 1],
                              [2, 1, 16],
                              [4, 4, 18],
                              [4, 17, 19],
                              [2, 14, 5],
                              [2, 16, 20],
                              [2, 21, 22],
                              [2, 11, 23],
                              [4, 10, 24],
                              [4, 6, 25]])

    data_x = np.array([[0.],
                       [0.04759989],
                       [0.09519978],
                       [0.14279967],
                       [0.19039955],
                       [0.23799944],
                       [0.28559933],
                       [0.33319922],
                       [0.38079911],
                       [0.428399],
                       [0.47599889],
                       [0.52359878],
                       [0.57119866],
                       [0.61879855],
                       [0.66639844],
                       [0.71399833],
                       [0.76159822],
                       [0.80919811],
                       [0.856798],
                       [0.90439789],
                       [0.95199777],
                       [0.99959766],
                       [1.04719755],
                       [1.09479744],
                       [1.14239733],
                       [1.18999722],
                       [1.23759711],
                       [1.28519699],
                       [1.33279688],
                       [1.38039677],
                       [1.42799666],
                       [1.47559655],
                       [1.52319644],
                       [1.57079633],
                       [1.61839622],
                       [1.6659961],
                       [1.71359599],
                       [1.76119588],
                       [1.80879577],
                       [1.85639566],
                       [1.90399555],
                       [1.95159544],
                       [1.99919533],
                       [2.04679521],
                       [2.0943951],
                       [2.14199499],
                       [2.18959488],
                       [2.23719477],
                       [2.28479466],
                       [2.33239455],
                       [2.37999443],
                       [2.42759432],
                       [2.47519421],
                       [2.5227941],
                       [2.57039399],
                       [2.61799388],
                       [2.66559377],
                       [2.71319366],
                       [2.76079354],
                       [2.80839343],
                       [2.85599332],
                       [2.90359321],
                       [2.9511931],
                       [2.99879299],
                       [3.04639288],
                       [3.09399276],
                       [3.14159265],
                       [3.18919254],
                       [3.23679243],
                       [3.28439232],
                       [3.33199221],
                       [3.3795921],
                       [3.42719199],
                       [3.47479187],
                       [3.52239176],
                       [3.56999165],
                       [3.61759154],
                       [3.66519143],
                       [3.71279132],
                       [3.76039121],
                       [3.8079911],
                       [3.85559098],
                       [3.90319087],
                       [3.95079076],
                       [3.99839065],
                       [4.04599054],
                       [4.09359043],
                       [4.14119032],
                       [4.1887902],
                       [4.23639009],
                       [4.28398998],
                       [4.33158987],
                       [4.37918976],
                       [4.42678965],
                       [4.47438954],
                       [4.52198943],
                       [4.56958931],
                       [4.6171892],
                       [4.66478909],
                       [4.71238898]])

    equ = AGraph(use_simplification=True)
    equ.command_array = command_array
    constants = np.random.rand((3))
    return equ, data_x, constants

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

if __name__ == "__main__":
    # equation to use in the example
    """
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
    """

    graph = _create_random_equation() # _get_smcbingo_model() #
    NUM_CONSTS = graph.get_number_local_optimization_params()
    print(get_formatted_string("console", graph._simplified_command_array, [f"C_{i}" for i in range(NUM_CONSTS)]))

    # you can print out the equation by uncommenting one of the following lines
    # print(get_formatted_string("console", COMMAND_ARRAY,
    #                            [f"C_{i}" for i in range(NUM_CONSTS)]))
    # print(get_formatted_string("stack", COMMAND_ARRAY,
    #                            [f"C_{i}" for i in range(NUM_CONSTS)]))

    # this is roughly representative of the dimensions of the data we are using

    num_trials = 1000

    np_times = np.zeros((num_trials))
    gpu_times = np.zeros((num_trials))
    cpu_times = np.zeros((num_trials))

    for i in range(num_trials):
        constant_data_size = int(sys.argv[1])
        data_size = 100
        CONSTANTS = np.random.rand(NUM_CONSTS, constant_data_size)
        X_DATA = np.linspace(-10, 10, data_size * 2).reshape(data_size, 2)
        X_DATA_GPU = cp.asarray(X_DATA)
        CONSTANTS_GPU = cp.asarray(CONSTANTS)

        gi.set_use_gpu(False)
        start = time()
        Y_PREDICTION = evaluate(graph._simplified_command_array, X_DATA, CONSTANTS)
        mid = time()
        gi.set_use_gpu(True)

        #start_gpu = cp.cuda.Event()
        #nd_gpu = cp.cuda.Event()
        #start_gpu.record()
        COMMAND_ARRAY_GPU = cp.asarray(graph._simplified_command_array)
        start_cpu = time()
        #Y_PREDICTION_GPU = _evaluate_from_np(graph, X_DATA, CONSTANTS)
        Y_PREDICTION_GPU = evaluate(COMMAND_ARRAY_GPU, X_DATA_GPU, CONSTANTS_GPU)
        end_cpu = time()
        #end_gpu.record()
        #end_gpu.synchronize()

        np_times[i] = mid - start
        cpu_times[i] = end_cpu - start_cpu
        #gpu_times[i] = cp.cuda.get_elapsed_time(start_gpu, end_gpu) / 1000

        #print(Y_PREDICTION_GPU)
        #print(Y_PREDICTION)
        print(repr(Y_PREDICTION_GPU))
        np.testing.assert_allclose(Y_PREDICTION_GPU.get(), Y_PREDICTION)

    avg_np_time = sum(np_times) / num_trials
    avg_gpu_time = sum(gpu_times) / num_trials
    avg_cpu_time = sum(cpu_times) / num_trials

    print("Average time elapsed on CPU for parallelized example (seconds): ", avg_cpu_time)
    print("Average time elapsed on GPU for parallelized example (seconds): ", avg_gpu_time)

    print("Average time elapsed for original example (seconds): ", avg_np_time)

    print("-----------------np stats-----------------")
    print(describe(reject_outliers(np_times)))
    print("-----------------cpu stats-----------------")
    print(describe(reject_outliers(cpu_times)))
#    print("-----------------gpu stats-----------------")
#    print(describe(reject_outliers(gpu_times)))

    #results = repeat(evaluate, (COMMAND_ARRAY, X_DATA_GPU, CONSTANTS_GPU), kwargs = {'use_gpu': True}, n_repeat = 1)
    #print(results)
    









