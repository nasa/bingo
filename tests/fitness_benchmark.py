import timeit

import numpy as np

from bingo.symbolic_regression.agraph \
    import agraph as agraph_module, backend as pyBackend
from bingo.symbolic_regression.implicit_regression \
    import ImplicitRegression, ImplicitTrainingData, calculate_partials
from bingo.symbolic_regression.explicit_regression \
    import ExplicitRegression, ExplicitTrainingData
from bingocpp.build import bingocpp as cppBackend

import tests.benchmark_data as benchmark_data
from tests.benchmark_data import TEST_AGRAPHS, TEST_X, TEST_AGRAPHS_CPP


def benchmark_explicit_regression():
    for indv in TEST_AGRAPHS:
        _ = TEST_EXPLICIT_REGRESSION.__call__(indv)


def benchmark_explicit_regression_cpp():
    for indv in TEST_AGRAPHS_CPP:
        _ = TEST_EXPLICIT_REGRESSION_CPP.__call__(indv)


def benchmark_implicit_regression_2():
    for indv in TEST_AGRAPHS:
        _ = TEST_IMPLICIT_REGRESSION_2.__call__(indv)


def benchmark_implicit_regression_2_cpp():
    for indv in TEST_AGRAPHS_CPP:
        _ = TEST_IMPLICIT_REGRESSION_2_CPP.__call__(indv)


def benchmark_implicit_regression_1():
    for indv in TEST_AGRAPHS:
        _ = TEST_IMPLICIT_REGRESSION_1.__call__(indv)


def benchmark_implicit_regression_1_cpp():
    for indv in TEST_AGRAPHS_CPP:
        _ = TEST_IMPLICIT_REGRESSION_1_CPP.__call__(indv)

def _initialize_implicit_data():
    x, dx_dt, _ = calculate_partials(TEST_X)
    return x, dx_dt

TEST_X_PARTIALS, TEST_DX_DT = _initialize_implicit_data()
TEST_Y_ZEROS = np.zeros(TEST_X_PARTIALS.shape)

TEST_EXPLICIT_TRAINING_DATA =  ExplicitTrainingData(TEST_X_PARTIALS, TEST_Y_ZEROS)
TEST_EXPLICIT_REGRESSION = ExplicitRegression(TEST_EXPLICIT_TRAINING_DATA)
TEST_EXPLICIT_TRAINING_DATA_CPP \
    = cppBackend.ExplicitTrainingData(TEST_X_PARTIALS, TEST_Y_ZEROS)
TEST_EXPLICIT_REGRESSION_CPP \
    = cppBackend.ExplicitRegression(TEST_EXPLICIT_TRAINING_DATA_CPP)

TEST_IMPLICIT_TRAINING_DATA_2 = ImplicitTrainingData(TEST_X_PARTIALS, TEST_DX_DT)
TEST_IMPLICIT_REGRESSION_2 = ImplicitRegression(TEST_IMPLICIT_TRAINING_DATA_2)
TEST_IMPLICIT_TRAINING_DATA_2_CPP \
    = cppBackend.ImplicitTrainingData(TEST_X_PARTIALS, TEST_DX_DT)
TEST_IMPLICIT_REGRESSION_2_CPP \
    = cppBackend.ImplicitRegression(TEST_IMPLICIT_TRAINING_DATA_2_CPP)

TEST_IMPLICIT_TRAINING_DATA_1 = ImplicitTrainingData(TEST_X_PARTIALS, TEST_DX_DT)
TEST_IMPLICIT_REGRESSION_1 = ImplicitRegression(TEST_IMPLICIT_TRAINING_DATA_1)
TEST_IMPLICIT_TRAINING_DATA_1_CPP \
    = cppBackend.ImplicitTrainingData(TEST_X_PARTIALS, TEST_DX_DT)
TEST_IMPLICIT_REGRESSION_1_CPP \
    = cppBackend.ImplicitRegression(TEST_IMPLICIT_TRAINING_DATA_1_CPP)

def do_benchmarking():
    printer = benchmark_data.StatsPrinter()
    benchmarks = [[benchmark_explicit_regression, 
                   benchmark_explicit_regression_cpp], 
                  [benchmark_implicit_regression_2,
                   benchmark_implicit_regression_2_cpp],
                  [benchmark_implicit_regression_1,
                   benchmark_implicit_regression_1_cpp]]
    for regression, regression_cpp in benchmarks:
        _run_benchmarks(printer, regression, regression_cpp)
    printer.print()

def _run_benchmarks(printer, regression, regression_cpp):
    for backend, name in [[pyBackend, " py"], [cppBackend, "c++"]]:
        agraph_module.Backend = backend
        printer.add_stats("py:  fitness " +name + ": evaluate ",
                          timeit.repeat(regression,
                                        number=100, repeat=10))
    printer.add_stats("c++: fitness c++: evaluate ",
                          timeit.repeat(regression_cpp,
                                        number=100, repeat=10))

if __name__ == '__main__':
    do_benchmarking()