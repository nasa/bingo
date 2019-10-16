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


def benchmark_implicit_regression():
    for indv in TEST_AGRAPHS:
        _ = TEST_IMPLICIT_REGRESSION.__call__(indv)


def benchmark_implicit_regression_cpp():
    for indv in TEST_AGRAPHS_CPP:
        _ = TEST_IMPLICIT_REGRESSION_CPP.__call__(indv)


def _initialize_implicit_data():
    x, dx_dt, _ = calculate_partials(TEST_X)
    return x, dx_dt


TEST_X_PARTIALS, TEST_DX_DT = _initialize_implicit_data()
TEST_Y_ZEROS = np.zeros(TEST_X_PARTIALS.shape)


def explicit_regression():
    TEST_EXPLICIT_TRAINING_DATA \
        = ExplicitTrainingData(TEST_X_PARTIALS, TEST_Y_ZEROS)
    return ExplicitRegression(TEST_EXPLICIT_TRAINING_DATA)


def explicit_regression_cpp():
    training_data = cppBackend.ExplicitTrainingData(TEST_X_PARTIALS, TEST_Y_ZEROS)
    return cppBackend.ExplicitRegression(training_data)


def implicit_regression():
    training_data = ImplicitTrainingData(TEST_X_PARTIALS, TEST_DX_DT)
    return ImplicitRegression(training_data)


def implicit_regression_cpp():
    training_data = cppBackend.ImplicitTrainingData(TEST_X_PARTIALS, TEST_DX_DT)
    return cppBackend.ImplicitRegression(training_data)


TEST_EXPLICIT_REGRESSION = explicit_regression()
TEST_EXPLICIT_REGRESSION_CPP = explicit_regression_cpp()

TEST_IMPLICIT_REGRESSION = implicit_regression()
TEST_IMPLICIT_REGRESSION_CPP = implicit_regression_cpp()


def do_benchmarking():
    benchmarks = [
        [benchmark_explicit_regression, benchmark_explicit_regression_cpp,
         "EXPLICIT REGRESSION BENCHMARKS"], 
        [benchmark_implicit_regression, benchmark_implicit_regression_cpp,
         "IMPLICIT REGRESSION BENCHMARKS"]]

    stats_printer_list = []
    for regression, regression_cpp, reg_name in benchmarks:
        printer = benchmark_data.StatsPrinter(reg_name)
        _run_benchmarks(printer, regression, regression_cpp)
        stats_printer_list.append(printer)

    _print_stats(stats_printer_list)


def _run_benchmarks(printer, regression, regression_cpp):
    for backend, name in [[pyBackend, " py"], [cppBackend, "c++"]]:
        agraph_module.Backend = backend
        printer.add_stats("py:  fitness " +name + ": evaluate ",
                          timeit.repeat(regression,
                                        number=100, repeat=10))
    printer.add_stats("c++: fitness c++: evaluate ",
                          timeit.repeat(regression_cpp,
                                        number=100, repeat=10))


def _print_stats(printer_list):
    for printer in printer_list:
        printer.print()


if __name__ == '__main__':
    do_benchmarking()