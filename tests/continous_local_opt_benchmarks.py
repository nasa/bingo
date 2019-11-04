import timeit

import numpy as np

from bingo.symbolic_regression.agraph \
    import agraph as agraph_module, backend as pyBackend
from bingo.symbolic_regression.agraph.agraph import AGraph
from bingo.local_optimizers.continuous_local_opt \
    import ContinuousLocalOptimization
from bingocpp.build import bingocpp as cppBackend


from tests.benchmark_data import StatsPrinter, TEST_AGRAPHS, TEST_AGRAPHS_CPP
from tests.fitness_benchmark import TEST_EXPLICIT_REGRESSION, \
                                    TEST_EXPLICIT_REGRESSION_CPP, \
                                    TEST_IMPLICIT_REGRESSION, TEST_IMPLICIT_REGRESSION_CPP

TEST_EXPLICIT_REGRESSION_OPTIMIZATION \
    = ContinuousLocalOptimization(TEST_EXPLICIT_REGRESSION)
TEST_IMPLICIT_REGRESSION_OPTIMIZATION \
    = ContinuousLocalOptimization(TEST_IMPLICIT_REGRESSION)
TEST_EXPLICIT_REGRESSION_OPTIMIZATION_CPP \
    = ContinuousLocalOptimization(TEST_EXPLICIT_REGRESSION_CPP)
TEST_IMPLICIT_REGRESSION_OPTIMIZATION_CPP \
    = ContinuousLocalOptimization(TEST_IMPLICIT_REGRESSION_CPP)


TEST_AGRAPH_LISTS = []
TEST_AGRAPH_LISTS_CPP = []


def benchmark_explicit_regression_with_optimization():
    for test_run in TEST_AGRAPH_LISTS:
        for indv in test_run:
            _ = TEST_EXPLICIT_REGRESSION_OPTIMIZATION.__call__(indv)


def benchmark_explicit_regression_cpp_with_optimization():
    for test_run in TEST_AGRAPH_LISTS_CPP:
        for indv in test_run:
            _ = TEST_EXPLICIT_REGRESSION_OPTIMIZATION_CPP.__call__(indv)


def benchmark_implicit_regression_with_optimization():
    for test_run in TEST_AGRAPH_LISTS:
        for indv in test_run:
            _ = TEST_IMPLICIT_REGRESSION.__call__(indv)


def benchmark_implicit_regression_cpp_with_optimization():
    for test_run in TEST_AGRAPH_LISTS_CPP:
        for indv in test_run:
            _ = TEST_IMPLICIT_REGRESSION_CPP.__call__(indv)


def reset_test_data():
    TEST_AGRAPH_LISTS.clear()
    for num_runs in range(0, 100):
        run_list = []
        for agraph in TEST_AGRAPHS:
            run_list.append(agraph.copy())
        TEST_AGRAPH_LISTS.append(run_list)


def reset_test_data_cpp():
    TEST_AGRAPH_LISTS_CPP.clear()
    for num_runs in range(0, 100):
        run_list = []
        for agraph in TEST_AGRAPHS_CPP:
            run_list.append(agraph.copy())
        TEST_AGRAPH_LISTS_CPP.append(run_list)


def do_benchmarking():
    benchmarks = [
        [benchmark_explicit_regression_with_optimization,
         benchmark_explicit_regression_cpp_with_optimization,
         "LOCAL OPTIMIZATION: EXPLICIT REGRESSION BENCHMARKS"], 
        [benchmark_implicit_regression_with_optimization, 
         benchmark_implicit_regression_cpp_with_optimization,
         "LOCAL OPTIMIZATION: IMPLICIT REGRESSION BENCHMARKS"]]

    stats_printer_list = []
    for regression, regression_cpp, reg_name in benchmarks:
        printer = StatsPrinter(reg_name)
        _run_benchmarks(printer, regression, regression_cpp)
        stats_printer_list.append(printer)

    _print_stats(stats_printer_list)


def _run_benchmarks(printer, regression, regression_cpp):
    for backend, name in [[pyBackend, " py"], [cppBackend, "c++"]]:
        agraph_module.Backend = backend
        printer.add_stats("py:  fitness " +name + ": evaluate ",
                          timeit.repeat(regression, setup=reset_test_data,
                                        number=1, repeat=10))
    printer.add_stats("c++: fitness c++: evaluate ",
                      timeit.repeat(regression_cpp, setup=reset_test_data_cpp,
                                    number=1, repeat=10))


def _print_stats(printer_list):
    for printer in printer_list:
        printer.print()


if __name__ == '__main__':
    do_benchmarking()