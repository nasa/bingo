import timeit

import numpy as np

from bingo.symbolic_regression.agraph \
    import agraph as agraph_module, backend as pyBackend
from bingocpp.build import bingocpp as cppBackend

import tests.benchmark_data as benchmark_data
from tests.benchmark_data import \
    TEST_AGRAPHS, \
    TEST_AGRAPHS_CPP, \
    TEST_X_PARTIALS, \
    TEST_Y_ZEROS, \
    TEST_DX_DT, \
    TEST_EXPLICIT_REGRESSION, \
    TEST_EXPLICIT_REGRESSION_CPP, \
    TEST_IMPLICIT_REGRESSION, \
    TEST_IMPLICIT_REGRESSION_CPP


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

    return stats_printer_list


def _run_benchmarks(printer, regression, regression_cpp):
    for backend, name in [[pyBackend, " py"], [cppBackend, "c++"]]:
        agraph_module.Backend = backend
        printer.add_stats(
            "py:  fitness " +name + ": evaluate ",
             timeit.repeat(regression,
                           number=benchmark_data.BENCHMARK_EVALUATION_COUNT,
                           repeat=benchmark_data.BENCHMARK_DATA_POINTS))
    printer.add_stats(
        "c++: fitness c++: evaluate ",
        timeit.repeat(regression_cpp,
                      number=benchmark_data.BENCHMARK_EVALUATION_COUNT,
                      repeat=benchmark_data.BENCHMARK_DATA_POINTS))


def _print_stats(printer_list):
    for printer in printer_list:
        printer.print()


if __name__ == '__main__':
    do_benchmarking()