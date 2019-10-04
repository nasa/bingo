import timeit

import numpy as np

from bingo.symbolic_regression.agraph \
    import agraph as agraph_module, backend as pyBackend
from bingocpp.build import bingocpp as cppBackend

import tests.benchmark_data as benchmark_data
from tests.benchmark_data import TEST_AGRAPHS, TEST_X

def benchmark_evaluate():
    for indv in TEST_AGRAPHS:
        _ = indv.evaluate_equation_at(TEST_X)


def benchmark_evaluate_w_x_derivative():
    for indv in TEST_AGRAPHS:
        _ = indv.evaluate_equation_with_x_gradient_at(TEST_X)


def benchmark_evaluate_w_c_derivative():
    for indv in TEST_AGRAPHS:
        _ = indv.evaluate_equation_with_local_opt_gradient_at(TEST_X)


def do_benchmarking():
    printer = benchmark_data.StatsPrinter()
    for backend, name in [[pyBackend, "py"], [cppBackend, "c++"]]:
        agraph_module.Backend = backend
        printer.add_stats(name + ": evaluate",
                          timeit.repeat(benchmark_evaluate,
                                        number=100, repeat=10))
        printer.add_stats(name + ": x derivative",
                          timeit.repeat(benchmark_evaluate_w_x_derivative,
                                        number=100, repeat=10))
        printer.add_stats(name + ": c derivative",
                          timeit.repeat(benchmark_evaluate_w_c_derivative,
                                        number=100, repeat=10))
    printer.print()