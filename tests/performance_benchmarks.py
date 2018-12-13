# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring

import timeit
import numpy as np

from bingo.AGraph import AGraph
from bingo.AGraph.AGraphGenerator import AGraphGenerator
from bingo.AGraph.ComponentGenerator import ComponentGenerator
from bingo.AGraph import Backend as pyBackend
from bingocpp.build import bingocpp as cppBackend


def generate_random_individuals(num_individuals, stack_size):
    np.random.seed(0)
    generate_agraph = set_up_agraph_generator(stack_size)

    individuals = [generate_agraph() for _ in range(num_individuals)]
    set_constants(individuals)

    return individuals


def set_up_agraph_generator(stack_size):
    generator = ComponentGenerator(input_x_dimension=4,
                                   num_initial_load_statements=2,
                                   terminal_probability=0.1)
    for i in range(2, 13):
        generator.add_operator(i)
    generate_agraph = AGraphGenerator(stack_size, generator)
    return generate_agraph


def set_constants(individuals):
    for indv in individuals:
        num_consts = indv.get_number_local_optimization_params()
        if num_consts > 0:
            consts = np.random.rand(num_consts) * 10.0
            indv.set_local_optimization_params(consts)


def generate_random_x(size):
    np.random.seed(0)
    return np.random.rand(size, 4)*10 - 5.0


TEST_AGRAPHS = generate_random_individuals(100, 128)
TEST_X = generate_random_x(128)


def benchmark_evaluate():
    for indv in TEST_AGRAPHS:
        _ = indv.evaluate_equation_at(TEST_X)


def benchmark_evaluate_w_x_derivative():
    for indv in TEST_AGRAPHS:
        _ = indv.evaluate_equation_with_x_gradient_at(TEST_X)


def benchmark_evaluate_w_c_derivative():
    for indv in TEST_AGRAPHS:
        _ = indv.evaluate_equation_with_local_opt_gradient_at(TEST_X)


class StatsPrinter:
    def __init__(self):
        self._header_format_string = \
            "{:<25}   {:>10} +- {:<10}   {:^10}   {:^10}"
        self._format_string = \
            "{:<25}   {:>10.4f} +- {:<10.4f}   {:^10.4f}   {:^10.4f}"
        self._output = ["-"*23+":::: PERFORMANCE BENCHMARKS ::::" + "-"*23,
                        self._header_format_string.format("NAME", "MEAN",
                                                          "STD", "MIN", "MAX"),
                        "-"*78]

    def add_stats(self, name, times):
        std_time = np.std(times)
        mean_time = np.mean(times)
        max_time = np.max(times)
        min_time = np.min(times)

        self._output.append(self._format_string.format(name, mean_time,
                                                       std_time, min_time,
                                                       max_time))

    def print(self):
        for line in self._output:
            print(line)


def do_benchmarking():
    printer = StatsPrinter()
    for backend, name in [[pyBackend, "py"], [cppBackend, "c++"]]:
        AGraph.Backend = backend
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


if __name__ == '__main__':

    do_benchmarking()
