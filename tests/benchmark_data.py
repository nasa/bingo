# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import csv
import numpy as np

from bingo.symbolic_regression.agraph import agraph as agraph_module
from bingo.symbolic_regression.agraph.generator import AGraphGenerator
from bingo.symbolic_regression.agraph.component_generator \
    import ComponentGenerator

from bingocpp.build import bingocpp

LOG_WIDTH = 78

class StatsPrinter:
    def __init__(self, title="PERFORMANCE BENCHMARKS"):
        self._header_format_string = \
            "{:<26}   {:>10} +- {:<10}   {:^10}   {:^10}"
        self._format_string = \
            "{:<26}   {:>10.4f} +- {:<10.4f}   {:^10.4f}   {:^10.4f}"
        diff = LOG_WIDTH - len(title) - 10
        self._output = [
            "-"*int(diff/2)+":::: {} ::::".format(title) + "-"*int((diff + 1)/2),
            self._header_format_string.format("NAME", "MEAN",
                                              "STD", "MIN", "MAX"),
            "-"*LOG_WIDTH]

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
        print()


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


def copy_to_cpp(indvs_python):
    indvs_cpp = []
    for indv in indvs_python:
        agraph_cpp = bingocpp.AGraph()
        agraph_cpp.genetic_age = indv.genetic_age
        agraph_cpp.fitness = indv.fitness if indv.fitness != None else 1e9
        agraph_cpp.fit_set = indv.fit_set
        agraph_cpp.set_local_optimization_params(indv.constants)
        agraph_cpp.command_array = indv.command_array
        indvs_cpp.append(agraph_cpp)
    return indvs_cpp

def generate_random_x(size):
    np.random.seed(0)
    return np.random.rand(size, 4)*10 - 5.0


def write_stacks(test_agraph_list):
    filename = '../bingocpp/app/test-agraph-stacks.csv'
    with open(filename, mode='w+') as stack_file:
        stack_file_writer = csv.writer(stack_file, delimiter=',')
        for agraph in test_agraph_list:
            stack = []
            for row in agraph._command_array:
                for i in np.nditer(row):
                    stack.append(i)
            stack_file_writer.writerow(stack)
    stack_file.close()


def write_constants(test_agraph_list):
    filename = '../bingocpp/app/test-agraph-consts.csv'
    with open(filename, mode='w+') as const_file:
        const_file_writer = csv.writer(const_file, delimiter=',')
        for agraph in test_agraph_list:
            consts = agraph._constants
            num_consts = len(consts)
            consts = np.insert(consts, 0, num_consts, axis=0)
            const_file_writer.writerow(consts)

    const_file.close()


def write_x_vals(test_x_vals):
    filename = '../bingocpp/app/test-agraph-x-vals.csv'
    with open(filename, mode='w+') as x_file:
        x_file_writer = csv.writer(x_file, delimiter=',')
        for row in test_x_vals:
            x_file_writer.writerow(row)
    x_file.close()

TEST_AGRAPHS = generate_random_individuals(100, 128)
TEST_X = generate_random_x(128)
TEST_AGRAPHS_CPP = copy_to_cpp(TEST_AGRAPHS)