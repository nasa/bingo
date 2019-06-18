import timeit

import numpy as np

from bingo.SymbolicRegression.AGraph.AGraphCrossover import AGraphCrossover
from bingo.SymbolicRegression.AGraph.AGraphMutation import AGraphMutation
from bingo.SymbolicRegression.AGraph.AGraphGenerator import AGraphGenerator
from bingo.SymbolicRegression.AGraph.ComponentGenerator \
    import ComponentGenerator
from bingo.SymbolicRegression.ExplicitRegression import ExplicitRegression, \
                                                        ExplicitTrainingData
from bingo.Base.AgeFitnessEA import AgeFitnessEA
from bingo.Base.Evaluation import Evaluation
from bingo.Base.Island import Island
from bingo.Base.ContinuousLocalOptimization import ContinuousLocalOptimization
from performance_benchmarks import StatsPrinter

POP_SIZE = 128
STACK_SIZE = 64
MUTATION_PROBABILITY = 0.4
CROSSOVER_PROBABILITY = 0.4
NUM_POINTS = 100
START = -10
STOP = 10
ERROR_TOLERANCE = 10e-9
SEED = 20

def init_x_vals(start, stop, num_points):
    return np.linspace(start, stop, num_points).reshape([-1, 1])

def equation_eval(x):
    return x**2 + 3.5*x**3

def init_island():
    np.random.seed(15)
    x = init_x_vals(START, STOP, NUM_POINTS)
    y = equation_eval(x)
    training_data = ExplicitTrainingData(x, y)

    component_generator = ComponentGenerator(x.shape[1])
    component_generator.add_operator(2)
    component_generator.add_operator(3)
    component_generator.add_operator(4)

    crossover = AGraphCrossover(component_generator)
    mutation = AGraphMutation(component_generator)

    agraph_generator = AGraphGenerator(STACK_SIZE, component_generator)

    fitness = ExplicitRegression(training_data=training_data)
    local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm='lm')
    evaluator = Evaluation(local_opt_fitness)

    ea_algorithm = AgeFitnessEA(evaluator, agraph_generator, crossover,
                                mutation, MUTATION_PROBABILITY,
                                CROSSOVER_PROBABILITY, POP_SIZE)

    island = Island(ea_algorithm, agraph_generator, POP_SIZE)
    return island

TEST_ISLAND = init_island()

class IslandStatsPrinter(StatsPrinter):
    def __init__(self):
        super().__init__()
        self._output = ["-"*24+":::: REGRESSION BENCHMARKS ::::" + "-"*23,
                        self._header_format_string.format("NAME", "MEAN",
                                                          "STD", "MIN", "MAX"),
                        "-"*78]

def explicit_regression_benchmark():
    island = init_island()
    while island.get_best_individual().fitness > ERROR_TOLERANCE:
        island._execute_generational_step()

def do_benchmarking():
    printer = IslandStatsPrinter()
    printer.add_stats("Explicit Regression",
                      timeit.repeat(explicit_regression_benchmark,
                                    number=10,
                                    repeat=10))
    printer.print()

if __name__ == "__main__":
    do_benchmarking()