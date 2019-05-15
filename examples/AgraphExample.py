# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
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
import bingo.animation

POP_SIZE = 128
STACK_SIZE = 10
MUTATION_PROBABILITY = 0.4
CROSSOVER_PROBABILITY = 0.4
NUM_POINTS = 100
START = -10
STOP = 10
ERROR_TOLERANCE = 1e-6

def init_x_vals(start, stop, num_points):
    return np.linspace(start, stop, num_points).reshape([-1, 1])

def equation_eval(x):
    return x**2 + 3.5*x**3

def init_island():
    np.random.seed(10)
    x = init_x_vals(START, STOP, NUM_POINTS)
    y = equation_eval(x)
    training_data = ExplicitTrainingData(x, y)

    component_generator = ComponentGenerator(x.shape[1])
    component_generator.add_operator(2)
    component_generator.add_operator(3)
    component_generator.add_operator(4)

    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)

    agraph_generator = AGraphGenerator(STACK_SIZE, component_generator)

    fitness = ExplicitRegression(training_data=training_data)
    local_opt_fitness = ContinuousLocalOptimization(fitness, algorithm='lm')
    evaluator = Evaluation(local_opt_fitness)

    ea = AgeFitnessEA(evaluator, agraph_generator, crossover,
                      mutation, MUTATION_PROBABILITY,
                      CROSSOVER_PROBABILITY, POP_SIZE)

    island = Island(ea, agraph_generator, POP_SIZE)
    return island

TEST_ISLAND = init_island()

def main():
    test_island = TEST_ISLAND
    i = 0
    best_indv_values = []
    best_indv_values.append(test_island.best_individual().values)
    while test_island.best_individual().fitness > ERROR_TOLERANCE:
        test_island.execute_generational_step()
        best_indv_values.append(test_island.best_individual().values)
        i+=1

    bingo.animation.animate_data(best_indv_values)
    print("Generation: ", i)
    print("Success!", test_island.best_individual().get_latex_string())

def report_max_min_mean_fitness(population):
    fitness = [indv.fitness for indv in population]
    print("Max fitness: \t", np.max(fitness))
    print("Min fitness: \t", np.min(fitness))
    print("Mean fitness: \t", np.mean(fitness))

if __name__ == '__main__':
    main()
