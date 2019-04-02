# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np

from bingo.SymbolicRegression.AGraph.AGraphCrossover import AGraphCrossover
from bingo.SymbolicRegression.AGraph.AgraphMutation import AGraphMutation
from bingo.SymbolicRegression.AGraph.AGraphGenerator import AGraphGenerator
from bingo.SymbolicRegression.AGraph.ComponentGenerator import ComponentGenerator
from bingo.SymbolicRegression.ExplicitRegression import ExplicitRegression, ExplicitTrainingData

from bingo.Base.AgeFitnessEA import AgeFitnessEA
from bingo.Base.Evaluation import Evaluation
from bingo.Base.Island import Island
from bingo.Base.ContinuousLocalOptimization import ContinuousLocalOptimization

POP_SIZE = 100
STACK_SIZE = 10

def init_x_vals(start, stop, num_points):
    return np.linspace(start, stop, num_points).reshape([-1, 1])

def equation_eval(x):
    return x**2 + 3.5*x**3

def execute_generational_steps():
    x = init_x_vals(-10, 10, 100)
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
    evaluator = SimpleEvaluation(local_opt_fitness)

    ea = AgeFitnessEA(evaluator, agraph_generator, crossover,
                      mutation, 0.4, 0.4, POP_SIZE)

    island = Island(ea, agraph_generator, POP_SIZE)

    curr_fitness = 10e6
    i = 0
    while (curr_fitness > 10e-6):
        island.execute_generational_step()
        print("\nGeneration #", i)
        i += 1
        print("-"*80, "\n")
        report_max_min_mean_fitness(island.population)
        curr_fitness = island.best_individual().fitness

    print("Success!", island.best_individual().get_latex_string())

def report_max_min_mean_fitness(population):
    fitness = [indv.fitness for indv in population]
    print("Max fitness: \t", np.max(fitness))
    print("Min fitness: \t", np.min(fitness))
    print("Mean fitness: \t", np.mean(fitness))

def main():
    execute_generational_steps()

if __name__ == '__main__':
    main()

