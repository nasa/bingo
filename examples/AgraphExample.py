# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np

from bingo.AGraph.AGraphCrossover import AGraphCrossover
from bingo.AGraph.AgraphMutation import AGraphMutation
from bingo.AGraph.AGraphGenerator import AGraphGenerator
from bingo.AGraph.ComponentGenerator import ComponentGenerator
from bingo.ExplicitRegression import ExplicitRegression, ExplicitTrainingData
from bingo.EA.AgeFitnessEA import AgeFitnessEA

from bingo.EA.MuPlusLambda import MuPlusLambda
from bingo.EA.TournamentSelection import Tournament
from bingo.EA.SimpleEvaluation import SimpleEvaluation
from bingo.Island import Island
from bingo.Base.ContinuousLocalOptimization import ContinuousLocalOptimization


POP_SIZE = 25

def init_x_vals(start, stop, num_points):
    return np.linspace(start, stop, num_points).reshape([-1, 1])

def equation_eval(x):
    return x**2 + 3.5*x**3

def execute_generational_steps():
    x = init_x_vals(-10, 10, 100)
    y = equation_eval(x)
    training_data = ExplicitTrainingData(x, y)

    component_generator = ComponentGenerator(x.shape[0])
    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)

    agraph_generator = AGraphGenerator(10, component_generator)
    fitness = ExplicitRegression(training_data=training_data)
    local_opt_fitness = ContinuousLocalOptimization(fitness)
    evaluator = SimpleEvaluation(local_opt_fitness)

    selection = Tournament(10)
    ea = MuPlusLambda(evaluator, selection, crossover, mutation, 0.4, 0.4, 20)

    island = Island(ea, agraph_generator, POP_SIZE)
    for i in range(25):
        island.execute_generational_step()
        print("\nGeneration #", i)
        print("-"*80, "\n")
        report_max_min_mean_fitness(island.population)
        print("\npopulation: \n")
        for indv in island.population:
            print(["{0:.2f}".format(val) for val in indv.list_of_values])

def report_max_min_mean_fitness(population):
    fitness = [indv.fitness for indv in population]
    print("Max fitness: \t", np.max(fitness))
    print("Min fitness: \t", np.min(fitness))
    print("Mean fitness: \t", np.mean(fitness))

def main():
    execute_generational_steps()

if __name__ == '__main__':
    main()