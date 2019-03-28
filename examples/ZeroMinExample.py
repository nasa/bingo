# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np

from bingo.Base.FitnessFunction import FitnessFunction
from bingo.EA.MuPlusLambda import MuPlusLambda
from bingo.EA.TournamentSelection import Tournament
from bingo.Base.Evaluation import Evaluation
from bingo.Island import Island
from bingo.Base.ContinuousLocalOptimization import ContinuousLocalOptimization
from bingo.MultipleValues import SinglePointCrossover, SinglePointMutation
from bingo.MultipleFloats import MultipleFloatChromosomeGenerator


class ZeroMinFitnessFunction(FitnessFunction):
    def __call__(self, individual):
        return np.linalg.norm(individual.values)


def get_random_float():
    return np.random.random_sample()


def main():
    crossover = SinglePointCrossover()
    mutation = SinglePointMutation(get_random_float)
    selection = Tournament(10)
    fitness = ZeroMinFitnessFunction()
    local_opt_fitness = ContinuousLocalOptimization(fitness)
    evaluator = Evaluation(local_opt_fitness)
    ea = MuPlusLambda(evaluator, selection, crossover, mutation, 0.4, 0.4, 20)
    generator = MultipleFloatChromosomeGenerator(get_random_float, 8)
    island = Island(ea, generator, 25)
    for i in range(25):
        island.execute_generational_step()
        print("\nGeneration #", i)
        print("-"*80, "\n")
        report_max_min_mean_fitness(island.population)
        print("\npopulation: \n")
        for indv in island.population:
            print(["{0:.2f}".format(val) for val in indv.values])


def report_max_min_mean_fitness(population):
    fitness = [indv.fitness for indv in population]
    print("Max fitness: \t", np.max(fitness))
    print("Min fitness: \t", np.min(fitness))
    print("Mean fitness: \t", np.mean(fitness))


if __name__ == '__main__':
    main()
