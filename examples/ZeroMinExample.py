# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np

from bingo.Base.FitnessFunction import FitnessFunction
from bingo.Base.MuPlusLambdaEA import MuPlusLambda
from bingo.Base.TournamentSelection import Tournament
from bingo.Base.Evaluation import Evaluation
from bingo.Base.Island import Island
from bingo.Base.ContinuousLocalOptimization import ContinuousLocalOptimization
from bingo.Base.MultipleValues import SinglePointCrossover, SinglePointMutation
from bingo.Base.MultipleFloats import MultipleFloatChromosomeGenerator


class ZeroMinFitnessFunction(FitnessFunction):
    def __call__(self, individual):
        return np.linalg.norm(individual.values)


def get_random_float():
    return np.random.random_sample() * 2.


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

    report_max_min_mean_fitness(island)
    island.evolve(500)
    report_max_min_mean_fitness(island)


def report_max_min_mean_fitness(island):
    fitness = [indv.fitness for indv in island.population]
    print("Max fitness: \t", np.max(fitness))
    print("Min fitness: \t", np.min(fitness))
    print("Mean fitness: \t", np.mean(fitness))


if __name__ == '__main__':
    main()
