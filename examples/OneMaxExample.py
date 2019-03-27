# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np

from bingo.Base.FitnessFunction import FitnessFunction
from bingo.EA.MuPlusLambda import MuPlusLambda
from bingo.EA.TournamentSelection import Tournament
from bingo.Base.Evaluation import Evaluation
from bingo.Island import Island
from bingo.MultipleValues import MultipleValueChromosomeGenerator, \
                                 SinglePointCrossover, \
                                 SinglePointMutation


class OneMaxFitnessFunction(FitnessFunction):
    def __call__(self, individual):
        self.eval_count += 1
        return individual.list_of_values.count(0)


def generate_0_or_1():
    return np.random.choice([0, 1])


def execute_generational_steps():
    crossover = SinglePointCrossover()
    mutation = SinglePointMutation(generate_0_or_1)
    selection = Tournament(tournament_size=10)
    fitness = OneMaxFitnessFunction()
    evaluator = Evaluation(fitness)
    ea = MuPlusLambda(evaluator, selection, crossover, mutation,
                      crossover_probability=.4,
                      mutation_probability=.4,
                      number_offspring=20)
    generator = MultipleValueChromosomeGenerator(generate_0_or_1,
                                                 values_per_chromosome=10)
    island = Island(ea, generator, 25)
    for i in range(10):
        island.execute_generational_step()
        print("\nGeneration #", i)
        print("----------------------\n")
        report_max_min_mean_fitness(island.population)
        print("\npopulation: \n")
        for indv in island.population:
            print(indv.list_of_values)


def report_max_min_mean_fitness(population):
    fitness = [indv.fitness for indv in population]
    print(fitness)
    print("Max fitness: \t", np.max(fitness))
    print("Min fitness: \t", np.min(fitness))
    print("Mean fitness: \t", np.mean(fitness))


def main():
    execute_generational_steps()


if __name__ == '__main__':
    main()
