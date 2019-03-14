import numpy as np

from bingo.Base.FitnessEvaluator import FitnessEvaluator
from bingo.EA.MuPlusLambda import MuPlusLambda
from bingo.EA.TournamentSelection import Tournament
from bingo.EA.SimpleEvaluation import SimpleEvaluation
from bingo.Island import Island
from bingo.EA.FloatLocalOptimizationFunction import \
    FloatLocalOptimizationFunction
from bingo.MultipleValues import MultipleValueGenerator, SinglePointCrossover, \
                                 SinglePointMutation
from bingo.MultiValueContinuousLocalOptimization import \
    MultiValueContinuousLocalOptimization, \
    MultiValueContinuousLocalOptimizationGenerator

class MultipleFloatValueFitnessEvaluator(FitnessEvaluator):
    def __call__(self, individual):
        return np.linalg.norm(individual.list_of_values)

def get_random_float():
    return np.random.random_sample()

def execute_generational_steps():
    crossover = SinglePointCrossover()
    mutation = SinglePointMutation(get_random_float)
    selection = Tournament(10)
    fitness = MultipleFloatValueFitnessEvaluator()
    local_opt_fitness = FloatLocalOptimizationFunction(fitness)
    evaluator = SimpleEvaluation(local_opt_fitness)
    ea = MuPlusLambda(evaluator, selection, crossover, mutation, 0.4, 0.4, 20)
    generator = MultiValueContinuousLocalOptimizationGenerator(get_random_float, 10)
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
