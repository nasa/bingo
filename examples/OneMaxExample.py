import numpy as np

from bingo.Base.FitnessEvaluator import FitnessEvaluator
from bingo.Base.Crossover import Crossover
from bingo.Base.Variation import Variation
from bingo.Base.Evaluation import Evaluation
from bingo.Base.Selection import Selection
from bingo.EA.SimpleEa import SimpleEa
from bingo.EA.VarAnd import VarAnd
from bingo.EA.VarOr import VarOr
from bingo.EA.MuPlusLambda import MuPlusLambda
from bingo.EA.TournamentSelection import Tournament
from bingo.EA.SimpleEvaluation import SimpleEvaluation
from bingo.EA.BasicIsland import Island
from bingo.MultipleValues import *

class MultipleValueFitnessEvaluator(FitnessEvaluator):
    def __call__(self, individual):
        fitness = np.count_nonzero(individual.list_of_values)
        self.eval_count += 1
        return len(individual.list_of_values) - fitness

def mutation_onemax_specific():
    return np.random.choice([True, False])

def execute_generational_steps():
    crossover = SinglePointCrossover()
    mutation = SinglePointMutation(mutation_onemax_specific)
    var_or = VarOr(crossover, mutation, 0.4, 0.4)
    selection = Tournament(10)
    fitness = MultipleValueFitnessEvaluator()
    evaluator = SimpleEvaluation(fitness)
    ea = MuPlusLambda(var_or, evaluator, selection)
    generator = MultipleValueGenerator(mutation_onemax_specific, 10)
    island = Island(ea, generator, 25)
    for i in range(50):
        next_gen = island.execute_generational_step()
        print("\nGeneration #", i)
        print("----------------------\n")
        report_max_min_mean_fitness(next_gen)
        print("\npopulation: \n")
        for indv in next_gen:
            print(indv.list_of_values)

def report_max_min_mean_fitness(population):
    fitness = [indv.fitness for indv in population]
    print(fitness)
    print("Max fitness: \t", np.max(fitness))
    print("Min fitness: \t", np.min(fitness))
    print("Mean fitness: \t", np.mean(fitness))

execute_generational_steps()
