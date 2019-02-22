import numpy as np

from Base.FitnessEvaluator import FitnessEvaluator
from Base.Crossover import Crossover
from Base.Variation import Variation
from bingo.Base.Evaluation import Evaluation
from bingo.Base.Selection import Selection
from bingo.EA.SimpleEa import SimpleEa
from bingo.EA.VarAnd import VarAnd
from bingo.EA.TournamentSelection import Tournament
from bingo.EA.SimpleEvaluation import SimpleEvaluation
from MultipleValues import *

class MultipleValueFitnessEvaluator(FitnessEvaluator):
    def __call__(self, individual):
        fitness = np.count_nonzero(individual.list_of_values)
        self.eval_count += 1
        return len(individual.list_of_values) - fitness

def mutation_onemax_specific():
    return np.random.choice([True, False])

def generate_population():
    generator = MultipleValueGenerator(mutation_onemax_specific, 10)
    population = [generator() for i in range(25)]
    return population

def execute_generational_steps():
    population = generate_population()
    selection = Tournament(10)
    crossover = SinglePointCrossover()
    mutation = SinglePointMutation(mutation_onemax_specific)
    fitness = MultipleValueFitnessEvaluator()
    evaluation = SimpleEvaluation(fitness)
    variation = VarAnd(crossover, mutation, 0.8, 0.8)
    ea = SimpleEa(variation, evaluation, selection)

    for i in range(10):
        next_gen = ea.generational_step(population)
        print("\nGeneration #", i)
        print("----------------------\n")
        report_max_min_mean_fitness(next_gen)
        print("\nparents: \n")
        for indv in population:
            print(indv.list_of_values)
        print("\noffspring: \n")
        for indv in next_gen:
            print(indv.list_of_values)
        population = next_gen

def report_max_min_mean_fitness(population):
    fitness = [indv.fitness for indv in population]
    print(fitness)
    print("Max fitness: \t", np.max(fitness))
    print("Min fitness: \t", np.min(fitness))
    print("Mean fitness: \t", np.mean(fitness))

execute_generational_steps()
