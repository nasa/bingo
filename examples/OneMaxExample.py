import numpy as np
from bingo.EA.VarOr import VarOr
from bingo.Base.FitnessFunction import FitnessFunction
from bingo.Base.Evaluation import Evaluation
from bingo.EA.TournamentSelection import Tournament
from bingo.Base.EvolutionaryAlgorithm import EvolutionaryAlgorithm
from bingo.Island import Island
from bingo.MultipleValues import MultipleValueChromosomeGenerator, \
                                 SinglePointCrossover, \
                                 SinglePointMutation


class OneMaxFitnessFunction(FitnessFunction):
    """Callable class to calculate fitness"""
    def __call__(self, individual):
        """Fitness = number of 0 elements in the individual's values"""
        self.eval_count += 1
        return individual.values.count(0)


def generate_0_or_1():
    """A function used in generation of values in individuals"""
    return np.random.choice([0, 1])


# Define an object used in generating chromosomes in the population
generator = MultipleValueChromosomeGenerator(generate_0_or_1,
                                             values_per_chromosome=10)

# Evolutionary Algorithms in bingo have 3 phases
# Variation phase: often a utilizes mutation and crossover
crossover = SinglePointCrossover()
mutation = SinglePointMutation(generate_0_or_1)
variation_phase = VarOr(crossover, mutation,
                        crossover_probability=0.4,
                        mutation_probability=0.4)

# Evaluation phase: defines fitness
fitness = OneMaxFitnessFunction()
evaluation_phase = Evaluation(fitness)

# Selection phase: how to select survivors into next generation
selection_phase = Tournament(tournament_size=10)

ev_alg = EvolutionaryAlgorithm(variation_phase,
                               evaluation_phase,
                               selection_phase)


# An Island is the fundamental unit of genetic algorithms in bingo. It is
# responsible for generating and evolving a population using a chromosome
# generator and evolutionary algorithm
island = Island(ev_alg, generator, population_size=25)
best_individual = island.best_individual()
print("Best individual at start: ", best_individual)
print("Best individual's fitness: ", best_individual.fitness)

# Evolve the population for 10 generations
for _ in range(10):
    island.execute_generational_step()

# Show the new best individual
best_individual = island.best_individual()
print("Best individual at end: ", best_individual)
print("Best individual's fitness: ", best_individual.fitness)
