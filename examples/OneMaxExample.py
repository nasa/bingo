"""
An example of bingo genetic optimization used to solve the one max problem.
"""
import numpy as np
from bingo.Base.VarOr import VarOr
from bingo.Base.FitnessFunction import FitnessFunction
from bingo.Base.Evaluation import Evaluation
from bingo.Base.TournamentSelection import Tournament
from bingo.Base.EvolutionaryAlgorithm import EvolutionaryAlgorithm
from bingo.Base.Island import Island
from bingo.Base.MultipleValues import MultipleValueChromosomeGenerator, \
                                 SinglePointCrossover, \
                                 SinglePointMutation

np.random.seed(0)  # used for reproducibility


def run_one_max_problem():
    generator = create_chromosome_generator()
    ev_alg = create_evolutionary_algorithm()

    island = Island(ev_alg, generator, population_size=10)
    display_best_individual(island)

    for _ in range(50):
        island.execute_generational_step()

    display_best_individual(island)


def create_chromosome_generator():
    return MultipleValueChromosomeGenerator(generate_0_or_1,
                                            values_per_chromosome=16)


def generate_0_or_1():
    """A function used in generation of values in individuals"""
    return np.random.choice([0, 1])


def create_evolutionary_algorithm():
    crossover = SinglePointCrossover()
    mutation = SinglePointMutation(generate_0_or_1)
    variation_phase = VarOr(crossover, mutation, crossover_probability=0.4,
                            mutation_probability=0.4)

    fitness = OneMaxFitnessFunction()
    evaluation_phase = Evaluation(fitness)

    selection_phase = Tournament(tournament_size=2)

    return EvolutionaryAlgorithm(variation_phase, evaluation_phase,
                                 selection_phase)


class OneMaxFitnessFunction(FitnessFunction):
    """Callable class to calculate fitness"""
    def __call__(self, individual):
        """Fitness = number of 0 elements in the individual's values"""
        self.eval_count += 1
        return individual.values.count(0)


def display_best_individual(island):
    best_individual = island.best_individual()
    print("Best individual: ", best_individual)
    print("Best individual's fitness: ", best_individual.fitness)


if __name__ == "__main__":
    run_one_max_problem()
