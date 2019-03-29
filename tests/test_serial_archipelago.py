# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import random
import copy

import pytest
import numpy as np

from bingo.MultipleValues import SinglePointCrossover, SinglePointMutation, \
                                 MultipleValueGenerator
from bingo.Island import Island
from bingo.EA.MuPlusLambda import MuPlusLambda
from bingo.EA.TournamentSelection import Tournament
from bingo.Base.Evaluation import Evaluation
from bingo.Base.FitnessFunction import FitnessFunction
from bingo.SerialArchipelago import SerialArchipelago

POP_SIZE = 5
SELECTION_SIZE = 10
VALUE_LIST_SIZE = 10
OFFSPRING_SIZE = 20

class MultipleValueFitnessFunction(FitnessFunction):
    def __call__(self, individual):
        fitness = np.count_nonzero(individual.list_of_values)
        self.eval_count += 1
        return len(individual.list_of_values) - fitness

def generate_three():
    return 3

def generate_two():
    return 2

def generate_one():
    return 1

def generate_zero():
    return 0

def mutation_function():
    return np.random.choice([False, True])

@pytest.fixture
def evol_alg():
    crossover = SinglePointCrossover()
    mutation = SinglePointMutation(mutation_function)
    selection = Tournament(SELECTION_SIZE)
    fitness = MultipleValueFitnessFunction()
    evaluator = Evaluation(fitness)
    return MuPlusLambda(evaluator, selection, crossover, mutation,
                        0.2, 0.4, OFFSPRING_SIZE)

@pytest.fixture
def zero_island(evol_alg):
    generator = MultipleValueGenerator(generate_zero, VALUE_LIST_SIZE)
    return Island(evol_alg, generator, POP_SIZE)

@pytest.fixture
def one_island(evol_alg):
    generator = MultipleValueGenerator(generate_one, VALUE_LIST_SIZE)
    return Island(evol_alg, generator, POP_SIZE)

@pytest.fixture
def two_island(evol_alg):
    generator = MultipleValueGenerator(generate_two, VALUE_LIST_SIZE)
    return Island(evol_alg, generator, POP_SIZE)

@pytest.fixture
def three_island(evol_alg):
    generator = MultipleValueGenerator(generate_three, VALUE_LIST_SIZE)
    return Island(evol_alg, generator, POP_SIZE)

@pytest.fixture
def island_list(zero_island, one_island, two_island, three_island):
    return [zero_island, one_island, two_island, three_island]

@pytest.fixture
def island(evol_alg):
    generator = MultipleValueGenerator(mutation_function, VALUE_LIST_SIZE)
    return Island(evol_alg, generator, POP_SIZE)


def test_archipelago_generated(island):
    archipelago = SerialArchipelago(island, num_islands=3)
    assert len(archipelago._islands) == 3
    for island_i in archipelago._islands:
        assert island_i != island
        assert island_i._population_size == island._population_size


def test_generational_step_executed(island):
    random.seed(0)
    archipelago = SerialArchipelago(island, num_islands=3)
    archipelago.step_through_generations(1)
    for island_i in archipelago._islands:
        assert island_i.best_individual()


def test_island_migration(one_island, island_list):
    archipelago = SerialArchipelago(one_island, num_islands=4)
    archipelago._islands = island_list 

    archipelago.coordinate_migration_between_islands()

    migration_count = 0
    for i, island in enumerate(archipelago._islands):
        initial_individual_values = [i]*VALUE_LIST_SIZE
        for individual in island.population:
            if initial_individual_values != individual.list_of_values:
                migration_count += 1
                break
    assert len(island_list) == migration_count